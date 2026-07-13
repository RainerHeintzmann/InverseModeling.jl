"""
    This example refers to a Ptychography scheme, where the setup is 
    laser -> sample -> diffusor(x-x0, y-y0) -> detector
    then several intensity images are acquired on the detector for various (x0,y0) shifted positions of the diffusor 

Installation: 
] add https://github.com/RainerHeintzmann/InverseModeling.jl
] add FourierTools PointSpreadFunctions NDTools InverseModeling Statistics TestImages IndexFunArrays Noise Plots
 if you want to visualize the result and run this on an NVidia GPU, you also need
] add View5D CUDA
"""

using FourierTools
using PointSpreadFunctions
using NDTools
using InverseModeling
using Statistics
using View5D
using TestImages
using IndexFunArrays
using Noise
using CUDA
# using Zygote
using Optimisers
using Plots

# function get_apsf(sz, z, )
# endEE

function simulate_data(obj, prop_od, prop_ds, diffusor)
    atdiffusor = conv_psf(obj, prop_od, (1,2))
    atdiffusor = atdiffusor .* diffusor
    atsensor = conv_psf(atdiffusor, prop_ds, (1,2))
    abs2.(atsensor)
end

"""
    get_diffusors(sz, screen, shifts)

returns shifted diffusors
"""
function get_diffusors(sz, screen, shifts)
    tmp = Array(Int.(shifts))
    d = ndims(tmp) == 2 ? 1 : ndims(tmp)
    slices = [screen[start[1]:start[1]+sz[1]-1,start[2]:start[2]+sz[2]-1] for start in eachslice(tmp, dims=d)]
    return stack(slices; dims=3)
end

function calc_diffusor_mean(nimg, shifts, bigsz)
    screen = similar(nimg, ComplexF32, bigsz); screen.=0;
    mysum = similar(nimg, Float32, bigsz); mysum.=0;
    tmp = Array(Int.(shifts))
    sz = size(nimg)[1:2]
    snum=1
    d = ndims(tmp) == 2 ? 1 : ndims(tmp)
    for start in eachslice(tmp, dims=d)
        screen[start[1]:start[1]+sz[1]-1,start[2]:start[2]+sz[2]-1] .+= nimg[:,:,snum]
        mysum[start[1]:start[1]+sz[1]-1,start[2]:start[2]+sz[2]-1] .+= 1
        snum += 1
    end
    mymask = mysum.>0
    screen[mymask] .= screen[mymask] ./ mysum[mymask]
    return screen 
end

function calc_diffusor_sum(nimg, shifts, bigsz)
    screen = similar(nimg, ComplexF32, bigsz); screen.=0;
    tmp = Array(Int.(shifts))
    sz = size(nimg)[1:2]
    snum=1
    d = ndims(tmp) == 2 ? 1 : ndims(tmp)
    for start in eachslice(tmp, dims=d) 
        screen[start[1]:start[1]+sz[1]-1,start[2]:start[2]+sz[2]-1] .+= nimg[:,:,snum]
        snum += 1
    end
    return screen 
end

"""
    fwd_model(params)

a scattering forward model:
obj -> atdiffusor * diffusor-> detection  
Note that the first part is intenstionally ignored and the result therefore needs a nother back propagation step

"""
function fwd_model(params, prop_ds, obj_size)
    screen = params(:screen)
    shifts = params(:shifts)

    diffusor = get_diffusors(obj_size, screen, shifts)

    atdiffusor = params(:obj) # the object is already assumed to have previously been forward propagted to the diffusor
    atdiffusor = atdiffusor .* diffusor
    # use conv_psf (no pre-planned FFT) so arbitrary batch sizes work
    atsensor = conv_psf(atdiffusor, prop_ds, (1,2))
    abs2.(atsensor)  # return the stack of intensity images at the sensor
end

"""
    epie(start_params, measured, iterations=100; alpha=1.0, beta=1.0, global_alpha=true, scale_method=maximum)

an implementation of the ePIE algorithm for the Ptychography problem at hand.
The algorithm works by iterating on the diffusor plane only and the back-propagating the final result.
# arguments
+ start_params: A named Tuple of the arguments containing the start variables of the iterative updates
    + :obj: the start for the object (emitted amplitude before propagation to the diffusor)
    + :prop_od, :prop_ds:  The real space propagators (ampitude spread functions) to be convolved with the object to propagate from object to diffusor and diffusor to sensor
    + :screen: The (bigger) diffusor screen out of which the varous diffusor subarrays are extract (see :shifts)
    + :shifts: The start positions at which to extrac to diffusor 
+ meassured: The measured data
+ iterations: number of iterations (default: 100)
+ alpha: the Overrelaxition factor ("learning rate") for the object
+ beta: the Overrelaxition factor ("learning rate") for the diffusor
+ global_alpha: determins whether the alpha update (for obj) is using the previous screen or individual diffusor screens 
+ scale_method: this function is used for alpha and beta (default: max)
"""
function epie(start_params, measured, iterations=100; α=1.0, β=1.0, fix_obj=false, fix_screen=false, global_alpha=true, scale_method=maximum)
    α = Float32(α)
    β = Float32(β)
    # obj = start_params[:obj].+0
    # prop_od = start_params[:prop_od].+0
    # conj_prop_od = ifft(conj.(fft(prop_od))) .+ 0
    # prop_ds = start_params[:prop_ds].+0    
    conj_prop_ds = ifft(conj.(fft(prop_ds)))
    screen = start_params[:screen].+0    
    shifts = start_params[:shifts]
    #obj_at_diffusor = conv_psf(obj, prop_od, (1,2))
    obj_at_diffusor = start_params[:obj].+0
    sqrt_measured = sqrt.(max.(0, measured))
    batchsize = size(measured, 3)
    losses = []
    p_ds = nothing
    otf = nothing
    conj_otf = nothing
    for n in 0:iterations  # screen and obj_at_diffusor are the variable which are updated in each round
        diffusor = get_diffusors(size(obj_at_diffusor)[1:2], screen, shifts)
        all_at_diffusor = obj_at_diffusor .* diffusor
        if (n==0)
            otf, p_ds = plan_conv_psf_buffer(all_at_diffusor, start_params[:prop_ds], (1,2))
            conj_otf = conj.(otf)
        end
        all_at_sensor = p_ds(all_at_diffusor, otf)  # conv_psf(all_at_diffusor, prop_ds, (1,2))
        # measure loss (diagnostics only)
        loss = sum(abs2.(sqrt_measured .- abs.(all_at_sensor))) # /batchsize
        println("iteration $(n), loss $(loss)")
        if (n>=1)
            push!(losses, loss)
        end
        # enforce the magniture constrain (keep phase all_at_sensor)
        all_at_sensor .= sqrt_measured .* cis.(angle.(all_at_sensor)) #  sqrt_measured./abs.()
        corrected_atdiffusor = p_ds(all_at_sensor, conj_otf) # conv_psf(all_at_sensor, conj_prop_ds, (1,2))
        # apply updates
        mybeta = (β/batchsize) ./ scale_method(abs2.(obj_at_diffusor), dims=(1,2))
        # println("max beta: $(scale_method(abs2.(obj_at_diffusor), dims=(1,2)))")
        # println("iteration: $n, mybeta: $(mybeta)")
        cor_diffusor = conj.(obj_at_diffusor).*(corrected_atdiffusor .- all_at_diffusor)
        if !(fix_obj)
            if (global_alpha)
                myalpha =  (α/batchsize) ./scale_method(abs2.(screen))
                # println("iteration: $n, myalpha screen: $(myalpha)")
                # println("max alpha: $(scale_method(abs2.(screen)))")
                # @vt myalpha .* sum(conj.(diffusor).*(corrected_atdiffusor .- all_at_diffusor), dims=3)
                obj_at_diffusor .+= myalpha .* sum(conj.(diffusor).*(corrected_atdiffusor .- all_at_diffusor), dims=3)
            else
                myalpha = (α/batchsize)./scale_method(abs2.(diffusor), dims=(1,2))
                # println("iteration: $n, myalpha diffusor: $(myalpha)")
                obj_at_diffusor .+= myalpha .* sum(conj.(diffusor).*(corrected_atdiffusor .- all_at_diffusor), dims=3)
            end
        end
        if !(fix_screen)
            # @vtp mybeta .* calc_diffusor_sum(cor_diffusor, shifts, size(screen))
            screen .+= mybeta .* calc_diffusor_sum(cor_diffusor, shifts, size(screen))
        end
    end
    # obj = conv_psf(obj_at_diffusor, conj_prop_od, (1,2))
    return (obj=obj_at_diffusor, screen=screen), losses
end

"""
    get_diffusor(sz; σ_mag=1.0, σ_phase=1000.0, fmax = 0.1)

returns a single diffusor screen
"""
function get_diffusor(sz; σ_mag=0.8, σ_phase=0.5, fmax = 0.1)
    bigsz = 2 .*sz
    amp = (1 .+ σ_mag.*randn((bigsz)...)).* cis.(2π.*σ_phase.*rand(bigsz...))
    return ComplexF32.(ift(ft(amp).* disc(bigsz, fmax; scale=ScaFT)))
end


function show_it(res, obj, v=nothing; show_gt=false, conj_prop_od=nothing)
    res = Array(res)
    if !(isnothing(conj_prop_od))
        res = conv_psf(res, conj_prop_od, (1,2)) # first back-propagate to original object position
    end
    relphase_obj = sum(conj.(Array(res)) .* obj)  # to adjust global phase for display
    if (show_gt)
        v = @vp obj  # starts a new viewer
    end
    view5d(res .*(relphase_obj/abs(relphase_obj)), v; mode=DisplAddTime, show_phase=true)
    set_gamma(1.0)
    return v
end

function main()
## simulate data
    mag = Float32.(testimage("resolution_test_512")) .+ 0.5f0
    mag = filter_gaussian(mag, 1.5)
    # mag = select_region(mag, (256, 256))
    # scansz = (10,10)  # (10,10) scan -> 100 images
    scansz = (10, 10)  # (5,5) scan -> 25 images
    scansz = (5, 5)  # (5,5) scan -> 25 images

    phase = 2π .* mag[:,end:-1:1] .- π
    obj = ComplexF32.(mag .* cis.(phase))

    # v = @vp obj
    # set_gamma(1.0)

    n = 1
    λ = 0.661

    if (true)
        k0 =  1 #./abs2(λ/pixelsize)
        pixelsize = 2.2
        # osz = (1024,1024) # size(obj)
        osz = size(obj)
        qxy2 = rr2(osz, scale=ScaFT)./abs2(pixelsize/λ)
        dz_od = 1535
        dz_ds = 2250
        mypropator_od = exp.(-1im*2pi .* (dz_od/λ) .* sqrt.(max.(0, abs2(k0) .- qxy2)))
        mypropator_ds = exp.(-1im*2pi .* (dz_ds/λ) .* sqrt.(max.(0, abs2(k0) .- qxy2)))
        prop_od = ComplexF32.(ift(mypropator_od))
        prop_ds = ComplexF32.(ift(mypropator_ds))
    else
        NA = 0.15
        # NA = 0.30
        # pp = PSFParams(λ, NA, n; pol=pol_scalar, method=PointSpreadFunctions.MethodPropagate)
        pp = PSFParams(λ, NA, n; pol=pol_x)
        samp_od = (pixelsize, pixelsize, -dz_od)
        samp_ds = (pixelsize, pixelsize, -dz_ds)
        prop_od = collect(apsf((size(obj)..., 3), pp; sampling=samp_od)[:,:,3,1:1])
        prop_ds = collect(apsf((size(obj)..., 3), pp; sampling=samp_ds)[:,:,3,1:1])
    end
    conj_prop_od = ifft(conj.(fft(prop_od))) .+ 0

    # @vt prop_od prop_ds

    obj_at_diffusor = conv_psf(obj, prop_od)

    wiggle = 0.5
    scanmag = 20.0
    ctr(x) = round.(Float32, (scanmag.*Vector(x .- (scansz .÷ 2 .+1) .+ wiggle.*rand(2))))
    shifts_r = (ctr.(Tuple.(CartesianIndices(scansz))))[:]
    screen = get_diffusor(size(obj); σ_mag=0.2, σ_phase=0.4, fmax=0.3);
    midpos = [(size(screen).÷2 .+1)...] .- [(size(obj).÷2 .+1)...]
    shifts = permutedims(cat(shifts_r..., dims=2) .+ midpos, (2,1))
    start_val = (obj=obj_at_diffusor, screen=Fixed(screen), shifts=Fixed(shifts))

    start_vals, fixed_vals, forward, backward, get_fit_results = create_forward((x)->fwd_model(x, prop_ds, size(obj)), start_val)

    pimg = forward(start_vals)
    nphotons = 10000;
    nfac = nphotons./mean(pimg)
    # nimg = Float32.(pimg.*nfac)
    nimg = Float32.(poisson(pimg.*nfac))
    # normalize the data 
    nfac2 = 1/maximum(nimg)

    nimg .*= nfac2
    obj = obj .*sqrt(nfac*nfac2)

    # start_val = (obj=const_obj, prop_ds=Fixed(prop_ds), diffusor=Fixed(diffusor))
 
# --------------- simulation done, lets solve the inverse problem -------

    # just a const value:
    const_obj = ComplexF32.(nimg[:,:,1:1] .* 0 .+ sqrt.(mean(nimg)))[:,:,1] 
    # const_obj = ComplexF32.(nimg[:,:,1:1] .* 0 .+ mean(sqrt.(nimg), dims=3))
    const_obj = ComplexF32.(sqrt.(mean(nimg, dims=3)))[:,:,1] # Rahel's 

    # start_val = (obj=const_obj, prop_ds=Fixed(prop_ds), screen=Fixed(screen), shifts=Fixed(shifts))

    # just a const phase diffusor
    const_screen = screen .* 0 .+ 0.1
    # estimate the diffusor from unshifted data
    const_screen = sqrt.(calc_diffusor_mean(nimg, shifts, size(screen))) ./ mean(const_obj) # size(shifts,1)
    # start_val = (obj=to_cu(const_obj), prop_ds=Fixed(to_cu(prop_ds)), screen=const_screen)

    use_cuda = true # true
    # use_cuda = true
    to_cu = (x)-> (use_cuda) ? cu(x) : x
    optional_reclaim = () -> (use_cuda) ?  CUDA.reclaim() : nothing

    torecon = to_cu(nimg)
    start_val_obj = (obj=to_cu(const_obj), screen=Fixed(to_cu(screen)), shifts=Fixed(to_cu(shifts)))
    # optimize only the diffusor:
    start_val_diffusor = (obj=Fixed(to_cu(obj_at_diffusor)), screen=to_cu(const_screen), shifts=Fixed(to_cu(shifts)))
    # optimize both obj and diffusor:
    start_val_both = (obj=to_cu(const_obj), screen=to_cu(const_screen), shifts=Fixed(to_cu(shifts)))


## reconstruct data now
    # here is the classical epie algorithm with the batchsize equal to the full measurement
    # To implement alternating direction methods, we need to copy data between the iterations
    optional_reclaim()
   
    α = 4f0; β=4f0
    N = 200
    start_val_epie_both = (obj=to_cu(const_obj), prop_ds=to_cu(prop_ds), screen=to_cu(const_screen), shifts=to_cu(shifts))
    # ground truth values for debugging only:
    # start_val_epie = (obj=to_cu(obj), prop_ds=to_cu(prop_ds), screen=to_cu(screen), shifts=to_cu(shifts))
    CUDA.@time res_epie_both, myloss_both_epie = epie(start_val_epie_both, torecon, N; α=α, β=β, global_alpha=true)    
    plot(myloss_both_epie, yaxis=:log10, label="both epie", xlabel="iterations", ylabel="Anscombe Loss")
    plot!(myloss_both_epie, yaxis=:log10, label="both epie")

    start_val_epie_screen = (obj=to_cu(obj_at_diffusor), prop_ds=to_cu(prop_ds), screen=to_cu(const_screen), shifts=to_cu(shifts))
    res_epie_screen, losses_epie_screen = epie(start_val_epie_screen, torecon, N; α=α, β=β, fix_obj=true)
    plot!(losses_epie_screen, yaxis=:log10, label="screen epie")

    start_val_epie_obj = (obj=to_cu(const_obj), prop_ds=to_cu(prop_ds), screen=to_cu(screen), shifts=to_cu(shifts))
    res_epie_obj, losses_epie_obj = epie(start_val_epie_obj, torecon, N; α=α, β=β, fix_screen=true)
    plot!(losses_epie_obj, yaxis=:log10, label="obj epie")
    # @vtp res_epie_both[:obj] res_epie_screen[:obj] res_epie_obj[:obj]

    # start_vals, fixed_vals, forward, backward, get_fit_results = create_forward(fwd_conv, start_val)
    # optim_res = InverseModeling.optimize(loss(nimg, forward), start_vals, iterations=80);
    optional_reclaim()
    # optimizer = InverseModeling.LBFGS()
    # optimizer = InverseModeling.GradientDescent()
    optimizer = InverseModeling.GradientDescent(; alphaguess=α)
 
    my_fwd= (x) -> fwd_model(x, to_cu(prop_ds), size(obj))
    @time res_obj, myloss_obj = optimize_model(start_val_obj, my_fwd, torecon, loss_anscombe; iterations=N, optimizer=optimizer)
    # @time res1, myloss1 = optimize_model(start_val, my_fwd, torecon, loss_anscombe; iterations=200, optimizer=GradientDescent())

    optional_reclaim()
    @time res_diffusor, myloss_diffusor = optimize_model(start_val_diffusor, my_fwd, torecon, loss_anscombe; iterations=N, optimizer=optimizer)

    optional_reclaim()
    # alpha = 0.01
    # optimizer = InverseModeling.GradientDescent(; alphaguess=alpha)
    optimizer = InverseModeling.LBFGS()
    @time res_both, myloss_both = optimize_model(start_val_both, my_fwd, torecon, loss_anscombe; iterations=N, optimizer=optimizer)

    optimizer = InverseModeling.LBFGS()
    @time res_both_sqrt, myloss_both_sqrt = optimize_model(start_val_both, my_fwd, torecon, loss_sqrt_anscombe; iterations=N, optimizer=optimizer)

    optimizer = InverseModeling.ConjugateGradient()
    optional_reclaim()
    @time res_both_cg, myloss_both_cg = optimize_model(start_val_both, my_fwd, torecon, loss_anscombe; iterations=N, optimizer=optimizer)

"""
    P!(vals, G, α=1.0, β=1.0)

a precoditioner to make the steepest gradient descent algorithm identical to ePIE
The Preconditioner is applied to each calculated gradient before performing the update step.
"""
# myprop = to_cu(prop_od)
    function P!(vals, G, α=α, β=β)  # 1.5
        # diffusor = get_diffusors(size(obj_at_diffusor)[1:2], screen, shifts)
        # @show keys(vals)
        obj_at_diffusor = vals[:obj] # conv_psf(vals[:obj], myprop)
        # println("max beta: $(maximum(abs2.(obj_at_diffusor)))")
        beta_screen = β/maximum(abs2.(obj_at_diffusor));
        G[:screen] .*=  beta_screen 
        # println("max alpha: $(maximum(abs2.(vals[:screen])))")
        alpha_obj = α/maximum(abs2.(vals[:screen]))
        G[:obj] .*= alpha_obj 
        # println("alpha_obj: $(alpha_obj), beta_screen: $(beta_screen)")
    end

    optional_reclaim()
    CUDA.@time res_both_gd, myloss_both_gd = optimize_model(start_val_both, my_fwd, torecon, loss_anscombe; iterations=N, optimize=steepest_decent_optimizer(Float32(1/size(nimg,3))/2; P! = P!, verbose=true)) # 0.05
    plot!(myloss_both_gd, label="gradient descent", linestyle = :dash)

    @time res_both_gd2, myloss_both_gd2 = optimize_model(start_val_both, my_fwd, torecon, loss_anscombe; iterations=N, optimize=steepest_decent_optimizer((obj=4f0/2/size(nimg,3),screen=4f0/2/size(nimg,3)), norm_vars=[:obj, :screen], verbose=true)) # 0.05
    plot!(myloss_both_gd2, label="gradient descent 2", linestyle = :dashdot)

    # ── SGD with mini-batches (via Optimisers.jl) ──
    # shifts must be (2, 1, n_frames) so the framework slices along batch_dim=3
    shifts_batch = reshape(permutedims(shifts), 2, 1, :)
    start_val_sgd = (obj=start_val_both.obj, screen=start_val_both.screen, shifts=Fixed(to_cu(shifts_batch)))
    batch_size = 5
    @time res_sgd, loss_sgd = optimize_model(
        start_val_sgd, my_fwd, torecon, loss_anscombe;
        iterations=N/5, batch_size=batch_size, optimizer=Adam(0.05f0), verbose=true)

    plot!(loss_sgd, label="batched sgd", linestyle = :dashdot)
        
##
    plot(myloss_obj, yaxis=:log, xlabel="iteration", ylabel="Anscombe loss", label="fixed screen")
    # plot!(myloss_obj, label="fixed screen, obj structured")
    plot!(myloss_diffusor, label="only diffusor")
    plot!(myloss_both, label="both LBFGS")
    plot!(abs2.(myloss_both_sqrt), label="both sqrt Anscombe")
    plot!(myloss_both_cg, label="both conj. grad.")
    plot!(myloss_both_gd, label="both grad. desc.")
    plot!(myloss_both_epie, label="both epie")
    plot!(loss_sgd, label="SGD Adam (batch_size=$(batch_size))", linestyle = :dashdot)
    
##    
    # @vtp res_epie[:obj]
    # @vtp res_epie[:screen]
    vo = show_it(res_obj[:obj], obj; show_gt=true)
    show_it(res_both[:obj], obj, vo)
    show_it(res_both[:obj], obj, vo, conj_prop_od=conj_prop_od)
    show_it(res_both_gd[:obj], obj, vo, conj_prop_od=conj_prop_od)
    show_it(res_epie[:obj], obj, vo, conj_prop_od=conj_prop_od)

    vd = show_it(res_diffusor[:screen], screen; show_gt=true)
    show_it(res_both[:screen], screen, vd)
    show_it(res_both_gd[:screen], screen, vd)
    show_it(res_both_epie[:screen], screen, vd)
    
##    
end


