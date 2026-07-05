# ] add https://github.com/RainerHeintzmann/InverseModeling.jl
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

# function get_apsf(sz, z, )
# end

function simulate_data(obj, prop_od, prop_ds, diffusor)
    atdiffusor = conv_psf(obj, prop_od, (1,2))
    atdiffusor = atdiffusor .* diffusor
    atsensor = conv_psf(atdiffusor, prop_ds, (1,2))
    abs2.(atsensor)
end

function fwd_model(params)
    obj = params(:obj)
    prop_od = params(:prop_od)
    prop_ds = params(:prop_ds)
    diffusor = params(:diffusor)

    atdiffusor = conv_psf(obj, prop_od, (1,2))
    atdiffusor = atdiffusor .* diffusor
    atsensor = conv_psf(atdiffusor, prop_ds, (1,2))
    abs2.(atsensor)
end


function get_diffusor(sz, shifts; σ_mag=1.0, σ_phase=1000.0, fmax = 0.1)
    bigsz = 2 .*sz
    amp = (1 .+ σ_mag.*randn((bigsz)...)).* cis.(2π.*rand(bigsz...))
    screen = ComplexF32.(ift(ft(amp).* disc(bigsz, fmax; scale=ScaFT)))
    diffusor=zeros(eltype(screen), sz..., length(shifts))
    for (slice, myshift) in zip(eachslice(diffusor, dims=3), shifts)
        slice .= select_region_view(shift(screen, myshift), sz)
    end
    return diffusor
end

function main()
    mag = Float32.(testimage("resolution_test_512"))
    phase = 2π .* mag[:,end:-1:1] .- π
    obj = mag .* cis.(phase)

    v = @vp obj
    set_gamma(1.0)

    λ=0.661
    NA = 0.05
    n = 1
    pp = PSFParams(λ, NA, n; pol=pol_x)
    samp_od = (2,2, 1535)
    samp_ds = (2,2, 2250)

    prop_od = apsf((size(obj)..., 3), pp; sampling=samp_od)[:,:,3,1]
    prop_ds = apsf((size(obj)..., 3), pp; sampling=samp_ds)[:,:,3,1]
    # @vt prop_od prop_ds

    scansz = (10,10)  # (10,10)

    wiggle = 0.5
    scanmag = 20.0
    ctr(x) = scanmag.*Tuple(x .- (scansz .÷ 2 .+1) .+ wiggle.*rand(2))
    shifts = ctr.(Tuple.(CartesianIndices(scansz)))
    diffusor = get_diffusor(size(obj), shifts[:]);
    # @vp diffusor

    # int_sensor = simulate_data(obj, prop_od, prop_ds, diffusor)

    # @vt int_sensor
# --------------- simulation done, lets solve the inverse problem -------

    start_val = (obj=obj, prop_od=Fixed(prop_od), prop_ds=Fixed(prop_ds), diffusor=Fixed(diffusor))
    start_vals, fixed_vals, forward, backward, get_fit_results = create_forward(fwd_model, start_val)

    pimg = forward(start_vals)
    nphotons = 100;
    nimg = Float32.(poisson(pimg./mean(pimg) .* nphotons))

    const_obj = ComplexF32.(nimg[:,:,1:1] .* 0 .+ mean(sqrt.(nimg)))
    start_val = (obj=const_obj, prop_od=Fixed(prop_od), prop_ds=Fixed(prop_ds), diffusor=Fixed(diffusor))
 
    use_cuda = true
    # cu = (x) -> CuArray(x)
    if (use_cuda)
        start_val = (obj=cu(const_obj), prop_od=Fixed(cu(prop_od)), prop_ds=Fixed(cu(prop_ds)), diffusor=Fixed(cu(diffusor)))        
        torecon = cu(nimg)
    else
        torecon = nimg
    end

    # start_vals, fixed_vals, forward, backward, get_fit_results = create_forward(fwd_conv, start_val)
    # optim_res = InverseModeling.optimize(loss(nimg, forward), start_vals, iterations=80);
    @time res1, myloss1 = optimize_model(start_val, fwd_model, torecon; iterations=20)

    res = res1[:obj]
    @vtp res 
    @vtp obj


end
