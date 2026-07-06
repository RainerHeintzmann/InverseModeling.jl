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

"""
    get_diffusors(screen, shifts)

returns shifted diffusors
"""
function get_diffusors(sz, screen, shifts)
    midpos = size(screen).÷2 .+1
    shifted_screen = []
    for myshift in eachslice(shifts, dims=1)
        push!(shifted_scree, select_region(screen, sz; center=ft_center_diff = midpos .+ Tuple(round.(Int, myshift))))
    end
    return cat(shifted_screen..., dims=3)
    # not working in Zygote either:
    # return cat((select_region(screen, sz; center=ft_center_diff = midpos .+ Tuple(round.(Int, myshift))) for myshift in eachslice(shifts, dims=1))..., dims=3)
    # return cat((select_region_view(shift(screen, myshift), sz) for myshift in shifts)..., dims=3)
end

function fwd_model(params)
    obj = params(:obj)
    prop_od = params(:prop_od)
    prop_ds = params(:prop_ds)
    screen = params(:screen)
    shifts = params(:shifts)

    diffusor = get_diffusors(size(obj)[1:2], screen, shifts)

    atdiffusor = conv_psf(obj, prop_od, (1,2))
    atdiffusor = atdiffusor .* diffusor
    atsensor = conv_psf(atdiffusor, prop_ds, (1,2))
    abs2.(atsensor)  # return the intensity at the sensor
end

function get_diffusor(sz; σ_mag=1.0, σ_phase=1000.0, fmax = 0.1)
    bigsz = 2 .*sz
    amp = (1 .+ σ_mag.*randn((bigsz)...)).* cis.(2π.*rand(bigsz...))
    return ComplexF32.(ift(ft(amp).* disc(bigsz, fmax; scale=ScaFT)))
end


function main()
    mag = Float32.(testimage("resolution_test_512"))
    mag = select_region(mag, (256, 256))
    # scansz = (10,10)  # (10,10) scan -> 100 images
    scansz = (5,5)  # (5,5) scan -> 25 images

    phase = 2π .* mag[:,end:-1:1] .- π
    obj = ComplexF32.(mag .* cis.(phase))

    # v = @vp obj
    # set_gamma(1.0)

    λ=0.661
    NA = 0.05
    n = 1
    pp = PSFParams(λ, NA, n; pol=pol_x)
    samp_od = (2,2, 1535)
    samp_ds = (2,2, 2250)

    prop_od = apsf((size(obj)..., 3), pp; sampling=samp_od)[:,:,3,1]
    prop_ds = apsf((size(obj)..., 3), pp; sampling=samp_ds)[:,:,3,1]
    # @vt prop_od prop_ds


    wiggle = 0.5
    scanmag = 20.0
    ctr(x) = Float32.(scanmag.*Vector(x .- (scansz .÷ 2 .+1) .+ wiggle.*rand(2)))
    shifts_r = (ctr.(Tuple.(CartesianIndices(scansz))))[:]
    shifts = permutedims(cat(shifts_r..., dims=2), (2,1))
    screen = get_diffusor(size(obj));
    # diffusor = get_diffusors(size(obj), screen, shifts)

    # @vp diffusor

    # int_sensor = simulate_data(obj, prop_od, prop_ds, diffusor)

    # @vt int_sensor

    start_val = (obj=obj, prop_od=Fixed(prop_od), prop_ds=Fixed(prop_ds), screen=Fixed(screen), shifts=Fixed(shifts))
    start_vals, fixed_vals, forward, backward, get_fit_results = create_forward(fwd_model, start_val)

    pimg = forward(start_vals)
    nphotons = 100;
    nimg = Float32.(poisson(pimg./mean(pimg) .* nphotons))

    # start_val = (obj=const_obj, prop_od=Fixed(prop_od), prop_ds=Fixed(prop_ds), diffusor=Fixed(diffusor))
 
# --------------- simulation done, lets solve the inverse problem -------

    const_obj = ComplexF32.(nimg[:,:,1:1] .* 0 .+ mean(sqrt.(nimg)))

    start_val = (obj=const_obj, prop_od=Fixed(prop_od), prop_ds=Fixed(prop_ds), screen=Fixed(screen), shifts=Fixed(shifts))

    const_screen = screen .* 0 .+ 0.5
    start_val = (obj=cu(const_obj), prop_od=Fixed(cu(prop_od)), prop_ds=Fixed(cu(prop_ds)), screen=const_screen)

    use_cuda = false # true
    # use_cuda = true
    # cu = (x) -> CuArray(x)
    if (use_cuda)
        # optimize only obj with the given diffusor:
        start_val = (obj=cu(const_obj), prop_od=Fixed(cu(prop_od)), prop_ds=Fixed(cu(prop_ds)), screen=Fixed(cu(screen)), shifts=Fixed(cu(shifts)))
        # optimize both obj and diffusor:
        start_val = (obj=cu(const_obj), prop_od=Fixed(cu(prop_od)), prop_ds=Fixed(cu(prop_ds)), screen=cu(const_screen), shifts=Fixed(cu(shifts)))
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
