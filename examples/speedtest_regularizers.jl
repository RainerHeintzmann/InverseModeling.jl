using InverseModeling
using CUDA
using BenchmarkTools

function main()
    data = rand(Float32, 1000, 1000, 100);
    datac = cu(data)
    datap = (obj=data,)
    datapc = (obj=datac,)
    myreg1 = TV(num_dims=3)


    myreg2 = reg_TV(:obj, num_dims=3)
    myreg2 = reg_GR(:obj, num_dims=3)

    @time q1 = myreg1(data)
    @time q1 = myreg1(datac)

    myreg2 = reg_TV(:obj)

    @time q2 = myreg2(datap)
    @time q2 = myreg2(datapc)

    @btime myreg1($data) # CPU: 34,9 ms
    @btime CUDA.@sync myreg1($datac) # CUDA: 29,26 ms ... 37 ms
    @btime myreg2($datap) # CPU: TV_cuda = 300 ms, TV_KA = 1.2 sec, TV_views = 1.2 sec, TV_cuda = 328 ms
    @btime CUDA.@sync myreg2(datapc) # CUDA: TV 38.3 ms, TV_cuda =  31 ms, TV_KA = 84 ms, TV_views = 84 ms, 

end
