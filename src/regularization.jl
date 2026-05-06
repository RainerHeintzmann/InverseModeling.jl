export combine
export reg_TV, reg_TH, reg_Tikhonov, reg_GR

"""
    combine(loss1::Function, loss2::Function)

combines multiple regularisations, passed as separate functions into one by creating a function that adds the results.
"""
function combine(loss1::Function, loss2::Function)
    combined_loss(params) = loss1(params) + loss2(params)
    return combined_loss
end

# function wrap_regularizer(dst:Symbol; namedargs...)    
# end

# function apply_regularizer(myreg::Function, data, dst::Symbol)
#     return myreg(data[dst])
# end

"""
    reg_TV(dst:Symbol, λ=1; namedargs...)

obtain a total variation regularization to be applied to a particular symbol `dst` in the dataset to optimize.
This regularzer is typically used in the context of `optimize()` applied to a model.

"""
function reg_TV(dst::Symbol, λ=1; namedargs...)
    # myreg = TV(;namedargs...) # crashes!
    # myreg = TV_cuda(;namedargs...)
    # my_regularizer(data) = eltype(data[dst])(λ)*myreg(get_fwd_val(data[dst]))

    # my_regularizer(data) = eltype(data[dst])(λ)*TV_KA(get_fwd_val(data[dst]); namedargs...)
    my_regularizer(data) = eltype(data[dst])(λ)*TV_views(get_fwd_val(data[dst]); namedargs...)
    return my_regularizer 
end

function reg_TH(dst::Symbol, λ=1; namedargs...)
    myreg = TH(;namedargs...)
    return my_regularizer(data) = eltype(data[dst])(λ)*myreg(get_fwd_val(data[dst]))
end

function reg_Tikhonov(dst::Symbol, λ=1; namedargs...)
    myreg = Tikhonov(;namedargs...)
    return my_regularizer(data) = eltype(data[dst])(λ)*myreg(get_fwd_val(data[dst]))
end

function reg_GR(dst::Symbol, λ=1; namedargs...)
    myreg = GR(;namedargs...)
    return my_regularizer(data) = eltype(data[dst])(λ)*myreg(get_fwd_val(data[dst]))
end
