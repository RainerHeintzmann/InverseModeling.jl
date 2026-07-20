module CUDASupportExt_FFTPlan
# Make cuFFT handle ForwardDiff.Dual element types on CuArrays.
# FFT is linear: FFT(Dual(x, ẋ)) = Dual(FFT(x), FFT(ẋ)) exactly.
# This is necessary for Zygote's forward-over-reverse Hessian-vector product
# (used in schedule_directional_hessian), which seeds Dual numbers that
# propagate through FFT operations in the inner gradient computation.
#
# Two representations can arise during the computation:
#   A: Dual{Nothing,Complex{T},1}  — Dual of Complex
#   B: Complex{Dual{Nothing,T,1}}  — Complex of Duals
#
# We define plan_* methods for Dual CuArrays so the generic AbstractFFTs.fft
# fallback works. Additionally, we define ChainRules rrules for fft/ifft/bfft
# on Dual CuArrays to prevent Zygote from recursively tracing through the
# AbstractFFTs rrule (which calls fft internally, causing circular _pullback
# generation). Our rrules compute the forward pass directly via value/derivative
# decomposition without calling fft, breaking the recursion.

using ForwardDiff: Dual, value, partials
using CUDA: CuArray
using AbstractFFTs
import ChainRulesCore
import ForwardDiff


# ── Value & derivative extraction ──
# Case A: Dual{Nothing, Complex{T}, 1}
_val(x::Dual{Nothing,Complex{T},1}) where T = x.value
_der(x::Dual{Nothing,Complex{T},1}) where T = x.partials[1]

# Case B: Complex{Dual{Nothing, T, 1}}
_val(x::Complex{Dual{Nothing,T,1}}) where T = Complex{T}(value(real(x)), value(imag(x)))
_der(x::Complex{Dual{Nothing,T,1}}) where T = Complex{T}(partials(real(x), 1), partials(imag(x), 1))

# ── Recombination (direct broadcasts, avoid closures for Zygote tracing) ──
# Case B: Complex{Dual} → Dual(real part) + Dual(imag part) → Complex{Dual}
_recombine(y_v::CuArray{Complex{T}}, y_d::CuArray{Complex{T}}, ::Type{Complex{Dual{Nothing,T,1}}}) where T =
    Complex.(Dual.(real.(y_v), real.(y_d)), Dual.(imag.(y_v), imag.(y_d)))
# Case A: Dual{Complex} → construct Dual from value and derivative
_recombine(y_v::CuArray{Complex{T}}, y_d::CuArray{Complex{T}}, ::Type{Dual{Nothing,Complex{T},1}}) where T =
    Dual.(y_v, y_d)

# ── Base element type (T from Dual or Complex{Dual}) ──
_dual_base(::Type{Complex{Dual{Nothing,T,1}}}) where T = T
_dual_base(::Type{Dual{Nothing,Complex{T},1}}) where T = T

# ── Wrapper plan that handles Dual arrays ──
struct DualFFTPlan{P, T}
    base::P
end

Base.:*(plan::DualFFTPlan{P,T}, x::CuArray{Complex{Dual{Nothing,T,1}}}) where {P,T} =
    _plan_apply(x, plan.base)
Base.:*(plan::DualFFTPlan{P,T}, x::CuArray{Dual{Nothing,Complex{T},1}}) where {P,T} =
    _plan_apply(x, plan.base)

function _plan_apply(x, base_plan)
    x_val = broadcast(_val, x)
    x_der = broadcast(_der, x)
    y_val = base_plan * x_val
    y_der = base_plan * x_der
    _recombine(y_val, y_der, eltype(x))
end

# ── plan_* methods for Dual CuArrays ──
for pf in (:plan_fft, :plan_ifft, :plan_bfft)
    @eval begin
        # Case B: Complex{Dual} — the most common form
        function AbstractFFTs.$pf(x::CuArray{Complex{Dual{Nothing,T,1}}}, region) where T
            base = AbstractFFTs.$pf(CuArray{Complex{T}}(undef, size(x)), region)
            return DualFFTPlan{typeof(base), T}(base)
        end
        # Case A: Dual{Complex}
        function AbstractFFTs.$pf(x::CuArray{Dual{Nothing,Complex{T},1}}, region) where T
            base = AbstractFFTs.$pf(CuArray{Complex{T}}(undef, size(x)), region)
            return DualFFTPlan{typeof(base), T}(base)
        end
    end
end

# ── Dual-aware FFT helper (avoids calling fft/ifft/bfft to prevent recursion) ──
function _fft_dual(::typeof(fft), x, dims)
    Tb = _dual_base(eltype(x))
    p = AbstractFFTs.plan_fft(CuArray{Complex{Tb}}(undef, size(x)), dims)
    x_v = broadcast(_val, x)
    x_d = broadcast(_der, x)
    _recombine(p * x_v, p * x_d, eltype(x))
end
function _fft_dual(::typeof(ifft), x, dims)
    Tb = _dual_base(eltype(x))
    p = AbstractFFTs.plan_ifft(CuArray{Complex{Tb}}(undef, size(x)), dims)
    x_v = broadcast(_val, x)
    x_d = broadcast(_der, x)
    _recombine(p * x_v, p * x_d, eltype(x))
end
function _fft_dual(::typeof(bfft), x, dims)
    Tb = _dual_base(eltype(x))
    p = AbstractFFTs.plan_bfft(CuArray{Complex{Tb}}(undef, size(x)), dims)
    x_v = broadcast(_val, x)
    x_d = broadcast(_der, x)
    _recombine(p * x_v, p * x_d, eltype(x))
end

# Mark _fft_dual as non-differentiable so Zygote doesn't trace through its
# internal broadcasts/plan operations. The correct gradient is provided by
# the custom rrules for fft/ifft/bfft below.
ChainRulesCore.@non_differentiable _fft_dual(::Any, ::Any, ::Any)

# ── Custom rrules to prevent Zygote recursive tracing ──
# These are more specific than the AbstractFFTs rrules (match CuArray with Dual
# elements), so Zygote finds them first. The forward pass uses _fft_dual which
# does NOT call fft/ifft/bfft directly, breaking the recursion.

function ChainRulesCore.rrule(::typeof(fft),
        x::CuArray{<:Union{Complex{<:ForwardDiff.Dual}, ForwardDiff.Dual}},
        dims)
    y = _fft_dual(fft, x, dims)
    function pullback(ȳ)
        x̄ = _fft_dual(bfft, ȳ, dims)
        return ChainRulesCore.NoTangent(), x̄, ChainRulesCore.NoTangent()
    end
    return y, pullback
end

function ChainRulesCore.rrule(::typeof(ifft),
        x::CuArray{<:Union{Complex{<:ForwardDiff.Dual}, ForwardDiff.Dual}},
        dims)
    y = _fft_dual(ifft, x, dims)
    n = 1
    for d in dims
        n *= size(y, d)
    end
    invN = 1.0f0 / n
    function pullback(ȳ)
        x̄ = _fft_dual(fft, ȳ, dims) .* invN
        return ChainRulesCore.NoTangent(), x̄, ChainRulesCore.NoTangent()
    end
    return y, pullback
end

function ChainRulesCore.rrule(::typeof(bfft),
        x::CuArray{<:Union{Complex{<:ForwardDiff.Dual}, ForwardDiff.Dual}},
        dims)
    y = _fft_dual(bfft, x, dims)
    function pullback(ȳ)
        x̄ = _fft_dual(fft, ȳ, dims)
        return ChainRulesCore.NoTangent(), x̄, ChainRulesCore.NoTangent()
    end
    return y, pullback
end

end
