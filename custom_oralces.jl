using SparseArrays: spzeros
import JuMP
const MOI = JuMP.MOI

"""
    NonNegKSparseLMO{T}(K::Int, right_hand_side::T)
LMO for the non-negative K-sparse polytope:
```
C = B_1(τK) ∩ B_∞(τ) ∩ R^n_+
```
with `τ` the `right_hand_side` parameter.
The LMO results in a vector with the K smallest negative values
of direction, taking values `τ`.
"""
struct NonNegKSparseLMO{T} <: FrankWolfe.LinearMinimizationOracle
    K::Int
    right_hand_side::T
end

function FrankWolfe.compute_extreme_point(lmo::NonNegKSparseLMO{T}, direction; kwargs...) where {T}
    K = min(lmo.K, length(direction))
    K_indices = sortperm(direction[1:K])
    K_values = direction[K_indices]
    for idx in K+1:length(direction)
        new_val = direction[idx]
        # new smaller value: shift everything right
        if new_val < K_values[1]
            K_values[2:end] .= K_values[1:end-1]
            K_indices[2:end] .= K_indices[1:end-1]
            K_indices[1] = idx
            K_values[1] = new_val
            # new value in the interior
        elseif new_val < K_values[K]
            # NOTE: not out of bound since unreachable with K=1
            j = K - 1
            while new_val < K_values[j]
                j -= 1
            end
            K_values[j+1:end] .= K_values[j:end-1]
            K_indices[j+1:end] .= K_indices[j:end-1]
            K_values[j+1] = new_val
            K_indices[j+1] = idx
        end
    end
    v = spzeros(T, length(direction))
    for (idx, val) in zip(K_indices, K_values)
        if val < 0
            v[idx] = lmo.right_hand_side
        end
    end
    return v
end



function FrankWolfe.convert_mathopt(
    lmo::NonNegKSparseLMO{T},
    optimizer::OT;
    dimension::Integer,
    kwargs...,
) where {T,OT}
    τ = lmo.right_hand_side
    n = dimension
    K = min(lmo.K, n)
    MOI.empty!(optimizer)
    (x, _) = MOI.add_constrained_variables(optimizer, fill(MOI.Interval(0.0, τ), n))
    MOI.add_constraint(optimizer, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(n), x), 0.0), MOI.LessThan(τ * K))
    return FrankWolfe.MathOptLMO(optimizer)
end
