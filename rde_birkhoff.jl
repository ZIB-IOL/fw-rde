using Pkg
Pkg.activate(@__DIR__)

# parse command line arguments if given
if length(ARGS) > 0
    subdir = ARGS[1]
# otherwise prompt user to specify
else
    print("Please enter sub directory to run RDE in: ")
    subdir = readline()
end
# input validation
while !isdir(joinpath(@__DIR__, subdir))
    print("Invalid directory $subdir. Please enter sub directory to run RDE in: ")
    global subdir = readline()
end

using PyCall
pushfirst!(PyVector(pyimport("sys")["path"]), joinpath(@__DIR__, subdir))

import FrankWolfe
include("custom_oralces.jl")
include(joinpath(@__DIR__, subdir, "config_birkhoff.jl"))  # load indices, rates, max_iter
cd(subdir)

# Get the Python side of RDE
rde = pyimport("rde")

for idx in indices

    # Load data sample and distortion functional
    x, fname = rde.get_data_sample(idx)
    f, df, node, pred = rde.get_distortion(x)

    # Setup LMO
    lmo = FrankWolfe.BirkhoffPolytopeLMO()

    # helper functions for prototype ordering vectors
    lin_p = convert(Vector{eltype(x)}, LinRange(0.0, 1.0, length(x)))

    function k_to_p(k)
        pk = zeros(eltype(x), length(x))
        pk[end-k+1:end] .= 1.0
        return pk
    end

    # Wrap objective and gradient functions
    function func(S)
        if (S isa Matrix) && !(S isa Matrix{eltype(x)})
            S = convert(Matrix{eltype(x)}, S)
        end
        f_sum = convert(eltype(x), 0.0)
        for (ridx, rate) in enumerate(all_rates)
            f_sum += f(S*k_to_p(rate))
        end
        return f_sum
    end

    function grad!(storage, S)
        if (S isa Matrix) && !(S isa Matrix{eltype(x)})
            S = convert(Matrix{eltype(x)}, S)
        end
        df_sum = zeros(eltype(x), length(x), length(x))
        for (ridx, rate) in enumerate(all_rates)
            pk = k_to_p(rate)
            g = df(S*pk)
            if any(isnan, g) || any(isnan, pk)
                @info "Warning: Numerical instabilities, skipped rate k=$rate"
            else
                # df_sum = df_sum + g*transpose(pk)
                BLAS.ger!(convert(eltype(x), 1.0), g, pk, df_sum)
            end
        end
        return @. storage = df_sum
    end

    # Run FrankWolfe
    println("Running sample $idx")
    S0 = Matrix{eltype(x)}(I, length(x), length(x))

    @time S, V, primal, dual_gap = FrankWolfe.frank_wolfe(
    #@time S, V, primal, dual_gap = FrankWolfe.away_frank_wolfe(
    #@time S, V, primal, dual_gap = FrankWolfe.blended_conditional_gradient(
    #@time S, V, primal, dual_gap = FrankWolfe.lazified_conditional_gradient(
        S -> func(S),
        (storage, S) -> grad!(storage, S),
        lmo,
        S0,
        ;fw_arguments...
    )
    # reset adaptive step size if necessary
    if fw_arguments.line_search isa FrankWolfe.MonotonousNonConvexStepSize
        fw_arguments.line_search.factor = 0
    end

    # Store results
    all_s = zeros(eltype(x), (length(rates), length(x)))
    for (ridx, rate) in enumerate(rates)
        all_s[ridx, :] = S*k_to_p(rate)
        rde.store_single_result(all_s[ridx,:], idx, fname, rate)
    end

    # Store multiple rate results
    rde.store_collected_results(all_s, idx, node, pred, fname, rates, nothing, S, S*lin_p)

end
