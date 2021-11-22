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
    function func(S, k)
        if (S isa Matrix) && !(S isa Matrix{eltype(x)})
            S = convert(Matrix{eltype(x)}, S)
        end
        return f(S*k_to_p(k))
    end

    function grad!(storage, S, k)
        if (S isa Matrix) && !(S isa Matrix{eltype(x)})
            S = convert(Matrix{eltype(x)}, S)
        end
        pk = k_to_p(k)
        g = vec(df(S*pk))
        if any(isnan, g) || any(isnan, pk)
            @info "Warning: Numerical instabilities, skipped rate k=$k"
        else
            # storage += g*transpose(pk), but faster with BLAS?
            BLAS.ger!(convert(eltype(x), 1.0), g, pk, storage)
        end
        return storage
    end

    f_stoch = FrankWolfe.StochasticObjective(func, grad!, 1:length(x), zeros(eltype(x), length(x), length(x)))

    # Run FrankWolfe
    println("Running sample $idx")
    S0 = Matrix{eltype(x)}(I, length(x), length(x))

    @time S, V, primal, dual_gap = FrankWolfe.stochastic_frank_wolfe(
        f_stoch,
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
