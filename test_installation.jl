using Pkg
Pkg.activate(@__DIR__)
using Conda
using PyCall

# check what is contained in the julia env
Pkg.status()

# check what is contained in the conda env
Conda.list()

# check which python pycall is using
@show pyimport("sys").executable
