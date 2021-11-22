# setup and activate project environment
using Pkg
Pkg.activate(@__DIR__)

# setup conda environment
Pkg.add("Conda")
using Conda
Conda.add_channel("conda-forge")
Conda.add("python=3.7.8")
Conda.add("h5py=2.10.0")
Conda.add("pillow=8.0.1")
Conda.add("numpy=1.17")
Conda.add("matplotlib=3.1.2")
Conda.add("tensorflow-gpu=1.15")
Conda.add("tqdm=4.53.0")
Conda.pip_interop(true)
Conda.pip("install", "keras-adf==19.1.0")

# setup pycall and make it use the conda env
ENV["PYTHON"] = ""  # empty path defaults to local conda
Pkg.add("PyCall")



