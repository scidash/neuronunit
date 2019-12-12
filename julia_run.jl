#=
ENV["PYTHON"]="/usr/local/lib/python3.5"
ENV["PYTHON"]="/usr/bin/python3.5"
using Pkg
Pkg.build("PyCall")
=#
using PyCall
py"""
from neuronunit.optimisation import optimization_management as om
"""
omj = py"om"
#using revisions
py"""
import sys
sys.path.insert(0, "./neuronunit")
"""
show(LOAD_PATH)
push!(LOAD_PATH, "/home/russell/git/SpikingNeuralNetworks.jl")
@show(typeof(typeof("haha")))

methodswith(PyCall)
methods(py)
applicable(py, "dumm string")
#eltype()
#typeof()
using Pkg
Pkg.develop("/home/russell/git/SpikingNeuralNetworks.jl")
