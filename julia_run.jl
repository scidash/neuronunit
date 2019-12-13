#=
ENV["PYTHON"]="/usr/local/lib/python3.5"
ENV["PYTHON"]="/usr/bin/python3.5"
using Pkg
Pkg.build("PyCall")
=#
using PyCall
py"""
from neuronunit.optimisation import optimization_management as om

anchor = neuronunit.__file__
anchor = os.path.dirname(anchor)
mypath = os.path.join(os.sep,anchor,'tests/russell_tests.p')
df = pd.DataFrame(rts)
for key,v in rts.items():
    helper_tests = [value for value in v.values() ]
    break
DO = om.make_ga_DO(param_edges, 1,  free_params=free_params, \
                   backend=backend, MU = 1,  protocol=protocol,seed_pop = seed_pop, hc=hold_constant)

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
