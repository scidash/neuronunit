#=
ENV["PYTHON"]="/usr/local/lib/python3.5"
ENV["PYTHON"]="/usr/bin/python3.5"
using Pkg
Pkg.build("PyCall")
=#
using Pkg
Pkg.resolve()
using Debugger
using PyCall
using Random: bitrand, randperm, shuffle
using LinearAlgebra: dot
using UnicodePlots
py"""
from neuronunit.optimisation import optimization_management as om
from neuronunit.optimisation import model_parameters
from neuronunit.optimisation.optimization_management import TSD
from neuronunit.models import VeryReducedModel as VRM
from neuronunit.optimisation.optimization_management import inject_and_plot

from types import MethodType

import numpy as np

import neuronunit
import os
import pickle
import pandas as pd
anchor = neuronunit.__file__
anchor = os.path.dirname(anchor)
mypath = os.path.join(os.sep,anchor,'tests/multicellular_constraints.p')
rts = pickle.load(open(mypath,'rb'))
df = pd.DataFrame(rts)

for key,v in rts.items():
    helper_tests = [value for value in v.values() ]
    break
backend=str('HH')
param_edges = model_parameters.MODEL_PARAMS['HH']
protocol = {'allen':False,'elephant':True}
vm_container = []
# dtc,_,_=inject_and_plot(neo_out[0]['pf'])
# dtc.vm
vm = pickle.load(open('neuronunit/examples/dummy_wave.p','rb'))

#
#from neuronunit.examples import elephant_data_tests
#a = elephant_data_tests.testHighLevelOptimisation()
#a.setUp()
#out = a.test_data_driven_ae()
#out = a.test_not_data_driven_rt_ae()
known_parameters = {k:np.mean(v) for k,v in param_edges.items()}
# score_specific_param_models(known_parameters,helper_tests)


"""
rts = py"rts"

py"""
print(list(param_edges.keys()))
tests = rts['Neocortex pyramidal cell layer 5-6']
tests = TSD(tests)
#import pdb
#pdb.set_trace()

#def make_ga_DO(explore_edges, max_ngen, test, \
#        free_params = None, hc = None,
#        selection = None, MU = None, seed_pop = None, \
#           backend = str('RAW'),protocol={'allen':False,'elephant':True}):
DO = om.make_ga_DO(param_edges,1,tests,free_params=list(param_edges.keys()),backend=backend, MU = 1,  protocol=protocol)

"""
omj = py"om"
DO = py"DO"
known_parameters = py"known_parameters"
function debug_s(DO)
    # DO.OM.score_specific_param_models(py"known_parameters",py"rts")
    @show(DO.OM)
    #@bp
end
debug_s(DO)
try
   using NSGAIII
catch
    Pkg.clone("https://github.com/gsoleilhac/NSGAIII.jl")
    using NSGAIII

end
using SpikingNeuralNetworks
SNN = SpikingNeuralNetworks
# varinfo()
include("units.jl")
include("plot.jl")
# ms = SNN.units.ms
# using Plots, SNN

E = SNN.HH(;N = 1)
E.I = [0.01825nA]


SNN.monitor(E, [:v])
SNN.sim!([E], []; dt = 0.015ms, duration = 2000ms)
SNN.vecplot(E, :v) |> display
#show(LOAD_PATH)

#push!(LOAD_PATH, "/home/russell/git/SpikingNeuralNetworks.jl")
#using SpikingNeuralNetworks
#methodswith(PyCall)
using Pkg
#using NSGAIII

try
   using NSGAIII
   using SpikingNeuralNetworks
catch
   Pkg.add("SpikingNeuralNetworks")
end
using Debugger
function myfunc(omj1)
     @show(omj1)
     return omj1
end
omj1 = myfunc(omj)

Pkg.status()

module NUnitAccess
    using PyCall
    function __init__()
        py"""
        from neuronunit.optimisation import optimization_management as om
        from neuronunit.optimisation import model_parameters
        from neuronunit.optimisation.optimization_management import TSD
        from neuronunit.models import VeryReducedModel as VRM
        from types import MethodType
        import numpy as np
        import neuronunit
        import os
        import pickle
        import pandas as pd
        def one(x):
            return 10*np.sin(x) ** 2 + np.cos(x) ** 2
        """
    end
   two(x) = py"one"(x) + py"one"(x)
end
@show(NUnitAccess.two(10))

#@run myfunc(omj)
#methods(py)
#applicable(py, "dumm string")


using NSGAIII#, PyPlot
#=
function plot_pop(P)
    #clf()
    p = plot(map(x -> x.y[1], P), map(x -> x.y[2], P), "bo", markersize=1)
    !isinteractive() && show()
    sleep(0.2)
end
=#
using UnicodePlots
function plot_pop(P)
    println()
    display(scatterplot(map(x -> x.y[1], P), map(x -> x.y[2], P)))
    sleep(0.4)
end

const d = RealCoding(8, [-3, -3], [3, 3])
z1(x1, x2) = -(3(1-x1)^2 * exp(-x1^2 - (x2+1)^2) - 10(x1/5 - x1^3 - x2^5) * exp(-x1^2-x2^2) -3exp(-(x1+2)^2 - x2^2) + 0.5(2x1 + x2))
z2(x1, x2) = -(3(1+x2)^2 * exp(-x2^2 - (1-x1)^2) - 10(-x2/5 + x2^3 + x1^5) * exp(-x1^2-x2^2) - 3exp(-(2-x2)^2 - x1^2))
z(x) = begin
    x1, x2 = decode(x, d)
    z1(x1, x2), z2(x1, x2)
end
nsga(300, 20, ()->bitrand(d.nbbitstotal), z, 10, fplot = plot_pop)
