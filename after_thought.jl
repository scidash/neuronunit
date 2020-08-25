#=
ENV["PYTHON"]="/usr/local/lib/python3.5"
ENV["PYTHON"]="/usr/bin/python3.5"
using Pkg
Pkg.build("PyCall")
=#
using Debugger
using PyCall
py"""
from neuronunit.optimisation import optimization_management as om
from neuronunit.optimisation import model_parameters
from neuronunit.optimisation.optimization_management import TSD
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
#show(LOAD_PATH)
#push!(LOAD_PATH, "/home/russell/git/SpikingNeuralNetworks.jl")
#using SpikingNeuralNetworks
#methodswith(PyCall)
using Pkg

try
   println("broken")
   # Pkg.develop("/home/russell/git/SpikingNeuralNetworks.jl")
catch
   @show("fail")
end
Pkg.add("SpikingNeuralNetworks.jl")
Pkg.resolve()
using SpikingNeuralNetworks
function myfunc(omj1)
     @show(omj1)
     #@bp # interactive debug mode will start here
end

#@run myfunc(omj)
#methods(py)
#applicable(py, "dumm string")
