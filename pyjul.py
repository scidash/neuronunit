
#Main.include("julia_opt.jl")
#Main.include("julia_run.jl")

#print(Main.varinfo())
#print(MU,NGEN)

## TSD in particular breaks julia

#from neuronunit.models import VeryReducedModel as VRM
from sciunit import Model

#from neuronunit.optimisation.optimization_management import inject_and_plot, inject_and_plot_model
from types import MethodType
import numpy as np
import neuronunit
import os
#from neuronunit.optimisation.optimization_management import TSD
#from neuronunit.optimisation.optimization_management import score_specific_param_models
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
import os
from julia.api import JuliaInfo
juliainfo = JuliaInfo.load()


 
from julia import Main
import julia
from julia import Main
j = jj = julia.Julia()
j.eval('ENV["MPLBACKEND"]="qt5agg"')
j.eval('using Pkg')
j.eval('Pkg.pin("PyCall", v"1.4.0")')
j.eval("MU = 3")
j.eval("NGEN = 1")
j.eval("using SpikingNeuralNetworks")
j.eval("using NSGAIII")
Main.include("julia_run.jl")

#print(Main.varinfo())
#print(MU,NGEN)
j.eval("MU = 3")
j.eval("NGEN = 1")

try:
    vm = pickle.load(open('neuronunit/examples/dummy_wave.p','rb'))
except:
    pass
'''
from neuronunit.optimisation import model_parameters
backend=str('RAW')
param_edges = model_parameters.MODEL_PARAMS[backend]

known_parameters = {k:np.min(v) for k,v in param_edges.items()}
attrs = known_parameters
param_edges = { k:(float(v[0]),float(v[1])) for k,v in param_edges.items() }
'''
protocol = {'allen':False,'elephant':True}
#vm_container = []
import pdb
pdb.set_trace()
jj.eval("""
function zzz(in_genes)
    jDOOM=py"DO.OM"
    fitnesses,pop,dtcpop = jDOOM.update_deap_pop(in_genes,jDOOM.tests,jDOOM.td)
    @show (fitnesses)
    @show (dtcpop)
    @show (pop)

    return pop
end
""")
j.eval("nsga(MU, NGEN, init, zzz)")#, fplot = plot_pop)")
#j.eval("using JLD")

#Main.inject_square_current(current)
