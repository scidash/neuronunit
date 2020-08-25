using Pkg
#=
try
   Pkg.add("Py2Jl")
   using Py2Jl
catch
   Pkg.add("Py2Jl")
   using Py2Jl
end

ENV["PYTHON"]="/usr/local/lib/python3.5"
ENV["PYTHON"]="/usr/bin/python3.5"
using Pkg
Pkg.build("PyCall")
=#
using Debugger
using PyCall
using Random: bitrand, randperm, shuffle
using LinearAlgebra: dot
using UnicodePlots
include("plot.jl")
using SpikingNeuralNetworks
include("units.jl")
using Pkg
using NSGAIII#, PyPlot
using UnicodePlots
using JLD

try
    using NSGAIII
catch
    Pkg.clone("https://github.com/gsoleilhac/NSGAIII.jl")
    using NSGAIII
end

#using NSGAIII

try
   using SpikingNeuralNetworks
catch
   Pkg.add("SpikingNeuralNetworks")
   using SpikingNeuralNetworks

end
SNN = SpikingNeuralNetworks

using Debugger

py"""
from neuronunit.optimisation import optimization_management as om
from neuronunit.optimisation import model_parameters
from neuronunit.optimisation.optimization_management import TSD
#from neuronunit.optimisation.optimization_management import score_specific_param_models
from neuronunit.models import VeryReducedModel as VRM
from neuronunit.optimisation.optimization_management import inject_and_plot, inject_and_plot_model
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

backend=str('JHH')
param_edges = model_parameters.MODEL_PARAMS[backend]
protocol = {'allen':False,'elephant':True}
vm_container = []
vm = pickle.load(open('neuronunit/examples/dummy_wave.p','rb'))
known_parameters = {k:np.min(v) for k,v in param_edges.items()}
attrs = known_parameters
"""
py"""
param_edges = { k:(float(v[0]),float(v[1])) for k,v in param_edges.items() }
"""
@show("""
a lot of problems come from converting python objects to the julia namespace.
objects derived from base classes are converted back to base classes]
""")
rts = py"rts"

try
    load("cached_genes.jld")
catch
    py"""
    print(list(param_edges.keys()))
    tests = rts['Neocortex pyramidal cell layer 5-6']
    tests = TSD(tests)
    DO = om.make_ga_DO(param_edges,1,tests,free_params=list(param_edges.keys()),backend=backend, MU = 2,  protocol=protocol)
    cached_genes = DO.set_pop()
    """
    #inj_current = py"tests"["Rheobase test"].params["injected_square_current"]
    save("cached_genes.jld",py"cached_genes")
end
genes = load("cached_genes.jld")
try
    load("current.jld")
catch
    inj_current = py"tests"["Rheobase test"].params["injected_square_current"]
    save("current.jld",inj_current)
end

genes = py"pre_genes"

omj = py"om"
DO = py"DO"
known_parameters = py"known_parameters"
param_edges = py"param_edges"


function plot_pop(P)
    println()
    display(scatterplot(map(x -> x.y[1], P), map(x -> x.y[2], P)))
    sleep(0.4)
end
MU = 8
NGEN = 8

param_edges


jDO=py"DO.OM"

#define how to generate a random genotype :
function init()
    jDOOM=py"DO.OM"
    jDO=py"DO"
    @show(jDO)
    genes = jDO.set_pop()
    @show(genes)
    return genes
end

MU = 2
NGEN = 2

function zzz(in_genes)
    jDOOM=py"DO.OM"
    fitnesses,pop,dtcpop = jDOOM.update_deap_pop(in_genes,jDOOM.tests,jDOOM.td)
    @show (fitnesses)
    @show (dtcpop)
    @show (pop)
    #Convert(fitnesses)
    jfit = tuple(fitnesses)
    return jfit#pop
end
#fitnesses = zzz(genes)
#=
try
   nsga(MU, NGEN, init, zzz, fplot = plot_pop)
catch
   P = [indiv(genes[1], zzz, fCV) for _=1:MU-length(seed)]
   append!(P, indiv.(convert.(X, seed),z, fCV))
   fast_non_dominated_sort!(P)
   associate_references!(P, references)
   Q = similar(P)
end
#include("indivs.jl")
X = typeof(init())
fCV=(x)->0
seed = 1.0
P = [indiv(genes[1], zzz, fCV) for _=1:MU-length(seed)]
append!(P, indiv.(convert.(X, seed),z, fCV))
fast_non_dominated_sort!(P)
associate_references!(P, references)
Q = similar(P)
=#
#=


=#

#init_function = () -> randperm(N) #e.g : a permutation coding

#define how to evaluate a genotype :
#z(x) = z1(x), z2(x) ... # Must return a Tuple
#objective(x1,x2,x3,x4,x5,x6,x7,x8) =
#nsga(300, 20, ()->bitrand(d.nbbitstotal), z, 10, fplot = plot_pop)

#nsga(popSize::Integer, nbGen::Integer, init::Function, z::Function, fplot = plot_pop)
#const ranges = RealCoding(MU, param_edges.vals)
