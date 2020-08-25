using Pkg

try
   using Py2Jl
catch
   Pkg.add("MLStyle")
   Pkg.add("Py2Jl")
   using Py2Jl
end

ENV["PYTHON"]="/home/travis/miniconda/lib/python3.7"
ENV["PYTHON"]="/home/travis/miniconda/lib/python3.7"
using Pkg
Pkg.build("PyCall")

using Debugger
using PyCall
using Random: bitrand, randperm, shuffle
using LinearAlgebra: dot
using UnicodePlots
include("plot.jl")
using SpikingNeuralNetworks
include("units.jl")
using Pkg
#using NSGAIII#, PyPlot
using UnicodePlots

try
    using NSGAIII
catch
    Pkg.clone("https://github.com/gsoleilhac/NSGAIII.jl")
    using NSGAIII
end


try
   using NSGAIII
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

backend=str('RAW')
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


py"""
print(list(param_edges.keys()))
tests = rts['Neocortex pyramidal cell layer 5-6']
tests = TSD(tests)
DO = om.make_ga_DO(param_edges,1,tests,free_params=list(param_edges.keys()),backend=backend, MU = 10,  protocol=protocol)
pre_genes = DO.set_pop()

using JLD
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

function debug_s(DO) @show(DO.OM) end
debug_s(DO)

function inject_square_current(current)
    SNN = SpikingNeuralNetworks
    E = SNN.HH(;N = 1)
    E.ge = [0.0]
    E.gi = [0.0]


    #@show(current["duration"])
    E.I = [current["amplitude"]*nA]
    SNN.monitor(E, [:v])
    SNN.sim!([E], []; dt = 0.015ms, duration = current["duration"]*ms)
    v = SNN.getrecord(E, :v)
    return v
end
function dont_do_this()
    v = inject_square_current(inj_current)

    E = SNN.HH(;N = 1)
    E.ge = [0.0]
    E.gi = [0.0]

    E.I = [0.01825nA]
    SNN.monitor(E, [:v])
    SNN.sim!([E], []; dt = 0.015ms, duration = 2000ms)
    # SNN.vecplot(E, :v) |> display
    v = SNN.getrecord(E, :v)
end

function myfunc(omj1)
     @show(omj1)
     return omj1
end
function dont_do_this_three()
    omj1 = myfunc(omj)
    Pkg.status()
end
module NUnitAccess
    using JLD
    using SpikingNeuralNetworks
    SNN = SpikingNeuralNetworks
    using UnicodePlots
    include("plot.jl")
    using SpikingNeuralNetworks
    include("units.jl")

    function inject_square_current(current)


        #SNN = SpikingNeuralNetworks
        E = SNN.HH(;N = 1)
        E.ge = [0.0]
        E.gi = [0.0]
        #@show(current)
        #current = current["current"]
        #@show(current["duration"])
        E.I = [current["amplitude"]*nA]
        SNN.monitor(E, [:v])
        SNN.sim!([E], []; dt = 0.015*ms, duration = current["duration"]*ms)
        v = SNN.getrecord(E, :v)


        return v
    end
    inj_current = load("current.jld")
    @show(inj_current)
    v = inject_square_current(inj_current)
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
        def get_membrane_potential(volts):
            vm = AnalogSignal(volts,
                 units = mV,
                 sampling_period = self.dt *ms)
            return vm
        #def one(x):
        #    return 10*np.sin(x) ** 2 + np.cos(x) ** 2
        """
    end
   get_membrane_potential(x) = py"get_membrane_potential"(inject_square_current(inj_current))
end
#@show(NUnitAccess.get_membrane_potential(10))
function plot_pop(P)
    println()
    display(scatterplot(map(x -> x.y[1], P), map(x -> x.y[2], P)))
    sleep(0.4)
end
MU = 8
NGEN = 8
const d = RealCoding(MU, [-3, -3], [3, 3])

param_edges
#z1(x1, x2) = -(3(1-x1)^2 * exp(-x1^2 - (x2+1)^2) - 10(x1/5 - x1^3 - x2^5) * exp(-x1^2-x2^2) -3exp(-(x1+2)^2 - x2^2) + 0.5(2x1 + x2))
#z2(x1, x2) = -(3(1+x2)^2 * exp(-x2^2 - (1-x1)^2) - 10(-x2/5 + x2^3 + x1^5) * exp(-x1^2-x2^2) - 3exp(-(2-x2)^2 - x1^2))
#z(x) = begin
#    x1, x2 = decode(x, d)
#    z1(x1, x2), z2(x1, x2)
#end


jDO=py"DO.OM"
function this_works()
    py"""
    returned = DO.OM.update_deap_pop(pre_genes,DO.OM.tests,DO.OM.td)
    """
end
function objective(x1,x2,x3,x4,x5,x6,x7,x8)
    DO.OM
end

function z(in_genes)
    jDOOM=py"DO.OM"
    out_genes = jDOOM.update_deap_pop(in_genes,jDOOM.tests,jDOOM.td)
    return out_genes
end

popsize = 200
nbGenerations = 100

#Define the number of division along each objective to generate the reference directions.
H = 5
#Alternatively, you can directly pass the reference directions as a Vector{Vector{Float64}} :
#With two objectives, H = 2 is equivalent to
H = [[1., 0.], [0.5, 0.5], [0., 1.]]

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
fitnesses = zzz(genes)
X = typeof(init())
fCV=(x)->0
seed = 1.0
P = [indiv(genes[1], zzz, fCV) for _=1:MU-length(seed)]
append!(P, indiv.(convert.(X, seed),z, fCV))
fast_non_dominated_sort!(P)
associate_references!(P, references)
Q = similar(P)
=#
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

#=


=#

#init_function = () -> randperm(N) #e.g : a permutation coding

#define how to evaluate a genotype :
#z(x) = z1(x), z2(x) ... # Must return a Tuple
#objective(x1,x2,x3,x4,x5,x6,x7,x8) =
#nsga(300, 20, ()->bitrand(d.nbbitstotal), z, 10, fplot = plot_pop)

#nsga(popSize::Integer, nbGen::Integer, init::Function, z::Function, fplot = plot_pop)
#const ranges = RealCoding(MU, param_edges.vals)
