
#include("../src/SpikingNeuralNetworks.jl")
using Pkg
#Pkg.add("Reexport")
try
   using SpikingNeuralNetworks
catch
   ] add "https://github.com/AStupidBear/SpikingNeuralNetworks.jl"
   #Pkg.develop("https://github.com/AStupidBear/SpikingNeuralNetworks.jl")
   using SpikingNeuralNetworks
end
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
