


using Pkg
#Pkg.add("Reexport")
try
   using SpikingNeuralNetworks
catch
   Pkg.add("https://github.com/AStupidBear/SpikingNeuralNetworks.jl")
   using SpikingNeuralNetworks
end
SNN = SpikingNeuralNetworks
# varinfo()
include("units.jl")
include("plot.jl")
#ms = SNN.units.ms
# using Plots, SNN
MODEL_PARAMS = Dict()

attrs = Dict("Vr"=>-68.9346,
  "Cm"=>0.0002,
  "gl"=>1.0*1e-5,
  "El"=> -65.0,
  "EK"=> -90.0,
  "ENa"=> 50.0,
  "gNa"=> 0.02,
  "gK"=> 0.006,
  "Vt"=> -63.0
)
#JHH = { k:(float(v)-0.25*float(v),float(v)+0.25*float(v)) for (key,value) in attrs }
new_attrs = Dict()
for (key,value) in attrs new_attrs[key] = value+0.125*value end
#JHH = { k:(float(v)-0.25*float(v),float(v)+0.25*float(v)) for k,v in JHH.items() }
MODEL_PARAMS["JHH"] = attrs
param = SNN.HHParameter(;El =  attrs["El"], Ek = attrs["EK"], En = attrs["ENa"], gl = attrs["gl"], gk = attrs["gK"], gn = attrs["gNa"])
#param = SNN.HHParameter()



E = SNN.HH(;N = 1)
E.param = param
E.I = [0.01825nA]


SNN.monitor(E, [:v])
SNN.sim!([E], []; dt = 0.015ms, duration = 2000ms)
SNN.vecplot(E, :v) |> display

param = SNN.HHParameter(;El = new_attrs["El"], Ek = new_attrs["EK"], En = new_attrs["ENa"], gl = new_attrs["gl"], gk = new_attrs["gK"], gn = new_attrs["gNa"])

E.param = param
E.I = [0.01825nA]

SNN.monitor(E, [:v])
SNN.sim!([E], []; dt = 0.015ms, duration = 2000ms)
SNN.vecplot(E, :v) |> display
for (key,value) in attrs new_attrs[key] = value-0.125*value end

param = SNN.HHParameter(;El = new_attrs["El"], Ek = new_attrs["EK"], En = new_attrs["ENa"], gl = new_attrs["gl"], gk = new_attrs["gK"], gn = new_attrs["gNa"])

E.param = param
E.I = [0.01825nA]

SNN.monitor(E, [:v])
SNN.sim!([E], []; dt = 0.015ms, duration = 2000ms)
SNN.vecplot(E, :v) |> display
