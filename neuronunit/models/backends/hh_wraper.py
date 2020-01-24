import julia
jl = julia.Julia()
from julia import Main
#jl.eval("using Pkg")
#jl.eval('Pkg.dev("https://github.com/russelljjarvis/SpikingNeuralNetworks.jl")')
jl.eval("using SpikingNeuralNetworks")
jl.eval("SNN = SpikingNeuralNetworks")
jl.eval('include("units.jl")')
jl.eval('include("plot.jl")')
# Main.eval("using Debugger")
 
import io
import math
import pdb
from numba import jit
import numpy as np
from .base import *
import quantities as pq
from quantities import mV as qmV
from quantities import ms as qms
from quantities import V as qV
import matplotlib as mpl

from neuronunit.capabilities import spike_functions as sf
mpl.use('Agg')
import matplotlib.pyplot as plt
from elephant.spike_train_generation import threshold_detection
ascii_plot = True



#@jit
def Id(t,delay,duration,tmax,amplitude):
    if 0.0 < t < delay:
        return 0.0
    elif delay < t < delay+duration:

        return amplitude#(100.0)

    elif delay+duration < t < tmax:
        return 0.0
    else:
        return 0.0
class JHHBackend(Backend):
    #def get_spike_count(self):
    #    return int(self.spike_monitor.count[0])
    def init_backend(self, attrs=None, cell_name='thembi',
                     current_src_name='spanner', DTC=None,
                     debug = False):
        backend = 'JHH'
        super(JHHBackend,self).init_backend()
        self.name = str(backend)

        self.model._backend.use_memory_cache = False
        self.current_src_name = current_src_name
        self.cell_name = cell_name
        self.vM = None
        self.attrs = attrs
        self.debug = debug
        self.temp_attrs = None
        self.n_spikes = None
        self.verbose = False
        self.E = None

        if type(attrs) is not type(None):
            self.set_attrs(**attrs)
            self.sim_attrs = attrs

        if type(DTC) is not type(None):
            if type(DTC.attrs) is not type(None):
                self.set_attrs(**DTC.attrs)
            if hasattr(DTC,'current_src_name'):
                self._current_src_name = DTC.current_src_name
            if hasattr(DTC,'cell_name'):
                self.cell_name = DTC.cell_name

    def get_spike_count(self):
        thresh = threshold_detection(self.vM,0.0*pq.mV)
        return len(thresh)

    def set_stop_time(self, stop_time = 650*pq.ms):
        """Sets the simulation duration
        stopTimeMs: duration in milliseconds
        """
        self.tstop = float(stop_time.rescale(pq.ms))


    def get_membrane_potential(self):
        """Must return a neo.core.AnalogSignal.
        And must destroy the hoc vectors that comprise it.
        """

        return self.vM

    def set_attrs(self, attrs):

        #Main.eval("SNN.HH(;N = 1)")
        Main.attrs = attrs
        Main.eval('param = SNN.HHParameter(;El =  attrs["El"], Ek = attrs["EK"], En = attrs["ENa"], gl = attrs["gl"], gk = attrs["gK"], gn = attrs["gNa"])')#', N = attrs["N"])')
        ###
        # SpikingNeuralNetworks.HH(; param, N, v, m, n, h, ge, gi, fire, I, records)
        ###

        attrs["N"] = 10000

        Main.temp = attrs["N"]
        #Main.eval('N = temp')
        #Main.eval('N = convert(Int32,temp)')
        Main.temp_i = attrs["Iext_"]
        Main.eval('I = convert(Array{Float32,1},temp_i)')
        Main.eval('N = size(I)[1]')

        Main.eval('E2 = SNN.HH(;N = 1)')
        #Main.eval("E2.param = param")
        Main.eval('E2.I = I')
        Main.eval('@assert size(E2.I)[1]>1')
        Main.eval('E2.N = size(E2.I)[1]')
        #Main.eval('E.ge = [0.0]')
        #Main.eval('E.gi = [0.0]')
        #Main.eval('E.v = convert(Array{Float32,1},[attrs["Vr"]])')
        #Main.eval('E2 = SNN.HH()')
        Main.eval('E2.v = ones(N).*attrs["Vr"]')

        #Main.eval("v = param.El .+ 5(ones(N) .- 1)")
        Main.eval("E2.m = zeros(N)")
        Main.eval("E2.n = zeros(N)")
        Main.eval("E2.h = ones(N)")
        Main.eval("E2.ge = (1.5randn(N) .+ 4) .* 10nS")
        Main.eval("E2.gi = (12randn(N) .+ 20) .* 10nS")
        Main.eval("E2.fire = zeros(Bool, N)")
        Main.eval('E2.I = convert(Array{Float32,1},temp_i)')
        Main.eval('E2.records = Dict()')
        Main.eval('@assert size(E2.m)[1]==size(E2.I)[1]')


        #Main.eval("I = ones(N)")
        ##
        # SpikingNeuralNetworks.HH(; param, N, v, m, n, h, ge, gi, fire, I, records)
        ##
        #Main.eval('E1 = SNN.HH(N,v,m, n, h, ge, gi, fire, I, records)')

        self.model.attrs.update(attrs)
        #return str('param = SNN.HHParameter(;El =  attrs["El"], Ek = attrs["EK"], En = attrs["ENa"], gl = attrs["gl"], gk = attrs["gK"], gn = attrs["gNa"])')

    def inject_square_current(self, current):#, section = None, debug=False):
        """Inputs: current : a dictionary with exactly three items, whose keys are: 'amplitude', 'delay', 'duration'
        Example: current = {'amplitude':float*pq.pA, 'delay':float*pq.ms, 'duration':float*pq.ms}}
        where \'pq\' is a physical unit representation, implemented by casting float values to the quanitities \'type\'.
        Description: A parameterized means of applying current injection into defined
        Currently only single section neuronal models are supported, the neurite section is understood to be simply the soma.

        """
        if 'injected_square_current' in current.keys():
            c = current['injected_square_current']
        else:
            c = current
        duration = float(c['duration'].rescale('ms'))
        delay = float(c['delay'].rescale('ms'))
        amp = float(c['amplitude'])#.rescale('uA')
        tmax = 2000.0
        self.set_stop_time(tmax*pq.ms)
        tmax = self.tstop
        tmin = 0.0
        DT = 0.01
        T = np.linspace(tmin, tmax, int(tmax/DT))
        Iext_ = []
        for t in T:
            Iext_.append(Id(t,delay,duration,tmax,amp))
        self.model.attrs['N'] = len(Iext_)
        self.model.attrs['Iext_'] = Iext_
        jstring = self.set_attrs(self.model.attrs)
        Main.attrs = self.model.attrs

        #Main.eval(jstring)

        #Main.Iext_ = [amp]
        #Main.eval('E.I = Iext_')



        Main.eval("SNN.monitor(E2, [:v])")
        Main.eval("SNN.monitor(E2, [:I])")
        Main.dur = current["duration"]
        Main.eval("v = SNN.getrecord(E2, :v)")
        Main.eval("ii = SNN.getrecord(E2, :I)")

        Main.eval("E2.I = convert(Vector{Float32},E2.I)")

        jl.eval("SNN.sim!([E2], []; dt = 0.015*ms, duration = dur*ms)")
        # jl.eval("SNN.sim!([param], []; dt = 0.015*ms, duration = dur*ms)")

        v = Main.v
        #if ascii_plot:
        jl.eval('SNN.vecplot(E, :v) |> display')
        jl.eval('E.I |> display')
        self.vM = AnalogSignal(v,units = qmV,sampling_period = DT * pq.ms)
        return self.vM

    def _backend_run(self):
        results = None
        results = {}
        results['vm'] = self.vM
        results['t'] = self.vM.times
        results['run_number'] = results.get('run_number',0) + 1
        return results
