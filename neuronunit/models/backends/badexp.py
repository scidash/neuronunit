#% matplotlib inline
import brian2 as b2
from neurodynex.adex_model import AdEx
b2.defaultclock.dt = 1 * b2.ms

from neurodynex.tools import plot_tools, input_factory
import io
import math
import pdb
from numba import jit
defaults = {'DT': 2e-05, 'ALLEN_STOP': 3.0199800000000003, 'ALLEN_FINISH': 10.020000000000001,'ALLEN_ONSET':1.02}
DT = defaults["DT"]
ALLEN_ONSET = defaults["ALLEN_ONSET"]
ALLEN_STOP = defaults["ALLEN_STOP"]
ALLEN_FINISH = defaults["ALLEN_FINISH"]
#from neuronunit.tests.base import ALLEN_ONSET, DT, ALLEN_STOP, ALLEN_FINISH
import numpy as np
from .base import *
import quantities as qt
from quantities import mV, ms, s, us, ns
import matplotlib as mpl

from neuronunit.capabilities import spike_functions as sf
mpl.use('Agg')
import matplotlib.pyplot as plt

from types import MethodType
# from neuronunit.optimisation import ascii_plot

#import matplotlib.pyplot as plt
# @jit(cache=True) I suspect this causes a memory leak
import asciiplotlib as apl


try:
    import asciiplotlib as apl
except:
    pass
import numpy

from scipy.interpolate import interp1d

class ADEXPBackend(Backend):
    def get_spike_count(self):
        print(int(self.spike_monitor.count[0]))
        #print(int(self.spike_monitor.count[0]))
        #import pdb; pdb.set_trace()
        return int(self.spike_monitor.count[0])
    def init_backend(self, attrs=None, cell_name='thembi',
                     current_src_name='spanner', DTC=None,
                     debug = False):
        backend = 'adexp'
        super(ADEXPBackend,self).init_backend()
        self.name = str(backend)

        #self.threshold = -20.0*qt.mV
        self.debug = None
        self.model._backend.use_memory_cache = False
        self.current_src_name = current_src_name
        self.cell_name = cell_name
        self.vM = None
        self.attrs = attrs
        self.debug = debug
        self.temp_attrs = None
        self.n_spikes = None
        self.spike_monitor = None


        self.model.get_spike_count = self.get_spike_count


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



    def set_stop_time(self, stop_time = 650*pq.ms):
        """Sets the simulation duration
        stopTimeMs: duration in milliseconds
        """
        self.tstop = float(stop_time.rescale(pq.ms))


    def get_membrane_potential(self):
        """Must return a neo.core.AnalogSignal.
        And must destroy the hoc vectors that comprise it.
        """
        if np.max(self.vM[i])!=0.020*qt.V:
            tdic = self.spike_monitor.spike_trains()
            for key,value in tdic.items():
                if len(value)==1:
                    i = int(float(value)/0.001)
                    self.vM[i] = 0.020*qt.mV
                else:
                    for v in value:
                       i = int(float(v)/0.001)
                       self.vM[i] = 0.020*qt.mV
                       #i = int(float(v)/0.001)
                       #self.vM[i] = 0.020*qt.mV
        return self.vM

    def set_attrs(self, **attrs):
        self.AdEx = None
        self.AdEx = AdEx
        self.AdEx.ADAPTATION_TIME_CONSTANT_tau_w = attrs['ADAPTATION_TIME_CONSTANT_tau_w']*AdEx.b2.units.ms
        self.AdEx.ADAPTATION_VOLTAGE_COUPLING_a = attrs['ADAPTATION_VOLTAGE_COUPLING_a']*AdEx.b2.units.nS
        self.AdEx.FIRING_THRESHOLD_v_spike = attrs['FIRING_THRESHOLD_v_spike']*AdEx.b2.units.mV
        self.AdEx.MEMBRANE_RESISTANCE_R = attrs['MEMBRANE_RESISTANCE_R']*AdEx.b2.units.Gohm
        self.AdEx.MEMBRANE_TIME_SCALE_tau_m = attrs['MEMBRANE_TIME_SCALE_tau_m']*AdEx.b2.units.ms
        self.AdEx.RHEOBASE_THRESHOLD_v_rh = attrs['RHEOBASE_THRESHOLD_v_rh']*AdEx.b2.units.mV
        self.AdEx.SHARPNESS_delta_T = attrs['SHARPNESS_delta_T']*AdEx.b2.units.mV
        self.AdEx.SPIKE_TRIGGERED_ADAPTATION_INCREMENT_b = attrs['SPIKE_TRIGGERED_ADAPTATION_INCREMENT_b']*AdEx.b2.units.pA
        self.AdEx.V_RESET = attrs['V_RESET']*AdEx.b2.units.mV
        self.AdEx.V_REST = attrs['V_REST']*AdEx.b2.units.mV
        self.model.attrs.update(attrs)

    def print_stuff(self,attrs):
        self.AdEx.simulate_AdEx_neuron()
        print(self.AdEx.ADAPTATION_TIME_CONSTANT_tau_w-attrs['ADAPTATION_TIME_CONSTANT_tau_w']*AdEx.b2.units.ms,
        self.AdEx.ADAPTATION_VOLTAGE_COUPLING_a-attrs['ADAPTATION_VOLTAGE_COUPLING_a']*AdEx.b2.units.nS,
        self.AdEx.FIRING_THRESHOLD_v_spike-attrs['FIRING_THRESHOLD_v_spike']*AdEx.b2.units.mV,
        self.AdEx.MEMBRANE_RESISTANCE_R-attrs['MEMBRANE_RESISTANCE_R']*AdEx.b2.units.Gohm,
        self.AdEx.MEMBRANE_TIME_SCALE_tau_m-attrs['MEMBRANE_TIME_SCALE_tau_m']*AdEx.b2.units.ms,
        self.AdEx.RHEOBASE_THRESHOLD_v_rh-attrs['RHEOBASE_THRESHOLD_v_rh']*AdEx.b2.units.mV,
        self.AdEx.SHARPNESS_delta_T-attrs['SHARPNESS_delta_T']*AdEx.b2.units.mV,
        self.AdEx.SPIKE_TRIGGERED_ADAPTATION_INCREMENT_b-attrs['b']*b2.pA,
        self.AdEx.V_RESET-attrs['V_RESET']*AdEx.b2.units.mV
        ,self.AdEx.V_REST-attrs['V_REST']*AdEx.b2.units.mV)

        print(self.AdEx.ADAPTATION_TIME_CONSTANT_tau_w,
        self.AdEx.ADAPTATION_VOLTAGE_COUPLING_a,
        self.AdEx.FIRING_THRESHOLD_v_spike,
        self.AdEx.MEMBRANE_RESISTANCE_R,
        self.AdEx.MEMBRANE_TIME_SCALE_tau_m,
        self.AdEx.RHEOBASE_THRESHOLD_v_rh,
        self.AdEx.SHARPNESS_delta_T,
        self.AdEx.SPIKE_TRIGGERED_ADAPTATION_INCREMENT_b,
        self.AdEx.V_RESET
        ,self.AdEx.V_REST)
        return
    def finalize(self):

        transform_function = interp1d([float(t) for t in self.vM.times],[float(v) for v in self.vM.magnitude])

        xnew = np.linspace(0, float(np.max(self.vM.times)), num=1004001, endpoint=True)
        vm_new = transform_function(xnew) #% generate the y values for all x values in xnew
        print(len(vm_new))
        self.vM = AnalogSignal(vm_new,units = mV,sampling_period = float(xnew[1]-xnew[0]) * pq.s)
        print(len(self.vM))
        #print(len(vm_new))
        self.vM = AnalogSignal(vm_new,units = mV,sampling_period = float(xnew[1]-xnew[0]) * pq.s)
        #print(len(self.vM))

    def inject_square_current(self, current):#, section = None, debug=False):
        """Inputs: current : a dictionary with exactly three items, whose keys are: 'amplitude', 'delay', 'duration'
        Example: current = {'amplitude':float*pq.pA, 'delay':float*pq.ms, 'duration':float*pq.ms}}
        where \'pq\' is a physical unit representation, implemented by casting float values to the quanitities \'type\'.
        Description: A parameterized means of applying current injection into defined
        Currently only single section neuronal models are supported, the neurite section is understood to be simply the soma.

        """
        b2.defaultclock.dt = 1 * b2.ms
        self.state_monitor = None
        self.spike_monitor = None
        self.AdEx = None
        self.AdEx = AdEx
        attrs = copy.copy(self.model.attrs)
        self.set_attrs(**attrs)
        if 'injected_square_current' in current.keys():
            c = current['injected_square_current'];
        else:
            c = current
        amplitude = float(c['amplitude'])#*1000.0
        duration = int(c['duration'])#/dt#/dt.rescale('ms')
        delay = int(c['delay'])#/dt#.rescale('ms')
        pre_current = int(duration)+100
        stim = input_factory.get_step_current(int(delay), int(pre_current), 1 * b2.ms, amplitude *b2.pA)
        st = (duration+delay+100)* b2.ms
        print(st, 'simulation time')
        #print(st, 'simulation time')


        self.state_monitor, self.spike_monitor = self.AdEx.simulate_AdEx_neuron(
        tau_m = attrs['MEMBRANE_TIME_SCALE_tau_m']*AdEx.b2.units.ms,
        R = np.abs(attrs['MEMBRANE_RESISTANCE_R'])*AdEx.b2.units.Gohm,
        v_rest = attrs['V_REST']*AdEx.b2.units.mV,
        v_reset = attrs['V_RESET']*AdEx.b2.units.mV,
        v_rheobase = attrs['RHEOBASE_THRESHOLD_v_rh']*AdEx.b2.units.mV,
        a = attrs['ADAPTATION_VOLTAGE_COUPLING_a']*AdEx.b2.units.nS,
        b =  attrs['b']*b2.pA,
        v_spike=attrs['FIRING_THRESHOLD_v_spike']*AdEx.b2.units.mV,
        delta_T = attrs['SHARPNESS_delta_T']*AdEx.b2.units.mV,
        tau_w = attrs['ADAPTATION_TIME_CONSTANT_tau_w']*AdEx.b2.units.ms ,
        I_stim = stim, simulation_time=st)


        #print("nr of spikes: {}".format(self.spike_monitor.count[0]))

        #print(self.AdEx.getting_started())
        self.state_monitor.clock.dt = 1 *b2.ms
        self.dt = self.state_monitor.clock.dt

        state_dic = self.state_monitor.get_states()
        vm = state_dic['v']
        vm = [ float(i) for i in vm ]

        self.vM = AnalogSignal(vm,units = mV,sampling_period = float(1.0) * pq.ms)


        tdic = self.spike_monitor.spike_trains()
        for key,value in tdic.items():

            if len(value)==1:
                i = int(float(value)/0.001)
                self.vM[i] = 0.020*qt.mV
            else:
                for v in value:
                    i = int(float(v)/0.001)
                    print(i)
                    #print(i)
                    self.vM[i] = 0.020*qt.mV


        self.n_spikes = int(self.spike_monitor.count[0])
        self.attrs = attrs


        t = [float(f) for f in self.vM.times]
        v = [float(f) for f in self.vM.magnitude]
        #print(len(v),len(t),'this is a short vector')
        fig = apl.figure()
        fig.plot(t, v, label=str('spikes: ')+str(self.n_spikes), width=100, height=20)
        fig.show()
        fig  = None
        try:
            fig = apl.figure()
            fig.plot(t, v, label=str('spikes: ')+str(self.n_spikes), width=100, height=20)
            fig.show()
            fig  = None
        except:
            pass
        if len(self.spike_monitor.spike_trains())>1:
            import matplotlib.pyplot as plt
            plt.plot(y,x)
            plt.savefig('debug.png')
        return self.vM

    def _backend_run(self):
        results = None
        results = {}

        results['vm'] = self.vM

        results['t'] = self.vM.times
        results['run_number'] = results.get('run_number',0) + 1
        return results
