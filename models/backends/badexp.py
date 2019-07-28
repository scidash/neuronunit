#% matplotlib inline
import brian2 as b2
from neurodynex.adex_model import AdEx
from neurodynex.tools import plot_tools, input_factory
import io
import math
import pdb
from numba import jit

import numpy as np
from .base import *
import quantities as qt
from quantities import mV, ms, s, us
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt

from types import MethodType

#import matplotlib.pyplot as plt
# @jit(cache=True) I suspect this causes a memory leak


class ADEXPBackend(Backend):
    def get_spike_count(self):
        return int(self.spike_monitor.count[0])


    def init_backend(self, attrs=None, cell_name='thembi',
                     current_src_name='spanner', DTC=None,
                     debug = False):
        backend = 'adexp'
        super(ADEXPBackend,self).init_backend()
        self.debug = None
        self.model._backend.use_memory_cache = False
        self.current_src_name = current_src_name
        self.cell_name = cell_name
        self.vM = None
        self.attrs = attrs
        self.debug = debug
        self.temp_attrs = None
        self.AdEx = AdEx
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
        return self.vM

    def set_attrs(self, **attrs):
        '''
        BAE1 = {}
        BAE1['ADAPTATION_TIME_CONSTANT_tau_w'] = 100#*AdEx.b2.units.ms
        BAE1['ADAPTATION_VOLTAGE_COUPLING_a'] = 0.5#*AdEx.b2.units.nS
        BAE1['FIRING_THRESHOLD_v_spike'] = -30#*AdEx.b2.units.mV
        BAE1['MEMBRANE_RESISTANCE_R'] =  0.5#*AdEx.b2.units.Gohm
        BAE1['MEMBRANE_TIME_SCALE_tau_m'] = 5#*AdEx.b2.units.ms
        BAE1['RHEOBASE_THRESHOLD_v_rh'] = -50#*AdEx.b2.units.mV
        BAE1['SHARPNESS_delta_T'] = 2#*AdEx.b2.units.mV
        BAE1['SPIKE_TRIGGERED_ADAPTATION_INCREMENT_b'] = 7#*AdEx.b2.units.pA
        BAE1['V_RESET'] = -51#*AdEx.b2.units.mV
        BAE1['V_REST'] = -70#*AdEx.b2.units.mV
        '''
        self.AdEx.ADAPTATION_TIME_CONSTANT_tau_w = attrs['ADAPTATION_TIME_CONSTANT_tau_w']*AdEx.b2.units.ms
        self.AdEx.ADAPTATION_VOLTAGE_COUPLING_a = attrs['ADAPTATION_VOLTAGE_COUPLING_a']*AdEx.b2.units.nS
        #self.AdEx.FIRING_THRESHOLD_v_spike = 10*b2.mV #.attrs['FIRING_THRESHOLD_v_spike']
        self.AdEx.FIRING_THRESHOLD_v_spike = attrs['FIRING_THRESHOLD_v_spike']*AdEx.b2.units.mV

        self.AdEx.MEMBRANE_RESISTANCE_R = attrs['MEMBRANE_RESISTANCE_R']*AdEx.b2.units.Gohm
        self.AdEx.MEMBRANE_TIME_SCALE_tau_m = attrs['MEMBRANE_TIME_SCALE_tau_m']*AdEx.b2.units.ms
        self.AdEx.RHEOBASE_THRESHOLD_v_rh = attrs['RHEOBASE_THRESHOLD_v_rh']*AdEx.b2.units.mV
        self.AdEx.SHARPNESS_delta_T = attrs['SHARPNESS_delta_T']*AdEx.b2.units.mV
        self.AdEx.SPIKE_TRIGGERED_ADAPTATION_INCREMENT_b = attrs['SPIKE_TRIGGERED_ADAPTATION_INCREMENT_b']*AdEx.b2.units.pA
        self.AdEx.V_RESET = attrs['V_RESET']*AdEx.b2.units.mV
        self.AdEx.V_REST = attrs['V_REST']*AdEx.b2.units.mV
        print(self.AdEx.MEMBRANE_RESISTANCE_R)
        print(self.AdEx.V_REST)
        self.model.attrs.update(attrs)

    def mini_test(self, current):
        #self.AdEx.
        attrs = copy.copy(self.model.attrs)
        #self.set_attrs(**attrs)

        self.set_attrs(**attrs)
        c = copy.copy(current)
        if 'injected_square_current' in c.keys():
            c = current['injected_square_current']

        amplitude = float(c['amplitude'])
        duration = int(c['duration'])#/dt#/dt.rescale('ms')
        delay = int(c['delay'])#/dt#.resc1ale('ms')



        current = input_factory.get_step_current(delay, duration, 1. * b2.ms, 65.0 * b2.pA)
        state_monitor, self.spike_monitor = self.AdEx.simulate_AdEx_neuron(I_stim=current, simulation_time=(duration+delay)* b2.ms)

        print("nr of spikes: {}".format(self.spike_monitor.count[0]))

        return int(self.spike_monitor.count[0])

    #


    #self.model.get_spike_count = get_spike_count
        # np.array(spike_times)

    def inject_square_current(self, current):#, section = None, debug=False):
        """Inputs: current : a dictionary with exactly three items, whose keys are: 'amplitude', 'delay', 'duration'
        Example: current = {'amplitude':float*pq.pA, 'delay':float*pq.ms, 'duration':float*pq.ms}}
        where \'pq\' is a physical unit representation, implemented by casting float values to the quanitities \'type\'.
        Description: A parameterized means of applying current injection into defined
        Currently only single section neuronal models are supported, the neurite section is understood to be simply the soma.

        """
        attrs = copy.copy(self.model.attrs)
        self.set_attrs(**attrs)


        if 'injected_square_current' in current.keys():
            c = current['injected_square_current'];
        else:
            c = current
        amplitude = float(c['amplitude'])#*1000.0
        duration = int(c['duration'])#/dt#/dt.rescale('ms')
        delay = int(c['delay'])#/dt#.rescale('ms')
        #spikes =  self.mini_test(current)
        #if self.AdEx.RHEOBASE_THRESHOLD_v_rh*10.0 < c['amplitude'] or
        pre_current = int(duration+delay)

        current = input_factory.get_step_current(delay, pre_current, 1. * b2.ms, amplitude * b2.pA)
        st = (duration+delay)* b2.ms
        state_monitor, self.spike_monitor = self.AdEx.simulate_AdEx_neuron(I_stim=current, simulation_time=st)
        state_monitor.clock.dt = 1. *b2.ms
        self.dt = state_monitor.clock.dt #* us


        vm = [ float(i) for i in state_monitor.get_states()['v'] ]
        self.vM = AnalogSignal(vm,units = mV,sampling_period = float(self.dt) * pq.ms)
        #import pdb; pdb.set_trace()

        self.n_spikes = self.spike_monitor.count[0]
        #print(self.n_spikes, 'nspikes',spike_monitor.spike_trains())
        self.attrs = attrs
        self.debug = True
        if self.debug == False:
            plt.clf()
            plt.plot(self.vM.times,self.vM)
            plt.savefig(str(float(self.vM[-1]))+'.png')
        return self.vM

    def _local_run(self):
        results = {}
        v = self.get_membrane_potential()
        #print(len(v))
        #if self.vM is None:
            #v = get_vm(**attrs)
            #v = np.divide(v, 1000.0)
        #import pdb; pdb.set_trace()
        self.vM = AnalogSignal(v,
                     units = mV,
                     sampling_period = 1.0 * pq.ms)
        results['vm'] = self.vM
        results['t'] = self.vM.times
        results['run_number'] = results.get('run_number',0) + 1
        return results
