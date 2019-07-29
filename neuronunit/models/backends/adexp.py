import copy
import pdb
import numpy as np
from .base import *
import quantities as qt
from quantities import mV, ms, s
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')
import brian2 as b2
from neurodynex.adex_model import AdEx
from neurodynex.tools import plot_tools, input_factory
from brian2 import NeuronGroup, StateMonitor, SpikeMonitor, run
'''
# Parameters
C = 281 * pF
gL = 30 * nS
taum = C / gL
EL = -70.6 * mV
VT = -50.4 * mV
DeltaT = 2 * mV
Vcut = VT + 5 * DeltaT

# Pick an electrophysiological behaviour
tauw, a, b, Vr = 144*ms, 4*nS, 0.0805*nA, -70.6*mV # Regular spiking (as in the paper)
#tauw,a,b,Vr=20*ms,4*nS,0.5*nA,VT+5*mV # Bursting
#tauw,a,b,Vr=144*ms,2*C/(144*ms),0*nA,-70.6*mV # Fast spiking


eqs = """
dvm/dt = (gL*(EL - vm) + gL*DeltaT*exp((vm - VT)/DeltaT) + I - w)/C : volt
dw/dt = (a*(vm - EL) - w)/tauw : amp
I : amp
"""

neuron = NeuronGroup(1, model=eqs, threshold='vm>Vcut',
                     reset="vm=Vr; w+=b", method='euler')
neuron.vm = EL
trace = StateMonitor(neuron, 'vm', record=0)
spikes = SpikeMonitor(neuron)

run(20 * ms)
neuron.I = 1*nA
run(100 * ms)
neuron.I = 0*nA
run(20 * ms)

# We draw nicer spikes
vm = trace[0].vm[:]
for t in spikes.t:
    i = int(t / defaultclock.dt)
    vm[i] = 20*mV

plot(trace.t / ms, vm / mV)
xlabel('time (ms)')
ylabel('membrane potential (mV)')
show()
'''

class BRIANADEXPBackend(Backend):


    def init_backend(self, attrs = None, cell_name= 'AdExp', current_src_name = 'hannah', DTC = None, dt=0.01, cell_type=None):
        backend = 'BRIAN2'
        super(BRIANADEXPBackend,self).init_backend()
        self.current_src_name = current_src_name
        self.cell_name = cell_name
        if type(cell_type) is None:
            self.cell_type = str('adexp')
            self.adexp = True
        else:
            self.adexp = False
        self.dt = dt
        #self.DCSource = DCSource
        self.setup = setup
        self.model_path = None
        self.related_data = {}
        self.lookup = {}
        self.attrs = {}
        #self.neuron = neuron
        self.model._backend.use_memory_cache = False
        #self.model.unpicklable += ['h','ns','_backend']
        #neuron.setup(timestep=dt, min_delay=1.0)

        if type(DTC) is not type(None):
            if type(DTC.attrs) is not type(None):

                self.set_attrs(**DTC.attrs)
                assert len(self.model.attrs.keys()) > 0

            if hasattr(DTC,'current_src_name'):
                self._current_src_name = DTC.current_src_name

            if hasattr(DTC,'cell_name'):
                self.cell_name = DTC.cell_name


    def get_membrane_potential(self):
        """Must return a neo.core.AnalogSignal.
        And must destroy the hoc vectors that comprise it.
        """

        #data = self.state_monitor
        #volts = data.filter(name="v")[0]

        vm = AnalogSignal(self.vm,
             units = mV,
             sampling_period = self.dt *ms)

        return vm
    '''
    def _local_run(self):
        '''
        '''
        #self.eif[0].set_parameters(**self.model.attrs)
        state_monitor, spike_monitor = AdEx.simulate_AdEx_neuron(I_stim=current, simulation_time=stop * b2.ms)
        trace = StateMonitor(neuron, 'vm', record=0)
        self.vm = AnalogSignal(copy.copy(trace),units = mV,sampling_period = self.dt *s)
        results = {}
        results['vm'] = self.vm
        results['t'] = self.vm.times
        results['run_number'] = results.get('run_number',0) + 1
        if self.debug == True:
            plt.clf()
            plt.plot(vm.times,vm)
            plt.savefig('debug_ad_exp.png')
        return results  
    '''	
    def load_model(self):
        neuron.setup(timestep=0.01, min_delay=1.0)
        #self.eif = neuron.Population(1,EIF_cond_exp_isfa_ista())

    def set_attrs(self, **attrs):
        self.model.attrs.update(attrs)
        assert type(self.model.attrs) is not type(None)
        defaults =
        MEMBRANE_TIME_SCALE_tau_m = 5 * b2.ms
        MEMBRANE_RESISTANCE_R = 500*b2.Mohm
        V_REST = -70.0 * b2.mV
        V_RESET = -51.0 * b2.mV
        RHEOBASE_THRESHOLD_v_rh = -50.0 * b2.mV
        SHARPNESS_delta_T = 2.0 * b2.mV
        ADAPTATION_VOLTAGE_COUPLING_a = 0.5 * b2.nS
        ADAPTATION_TIME_CONSTANT_tau_w = 100.0 * b2.ms
        SPIKE_TRIGGERED_ADAPTATION_INCREMENT_b = 7.0 * b2.pA
        AdEx.simulate_AdEx_neuron()
        #self.eif[0].set_parameters(**attrs)
        return self


    def set_stop_time(self, stop_time = 650*pq.ms):
        """Sets the simulation duration
        stopTimeMs: duration in milliseconds
        """
        self.tstop = float(stop_time.rescale(pq.ms))



        #current = input_factory.get_step_current(delay, duration, 1. * b2.ms, amplitude * b2.pA)
        #state_monitor, spike_monitor = self.AdEx.simulate_AdEx_neuron(I_stim=current, simulation_time=(duration+delay)* b2.ms)


    def inject_square_current(self, current):

        #plot_tools.plot_voltage_and_current_traces(state_monitor, current)
        #print("nr of spikes: {}".format(spike_monitor.count[0]))


        self.set_attrs(**attrs)
        c = copy.copy(current)
        if 'injected_square_current' in c.keys():
            c = current['injected_square_current']

        amplitude = float(c['amplitude'])
        duration = int(c['duration'])#/dt#/dt.rescale('ms')
        delay = int(c['delay'])#/dt#.rescale('ms')

        current = input_factory.get_step_current(delay, duration, 1. * b2.ms, amplitude * b2.pA)
        state_monitor, spike_monitor = self.AdEx.simulate_AdEx_neuron(I_stim=current, simulation_time=(duration+delay)* b2.ms)

        self.vM = AnalogSignal(state_monitor.get_states()['v'],
                     units = mV,
                     sampling_period = dt * ms)
        self.attrs = attrs
        if self.debug == True:
            plt.plot(self.vM.times,self.vM)
            plt.savefig('izhi_debug.png')
        return self.vM

    def _local_run(self):
        results = {}
        if self.vM is None:
            v = get_vm(**attrs)
            v = np.divide(v, 1000.0)
            self.vM = AnalogSignal(v,
                         units = mV,
                         sampling_period = dt * ms)
        results['vm'] = self.vM
        results['t'] = self.vM.times
        results['run_number'] = results.get('run_number',0) + 1
        return results
