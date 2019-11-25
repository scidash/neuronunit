
import brian2 as b2
from neurodynex.hodgkin_huxley import HH
b2.defaultclock.dt = 1 * b2.ms

from neurodynex.tools import plot_tools, input_factory
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
SLOW_ZOOM = False
#, ms, s, us, ns, V
import matplotlib as mpl

from neuronunit.capabilities import spike_functions as sf
mpl.use('Agg')
import matplotlib.pyplot as plt
from elephant.spike_train_generation import threshold_detection


# from neuronunit.optimisation import ascii_plot

#import matplotlib.pyplot as plt
# @jit(cache=True) I suspect this causes a memory leak
getting_started = False
try:
    import asciiplotlib as apl
    fig = apl.figure()
    fig.plot([1,0], [0,1])
    ascii_plot = True
    import gc

except:
    ascii_plot = False
import numpy
try:
    brian2.clear_cache('cython')
except:
    pass
    #I_stim = stim, simulation_time=st)

def simulate_HH_neuron_local(I_stim=None, simulation_time=None,El=None,\
                            EK=None,ENa=None,gl=None,\
                            gK=None,gNa=None,C=None,Vr=None):
    # code lifted from:
    # /usr/local/lib/python3.5/dist-packages/neurodynex/hodgkin_huxley

    input_current = I_stim #= #stim, simulation_time=st)

    """A Hodgkin-Huxley neuron implemented in Brian2.

    Args:
        input_current (TimedArray): Input current injected into the HH neuron
        simulation_time (float): Simulation time [seconds]

    Returns:
        StateMonitor: Brian2 StateMonitor with recorded fields
        ["vm", "I_e", "m", "n", "h"]

    """
    # forming HH model with differential equations
    eqs = """
    I_e = input_current(t,i) : amp
    membrane_Im = I_e + gNa*m**3*h*(ENa-vm) + \
        gl*(El-vm) + gK*n**4*(EK-vm) : amp
    alphah = .07*exp(-.05*vm/mV)/ms    : Hz
    alpham = .1*(25*mV-vm)/(exp(2.5-.1*vm/mV)-1)/mV/ms : Hz
    alphan = .01*(10*mV-vm)/(exp(1-.1*vm/mV)-1)/mV/ms : Hz
    betah = 1./(1+exp(3.-.1*vm/mV))/ms : Hz
    betam = 4*exp(-.0556*vm/mV)/ms : Hz
    betan = .125*exp(-.0125*vm/mV)/ms : Hz
    dh/dt = alphah*(1-h)-betah*h : 1
    dm/dt = alpham*(1-m)-betam*m : 1
    dn/dt = alphan*(1-n)-betan*n : 1
    dvm/dt = membrane_Im/C : volt
    """

    neuron = b2.NeuronGroup(1, eqs, method="exponential_euler")

    # parameter initialization
    neuron.vm = 0
    neuron.m = 0.05
    neuron.h = 0.60
    neuron.n = 0.32
    #spike_monitor = b2.SpikeMonitor(neuron)

    # tracking parameters
    st_mon = b2.StateMonitor(neuron, ["vm", "I_e", "m", "n", "h"], record=True)

    # running the simulation
    hh_net = b2.Network(neuron)
    hh_net.add(st_mon)
    hh_net.run(simulation_time)

    state_dic = st_mon.get_states()
    vm = state_dic['vm']
    vm = [ (float(v)+Vr) for v in vm]
    vM = AnalogSignal(vm,units = pq.mV,sampling_period = float(1.0) * pq.ms)
    return st_mon,vM

getting_started = False
class BHHBackend(Backend):
    #def get_spike_count(self):
    #    return int(self.spike_monitor.count[0])
    def init_backend(self, attrs=None, cell_name='thembi',
                     current_src_name='spanner', DTC=None,
                     debug = False):
        backend = 'BHH'
        super(BHHBackend,self).init_backend()
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
        thresh = threshold_detection(self.vM)
        print(len(thresh),'num spikes')
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

    def set_attrs(self, **attrs):

        self.HH = None
        self.HH = HH

        if len(attrs):
            self.El = attrs['El'] * b2.units.mV
            self.EK = attrs['EK'] * b2.units.mV
            self.ENa = attrs['ENa'] * b2.units.mV
            self.gl =  attrs['gl'] * b2.units.msiemens
            self.gK = attrs['gK'] * b2.units.msiemens
            self.gNa = attrs['gNa'] * b2.units.msiemens
            self.C = attrs['C'] * b2.units.ufarad
            self.Vr = attrs['Vr']


            self.model.attrs.update(attrs)
        if attrs is None:
            b2.defaultclock.dt = 1 * b2.ms

            self.HH =HH
    def finalize(self):
        '''
        Necessary for imputing missing sampling, simulating at high sample frequency is prohibitevely slow, and yet, there is
        no significant difference in behavior.
        '''
        transform_function = interp1d([float(t) for t in self.vM.times],[float(v) for v in self.vM.magnitude])
        xnew = np.linspace(0, float(np.max(self.vM.times)), num=1004001, endpoint=True)
        vm_new = transform_function(xnew) #% generate the y values for all x values in xnew
        self.vM = AnalogSignal(vm_new,units = V,sampling_period = float(xnew[1]-xnew[0]) * pq.s)
        if self.verbose:

            print(len(self.vM))
        self.vM = AnalogSignal(vm_new,units = V,sampling_period = float(xnew[1]-xnew[0]) * pq.s)
        return self.vM


    def inject_square_current(self, current):#, section = None, debug=False):
        """Inputs: current : a dictionary with exactly three items, whose keys are: 'amplitude', 'delay', 'duration'
        Example: current = {'amplitude':float*pq.pA, 'delay':float*pq.ms, 'duration':float*pq.ms}}
        where \'pq\' is a physical unit representation, implemented by casting float values to the quanitities \'type\'.
        Description: A parameterized means of applying current injection into defined
        Currently only single section neuronal models are supported, the neurite section is understood to be simply the soma.

        """
        b2.defaultclock.dt = 1 * b2.ms
        self.state_monitor = None
        #self.spike_monitor = None
        self.HH = None
        self.HH = HH
        attrs = copy.copy(self.model.attrs)
        if self.model.attrs is None or not len(self.model.attrs):
            self.HH = HH
        else:
            self.set_attrs(**attrs)
        if 'injected_square_current' in current.keys():
            c = current['injected_square_current']
        else:
            c = current
        amplitude = float(c['amplitude'])/100.0#*100000.0
        duration = int(c['duration'])#/dt#/dt.rescale('ms')
        delay = int(c['delay'])#/dt#.rescale('ms')
        pre_current = int(duration)+100

        if getting_started == False:
            stim = input_factory.get_step_current(delay, duration, b2.ms, amplitude * b2.uA)
            st = (duration+delay+100)* b2.ms
        else:
            stim = input_factory.get_step_current(10, 45, b2.ms, 7.2 * b2.uA)
            st = 70 * b2.ms

        if self.model.attrs is None or not len(self.model.attrs):

            self.HH = HH
            self.state_monitor = self.HH.simulate_HH_neuron(I_stim = stim, simulation_time=st)

        else:
            if self.verbose:
                print(attrs)
            self.set_attrs(**attrs)


            self.state_monitor,self.vM = simulate_HH_neuron_local(
            El = attrs['El'] * b2.units.mV,
            EK = attrs['EK'] * b2.units.mV,
            ENa = attrs['ENa'] * b2.units.mV,
            gl =  attrs['gl'] * b2.units.msiemens,
            gK = attrs['gK'] * b2.units.msiemens,
            gNa = attrs['gNa'] * b2.units.msiemens,
            C = attrs['C'] * b2.units.ufarad,
            Vr = attrs['Vr'],
            I_stim = stim, simulation_time=st)

        #self.state_monitor.clock.dt = 1 *b2.ms
        self.dt = self.state_monitor.clock.dt

        self.attrs = attrs

        if ascii_plot:
            if SLOW_ZOOM and self.get_spike_count()>=1 :
                from neuronunit.capabilities.spike_functions import get_spike_waveforms
                vm = get_spike_waveforms(self.vM)
            else:
                vm = self.vM
            t = [float(f) for f in vm.times]
            v = [float(f) for f in vm.magnitude]
            fig = apl.figure()
            fig.plot(t, v, label=str('spikes: ')+str(self.n_spikes), width=100, height=20)
            fig.show()
            gc.collect()

        fig  = None
        return self.vM

    def _backend_run(self):
        results = None
        results = {}
        results['vm'] = self.vM
        results['t'] = self.vM.times
        results['run_number'] = results.get('run_number',0) + 1
        return results
