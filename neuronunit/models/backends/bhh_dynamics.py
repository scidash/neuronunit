
import brian2 as b2
from neurodynex.hodgkin_huxley import HH
#b2.defaultclock.dt = 1 * b2.ms
#import brian2 as b2
second = 1000*b2.units.ms
b2.A = 1000000000000*b2.pA
b2.units.V = 1000.0*b2.units.mV
# Hodgkin Huxley parameters
'''
# Parameters
Cm = 1*b2.ufarad*cm**-2 * area
gl = 5e-5*siemens*cm**-2 * area
El = -65*mV
EK = -90*mV
ENa = 50*mV
g_na = 100*msiemens*cm**-2 * area
g_kd = 30*msiemens*cm**-2 * area
VT = -63*mV
'''
#b2.A = 1000000000000*b2.pA
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
#pq.PREFERRED = [pq.mV, pq.pA, pq.UnitQuantity('femtocoulomb', 1e-15*pq.C, 'fC')]

SLOW_ZOOM = False
#, ms, s, us, ns, V
import matplotlib as mpl
from neuronunit.capabilities import spike_functions as sf
mpl.use('Agg')
import matplotlib.pyplot as plt
from elephant.spike_train_generation import threshold_detection


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

from neuronunit.capabilities.spike_functions import get_spike_waveforms

def simulate_HH_neuron_local(input_current=None,
                            st=None,
                            El=None,\
                            EK=None,
                            ENa=None,
                            gl=None,\
                            gK=None,
                            gNa=None,
                            C=None,
                            Vr=None):
    # code lifted from:
    # /usr/local/lib/python3.5/dist-packages/neurodynex/hodgkin_huxley
    #input_current = I_stim #= #stim, simulation_time=st)
    """A Hodgkin-Huxley neuron implemented in Brian2.

    Args:
        input_current (TimedArray): Input current injected into the HH neuron
        st (float): Simulation time [seconds]

    Returns:
        StateMonitor: Brian2 StateMonitor with recorded fields
        ["vm", "I_e", "m", "n", "h"]

    https://brian2.readthedocs.io/en/stable/examples/IF_curve_Hodgkin_Huxley.html
    area = 20000*umetre**2
    Cm = 1*ufarad*cm**-2 * area
    gl = 5e-5*siemens*cm**-2 * area
    El = -65*mV
    EK = -90*mV
    ENa = 50*mV
    g_na = 100*msiemens*cm**-2 * area
    g_kd = 30*msiemens*cm**-2 * area
    VT = -63*mV
    """
    area = 20000*b2.umetre**2
    Cm = float(C) *b2.ufarad*b2.cm**-2 * area
    VT = -63*b2.mV
    assert float(El)<0.0
    # The model
    eqs = ('''
    dvm/dt = (gl*(El-vm) - g_na*(m*m*m)*h*(vm-ENa) - g_kd*(n*n*n*n)*(vm-EK) + input_current)/Cm : volt
    dm/dt = 0.32*(mV**-1)*4*mV/exprel((13.*mV-vm+VT)/(4*mV))/ms*(1-m)-0.28*(mV**-1)*5*mV/exprel((vm-VT-40.*mV)/(5*mV))/ms*m : 1
    dn/dt = 0.032*(mV**-1)*5*mV/exprel((15.*mV-vm+VT)/(5*mV))/ms*(1.-n)-.5*exp((10.*mV-vm+VT)/(40.*mV))/ms*n : 1
    dh/dt = 0.128*exp((17.*mV-vm+VT)/(18.*mV))/ms*(1.-h)-4./(1+exp((40.*mV-vm+VT)/(5.*mV)))/ms*h : 1
    input_current : amp
    ''')
    # Threshold and refractoriness are only used for spike counting
    neuron = b2.NeuronGroup(1, eqs,
                        threshold='v > -40*mV',
                        refractory='v > -40*mV',
                        method='exponential_euler')
    neuron.vm = El#*b2.units.mV
    
    #neuron.v = El
    #neuron = b2.NeuronGroup(1, eqs, method="exponential_euler")
    # parameter initialization
    #neuron.m = 0.05
    #neuron.h = 0.60
    #neuron.n = 0.32

    #spike_monitor = b2.SpikeMonitor(neuron)
    # tracking parameters
    st_mon = b2.StateMonitor(neuron, ["vm"], record=True)

    # running the simulation
    neuron = b2.Network(neuron)
    neuron.add(st_mon)
    # dur0 = 0.1*second
    neuron.I = '0.0*nA'

    dur0 = int(input_current['delay'])*second
    neuron.run(dur0)
    amp = input_current['amp']
    
    neuron.I = str(amp)+str('nA')
    dur1 = int(input_current['delay']+input_current['duration'])*second
    neuron.run(dur1)
    dur2 = 0.2*second
    neuron.I = '0.0*nA'
    neuron.run(dur2)
    import pdb
    pdb.set_trace()
    vm_b = neuron.vm
    vm_b = [ float(i) for i in vm_b ]
    vm_b = AnalogSignal(vm_b,units = pq.V,sampling_period = float(0.001) * pq.s)
    self.vM = vm_b
    """
    eqs =
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

    return st_mon,selfvM,vm

    """
    state_dic = st_mon.get_states()
    vm = state_dic['vm']
    v_nan = []
    for v in vm:
        v = v*1000.0
        if np.isnan(v):
           v_nan.append(-65.0*b2.units.mV)
        else:
           v_nan.append(v)
    vM = AnalogSignal(v_nan,units = pq.mV,sampling_period = float(0.001) * pq.s)
    """
    #vM = AnalogSignal(v_nan,units = pq.mV,sampling_period = 1*pq.ms)#b2.defaultclock.dt*pq.s)
    '''
    try:
     	vM.rescale_prefered()
    except:
		import pdb
		pdb.set_trace()
    '''

getting_started = False
class BHHBackend(Backend):
    
    name = 'BHH'
    
    def init_backend(self, attrs=None, cell_name='thembi',
                     current_src_name='spanner', DTC=None,
                     debug = False):
        
        super(BHHBackend,self).init_backend()

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
            self.set_attrs(attrs)
            self.sim_attrs = attrs

        if type(DTC) is not type(None):
            if type(DTC.attrs) is not type(None):
                self.set_attrs(DTC.attrs)
            if hasattr(DTC,'current_src_name'):
                self._current_src_name = DTC.current_src_name
            if hasattr(DTC,'cell_name'):
                self.cell_name = DTC.cell_name

    def get_spike_count(self):
        #if np.max(self.vM)>20.0*np.mean(self.vM):
        #thresh = threshold_detection(self.vM,np.max(self.vM)-0.10*np.max(self.vM))
        thresh = threshold_detection(self.vM,0.0*pq.mV)

        #else:
        #    thresh = []
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

    def set_attrs(self,attrs):

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
            #b2.defaultclock.dt = 1 * b2.ms

            self.HH =HH



    def inject_square_current(self, current):#, section = None, debug=False):
        """Inputs: current : a dictionary with exactly three items, whose keys are: 'amplitude', 'delay', 'duration'
        Example: current = {'amplitude':float*pq.pA, 'delay':float*pq.ms, 'duration':float*pq.ms}}
        where \'pq\' is a physical unit representation, implemented by casting float values to the quanitities \'type\'.
        Description: A parameterized means of applying current injection into defined
        Currently only single section neuronal models are supported, the neurite section is understood to be simply the soma.

        """
        #b2.defaultclock.dt = 1 * b2.ms
        self.state_monitor = None
        #self.spike_monitor = None
        self.HH = None
        self.HH = HH
        attrs = copy.copy(self.model.attrs)
        if self.model.attrs is None or not len(self.model.attrs):
            self.HH = HH
        else:
            self.set_attrs(attrs)
        if 'injected_square_current' in current.keys():
            c = current['injected_square_current']
        else:
            c = current


        duration = int(c['duration'])#/10.0)#/dt#/dt.rescale('ms')
        delay = int(c['delay'])#/10.0)#/dt#.rescale('ms')
        pre_current = int(duration)+100
        amp = c['amplitude']#.rescale('uA')


        """
        #amplitude = amp.simplified#/1000000.0
        params = {'delay':delay, 'duration':duration, 'amp':amp,'':}
        getting_started = False
        if getting_started == False:
            stim = input_factory.get_step_current(delay, duration, b2.ms, amp * b2.pA)
            st = (duration+delay+100)* b2.ms
        else:
            stim = input_factory.get_step_current(10, 7, b2.ms, 45.0 * b2.nA)

            st = 70 * b2.ms
	"""
        if self.model.attrs is None or not len(self.model.attrs):

            self.HH = HH
            self.state_monitor,self.vM = self.HH.simulate_HH_neuron(I_stim = {'delay':delay,'duration':duration,'amp':amp}, simulation_time=st)

        else:
            if self.verbose:
                print(attrs)
            self.set_attrs(attrs)

            (self.state_monitor,self.vM,vm) = simulate_HH_neuron_local(
            El = attrs['El'] * b2.units.V,
            EK = attrs['EK'] * b2.units.V,
            ENa = attrs['ENa'] * b2.units.V,
            gl =  attrs['gl'] * b2.units.msiemens,
            gK = attrs['gK'] * b2.units.msiemens,
            gNa = attrs['gNa'] * b2.units.msiemens,
            C = attrs['C'] * b2.units.ufarad,
            Vr = attrs['Vr'],
            input_current = {'delay':delay,'duration':duration,'amp':amp}
            )
            #params = params)

        #self.state_monitor.clock.dt = 1 *b2.ms
        self.dt = self.state_monitor.clock.dt

        self.attrs = attrs

        if ascii_plot:
            SLOW_ZOOM = False
            if SLOW_ZOOM and self.get_spike_count()>=1 :

                vm = get_spike_waveforms(self.vM)
            else:
                vm = self.vM
            t = [float(f) for f in vm.times]
            v = [float(f) for f in vm.magnitude]
            fig = apl.figure()
            fig.plot(t, v, label=str('brian HH: ')+str(vm.units)+str(current['amplitude']), width=100, height=20)

            fig.show()
            gc.collect()
            fig = None

        fig  = None
        return self.vM

    def _backend_run(self):
        results = None
        results = {}
        results['vm'] = self.vM
        results['t'] = self.vM.times
        results['run_number'] = results.get('run_number',0) + 1
        return results
