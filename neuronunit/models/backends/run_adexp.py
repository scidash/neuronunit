

from quantities import mV, ms, s, V
import sciunit
from neo import AnalogSignal
import neuronunit.capabilities as cap
import numpy as np
#from .base import Backend
from .base import *
import quantities as qt
import quantities as pq

#import matplotlib as mpl
try:
    import asciiplotlib as apl
except:
    pass
import numpy
voltage_units = mV

from elephant.spike_train_generation import threshold_detection


import numpy as np
from numba import jit

class Base_Population():
  def __init__(self, name, N=10, tau_psc=5.0, connectivity=None, spike_delta=100, v_rest=-70, scale=1.0):
    self.name = name
    self.N         = N           # Number of neurons in population
    self.receiver = [] #stored list of synapse weight matrices from other populations
    self.tau_psc = tau_psc #postsynaptic current filter constant
    self.spike_delta = spike_delta
    self.T = None #needs to be set upon starting simulation.
    self.v = np.ones(N) * v_rest #voltage trace
    self.I_ext = np.zeros(N) #externally injected current
    self.I_rec = np.zeros(N) #current from recurrent connections + other populations

  def initialize(self, T, len_time_trace, integration_time, save_data, now, properties_to_save, dt, stdp_t_window=50):
    self.T = T
    self.spike_raster = np.zeros((self.N, len_time_trace))
    self.psc = np.zeros((self.N, len_time_trace))
    self.integrate_window = np.int(np.ceil(integration_time/dt))
    self.dt = dt

    if self.integrate_window > len_time_trace:
      self.integrate_window = len_time_trace #if we are running a short simulation then integrate window will overflow available time slots!

  def update_currents(self, all_populations, I_ext, i, t, dt):
    #I_ext is reset at the very end of the update_state
    self.I_rec = np.zeros(self.I_rec.shape)
    #update I_ext vector for current time step
    for name, inj in I_ext.items():
      if name == self.name:
        for stim in inj:
          if stim['start'] <= t <= stim['stop']:
            #drive only neurons specified in stim['neurons']
            current = np.zeros(self.N)
            current[stim['neurons']] = stim['mV']
            self.I_ext += current

  def update_state(self, i, T, t, dt):
    return True

class Network():
    def __init__(self, populations=[]):
      self.populations = {}

      for p in populations:
         self.populations[p.name] = p

    def get(self,pname):
      if self.populations[pname]:
        return self.populations[pname]
      else:
        print("%s not found!" % pname)
        return None
    def setup(self, experiment_name='My Experiment', T=50,dt=0.125,integration_time=30, I_ext={},spike_delta=50):
      self.T          = T
      self.dt         = dt
      self.time_trace = np.arange(0,T+dt,dt)#time array
      self.I_ext = I_ext

      self.params = {
        'experiment_name' : experiment_name,
        'T' : self.T,
        'dt' : self.dt,
        'populations' : {},
        'I_ext' : I_ext,
        'save_data' : False
      }

      for name, p in self.populations.items(): 
        self.params['populations'][name] = p.N
      #print('gets here',p.N,'empty dic')
      #initialize populations for simulation.
      save_data='./'
      properties_to_save=[]
      self.now = 0.0
      for name, p in self.populations.items():
        p.initialize(T, 
		    len(self.time_trace), 
		    integration_time, 
		    save_data,
		    self.now, 
        properties_to_save, 
        dt)
      return self.params

    def _setup(self,
	    experiment_name='My Experiment',
	    T=50,dt=0.125,integration_time=30,
	    I_ext={},spike_delta=50):
      _ = self.setup(experiment_name, 
			T,dt,
			integration_time, 
			I_ext,
			spike_delta)


    def simulate(self, experiment_name='My Experiment', T=50,dt=0.125,integration_time=30, I_ext={},spike_delta=50):
      """
      Simulate the Network
      """
      #check spike_raster size and time_trace size...
      #run the simulation
      vm = []  
      for i, t in enumerate(self.time_trace[1:],1):
        #update state variables
        p = self.populations['Charly the Neuron']
        p.update_currents(all_populations=self.populations, I_ext=I_ext, i=i, t=t, dt=self.dt)
        p.update_state(i=i, T=self.T, t=t, dt=self.dt)
        vm.append(p.v[0])
      return vm


class AdExPopulation(Base_Population):
  def __init__(self, name, 
    cm=0.281,
    tau_refrac=0.1, 
    v_spike=-40.0, 
    v_reset=-70.6, 
    v_rest=-70.6, 
    tau_m=9.3667, 
    i_offset=0.0, 
    a=4.0, 
    b=0.0805, 
    delta_T=2.0,
    tau_w=144.0,
    v_thresh=-50.4,
    e_rev_E=0.0, 
    tau_syn_E=5.0, 
    e_rev_I=-80.0, 
    tau_syn_I=5.0, 
    N=1, 
    tau_psc=5.0, 
    connectivity=None, 
    spike_delta=30,
    scale=0.5):
    """
    AdEx Constructor


    """
    Base_Population.__init__(self, name, N, tau_psc, connectivity, spike_delta, v_reset,scale)
    self.cm         = cm         # Capacitance of the membrane in nF
    self.tau_refrac = tau_refrac # Duration of refractory period in ms.
    self.v_spike    = v_spike    # Spike detection threshold in mV.
    self.v_reset    = v_reset    # Reset value for V_m after a spike. In mV.
    self.v_rest     = v_rest     # Resting membrane potential (Leak reversal potential) in mV.
    self.tau_m      = tau_m      # Membrane time constant in ms
    self.i_offset   = i_offset   # Offset current in nA
    self.a          = a          # Subthreshold adaptation conductance in nS.
    self.b          = b          # Spike-triggered adaptation in nA
    self.delta_T    = delta_T    # Slope factor in mV
    self.tau_w      = tau_w      # Adaptation time constant in ms
    self.v_thresh   = v_thresh   # Spike initiation threshold in mV
    self.e_rev_E    = e_rev_E    # Excitatory reversal potential in mV.
    self.tau_syn_E  = tau_syn_E  # Decay time constant of excitatory synaptic conductance in ms.
    self.e_rev_I    = e_rev_I    # Inhibitory reversal potential in mV.
    self.tau_syn_I  = tau_syn_I  # Decay time constant of the inhibitory synaptic conductance in ms.
    #state variables
    #self.v is already present
    self.w = np.ones(N)

  def update_state(self, i, T, t, dt):
    #compute v and adaptation resets
    prev_spiked = np.nonzero(self.spike_raster[:,i-1] == True)
    self.v[prev_spiked] = self.v_reset
    self.w[prev_spiked] += self.b
    #compute deltas and apply to state variables
    dv  = (((self.v_rest-self.v) + self.delta_T*np.exp((self.v - self.v_thresh)/self.delta_T))/self.tau_m + (self.i_offset + self.I_ext + self.I_rec - self.w)/self.cm) *dt
    self.v += dv
    self.w += dt * (self.a*(self.v - self.v_rest) - self.w)/self.tau_w * dt
    #decide whether to spike or not
    spiked = np.nonzero(self.v > self.v_thresh)
    self.v[spiked] = self.spike_delta
    self.spike_raster[spiked,i] = 1

    self.I_ext = np.zeros(self.I_ext.shape[0])
'''
# from pyN import Network
import numpy as np
one_neurons = AdExPopulation(name='Charly the Neuron', N=1)

neuron = Network(populations=[one_neurons])
stim = [{'start':100,'stop':1100,'mV':1.5,'neurons':[0]}]



neuron._setup(
    experiment_name='Single Neuron exhibiting tonic spiking',\
    T=2000,\
    dt=0.25,\
    I_ext={'Charly the Neuron':stim})

vm = neuron.simulate(
    experiment_name='Single Neuron exhibiting tonic spiking',\
    T=2000,\
    dt=0.25,\
    I_ext={'Charly the Neuron':stim})
vM = AnalogSignal(vm,
                       units = voltage_units,
                       sampling_period = 0.25*pq.ms)

thresh = threshold_detection(vM)
print(len(thresh),'spikes')
import matplotlib.pyplot as plt
plt.plot(vM.times,vM.magnitude)
plt.show()
import numpy as np
'''
#Backend
class ADEXPBackend(Backend):
  name = 'ADEXP'
  def init_backend(self, attrs=None, cell_name='alice',
                    current_src_name='hannah', DTC=None,
                    debug = False):
    super(ADEXPBackend,self).init_backend()
    self.model._backend.use_memory_cache = False
    self.current_src_name = current_src_name
    self.cell_name = cell_name
    self.vM = None
    self.attrs = attrs
    self.debug = debug
    self.temp_attrs = None
    self.default_attrs = {'C':89.7960714285714, 'a':0.01, 'b':15, 'c':-60, 'd':10, 'k':1.6, 'vPeak':(86.364525297619-65.2261863636364), 'vr':-65.2261863636364, 'vt':-50, 'dt':0.010, 'Iext':[]}

    if type(attrs) is not type(None):
        self.attrs = attrs
    # set default parameters anyway.
    if type(DTC) is not type(None):
        if type(DTC.attrs) is not type(None):
            self.set_attrs(**DTC.attrs)
        if hasattr(DTC,'current_src_name'):
            self._current_src_name = DTC.current_src_name
        if hasattr(DTC,'cell_name'):
            self.cell_name = DTC.cell_name

  def get_spike_count(self):
    thresh = threshold_detection(self.vM)
    return len(thresh)

  def get_membrane_potential(self):
    """Must return a neo.core.AnalogSignal.
    And must destroy the hoc vectors that comprise it.
    """
    return self.vM


  def inject_square_current(self,current):#, section = None, debug=False):
    """Inputs: current : a dictionary with exactly three items, whose keys are: 'amplitude', 'delay', 'duration'
    Example: current = {'amplitude':float*pq.pA, 'delay':float*pq.ms, 'duration':float*pq.ms}}
    where \'pq\' is a physical unit representation, implemented by casting float values to the quanitities \'type\'.
    Description: A parameterized means of applying current injection into defined
    Currently only single section neuronal models are supported, the neurite section is understood to be simply the soma.
    """
    self.model = None
    
    '''
    attrs = copy.copy(self.model.attrs)
    '''
    if 'injected_square_current' in current.keys():
        c = current['injected_square_current']
    else:
        c = current
    amplitude = float(c['amplitude'])#/1000.0#*10.0#*1000.0 #this needs to be in every backends
    #print(amplitude)
    duration = float(c['duration'])#/dt#/dt.rescale('ms')
    delay = float(c['delay'])#/dt#.rescale('ms')
    tMax = delay + duration + 200.0#/dt#*pq.ms
    # from pyN import Network
    self.model = AdExPopulation(name='Charly the Neuron', N=1)
    #self.model = one_neurons
    neuron = Network(populations=[self.model])
    stim = [{'start':delay,'stop':duration+delay,'mV':amplitude,'neurons':[0]}]
    neuron._setup(
        experiment_name='Single Neuron exhibiting tonic spiking',\
        T=tMax,\
        dt=0.25,\
        I_ext={'Charly the Neuron':stim})

    vm = neuron.simulate(
        experiment_name='Single Neuron exhibiting tonic spiking',\
        T=tMax,\
        dt=0.25,\
        I_ext={'Charly the Neuron':stim})
    vM = AnalogSignal(vm,
                          units = voltage_units,
                          sampling_period = 0.25*pq.ms)
    self.vM = vM
    self.get_spike_count()
    #print(np.max(self.vM))
    return self.vM


