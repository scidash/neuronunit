

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
#@jit

# code is a very aggressive
# hack on this repository:
# https://github.com/ericjang/pyN
@jit
def update_currents(I_ext, i, t, dt):
  #I_rec = np.zeros(self.I_rec.shape)
  stim = I_ext['Charly the Neuron'][0]
  I_ext_out = 0
  if stim['start'] <= t <= stim['stop']:
    I_ext_out = stim['mV']
    #print(I_ext_out)
  return I_ext_out

@jit
def I_inj(t,delay,duration,amplitude):
	"""
	External Current

	|  :param t: time
	|  :return: nothing at time interval uA/cm^2 at 0<delay
		    step up to amplitude uA/cm^2 at t>delay
	|           step down to 0 uA/cm^2 at t>delay+duration
	|           nothing uA/cm^2 at delay+duration>t>end
	"""

	return 0*(t<delay) +amplitude*(t>delay) +0*(t>delay+duration) # a scalar value.
@jit
def update_state(p, i, T, t, dt):
  #compute v and adaptation resets
  prev_spiked = np.nonzero(p.spike_raster[:,i-1] == True)
  p.v[prev_spiked] = p.v_reset
  p.w[prev_spiked] += p.b
  #compute deltas and apply to state variables
  dv  = (((p.v_rest-p.v) + \
        p.delta_T*np.exp((p.v - p.v_thresh)/p.delta_T))/p.tau_m + \
        (p.i_offset + p.I_ext + p.I_rec - p.w)/p.cm) *dt

  p.v += dv
  p.w += dt * (p.a*(p.v - p.v_rest) - p.w)/p.tau_w * dt
  #decide whether to spike or not
  spiked = np.nonzero(p.v > p.v_thresh)
  p.v[spiked] = p.spike_delta
  p.spike_raster[spiked,i] = 1
  #print(p.spike_raster[spiked,i])

  p.I_ext = 0#np.zeros(p.I_ext)
  return p

@jit
def evaluate_vm(p,time_trace,I_ext,dt,T):
  vm = []
  for i, t in enumerate(time_trace[1:],1):
    p.I_ext = update_currents(I_ext=I_ext, i=i, t=t, dt=dt)
    p = update_state(p,i=i, T=T, t=t, dt=dt)
    vm.append(p.v[0])
  return vm

class Base_Population():
  def __init__(self, name, N=1, tau_psc=1.0, connectivity=None, spike_delta=100, v_rest=-70, scale=1.0):
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
    self.spike_raster = np.zeros((1, len_time_trace))
    #self.psc = np.zeros((self.N, len_time_trace))
    self.integrate_window = np.int(np.ceil(integration_time/dt))
    self.dt = dt

    if self.integrate_window > len_time_trace:
      self.integrate_window = len_time_trace #if we are running a short simulation then integrate window will overflow available time slots!
  '''
  def update_currents(I_ext, i, t, dt):
    #I_ext is reset at the very end of the update_state
    #self.I_rec = np.zeros(self.I_rec.shape)
    #update I_ext vector for current time step
    #for inj in I_ext.values():
    #if name == self.name:
    #I_ext = []
    for stim in I_ext['inj']:
      if stim['start'] <= t <= stim['stop']:
        #drive only neurons specified in stim['neurons']
        current = np.zeros(self.N)
        current[stim['neurons']] = stim['mV']
        I_ext += current
    return I_ext
    '''

  #def update_state(self, i, T, t, dt):
  #  return True

class Network():
    def __init__(self, populations=[],attrs = None):
      self.populations = {}
      self.attrs = attrs

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


    def simulate(self, attrs=None, T=50,dt=0.125,integration_time=30, I_ext={},spike_delta=50):#,v_rest = v_rest):
      self.spike_delta = spike_delta
      N = 1
      #self.T = None #needs to be set upon starting simulation.
      #self.v = np.ones(N) * v_rest #voltage trace
      #self.I_ext = np.zeros(N) #externally injected current
      self.I_rec = np.zeros(N) #current from recurrent connections + other populations

      self.T          = T
      self.dt         = dt
      self.time_trace = np.arange(0,T+dt,dt)#time array
      self.I_ext = I_ext
      len_time_trace = len(self.time_trace)
      #self.T = T
      integration_time = 30.0
      self.spike_raster = np.zeros((1, len_time_trace))
      #self.psc = np.zeros((self.N, len_time_trace))
      self.integrate_window = np.int(np.ceil(integration_time/dt))
      self.dt = dt

      if self.integrate_window > len_time_trace:
        self.integrate_window = len_time_trace #if we are running a short simulation then integrate window will overflow available time slots!
      p = self.populations['Charly the Neuron']
      self.attrs = attrs
      p.cm = self.attrs['cm']
      p.tau_refrac = self.attrs['tau_refrac']
      p.v_spike = self.attrs['v_spike']
      p.v_rest =  self.attrs['v_rest']
      p.tau_m = self.attrs['tau_m']
      p.i_offset = self.attrs['i_offset']
      p.a = self.attrs['a']
      p.b = self.attrs['b']
      p.delta_T = self.attrs['delta_T']
      p.tau_w = self.attrs['tau_w']
      p.v_thresh = self.attrs['v_thresh']
      p.e_rev_E = self.attrs['e_rev_E']
      p.tau_syn_E = self.attrs['tau_syn_E']
      p.e_rev_I = self.attrs['e_rev_I']
      p.tau_syn_I = self.attrs['tau_syn_I']
      p.spike_delta = self.attrs['spike_delta']
      p.scale = self.attrs['scale']
      vm = evaluate_vm(p,self.time_trace,self.I_ext,self.dt,T)
      return vm


class AdExPopulation(Base_Population):
  def __init__(self,
    name,
    cm=None,
    tau_refrac=0.1,
    v_spike=None,
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
    self.w = np.ones(N)
    Base_Population.__init__(self, name, N, tau_psc, connectivity, spike_delta, v_reset,scale)


class ADEXPBackend(Backend):
  name = 'ADEXP'
  def init_backend(self, attrs=None, DTC=None):
    self.model._backend.use_memory_cache = False
    #self.current_src_name = current_src_name
    #self.cell_name = cell_name
    self.vM = None
    #self.attrs = attrs
    #self.debug = debug
    self.temp_attrs = None

    BAE1 = {}
    BAE1['cm']=0.281
    BAE1['tau_refrac']=0.1
    BAE1['v_spike']=-40.0
    BAE1['v_reset']=-70.6
    BAE1['v_rest']=-70.6
    BAE1['tau_m']=9.3667
    BAE1['i_offset']=0.0
    BAE1['a']=4.0
    BAE1['b']=0.0805
    BAE1['delta_T']=2.0
    BAE1['tau_w']=144.0
    BAE1['v_thresh']=-50.4
    BAE1['e_rev_E']=0.0
    BAE1['tau_syn_E']=5.0
    BAE1['e_rev_I']=-80.0
    BAE1['tau_syn_I']=5.0
    BAE1['spike_delta']=30
    BAE1['scale']=0.5

    self.default_attrs = BAE1
    super(ADEXPBackend,self).init_backend()

    if type(DTC) is not type(None):
        if type(DTC.attrs) is not type(None):
            self.set_attrs(**DTC.attrs)
        if DTC.attrs is None:
          self.set_attrs(self.default_attrs)

  def get_spike_count(self):
    thresh = threshold_detection(self.vM)
    return len(thresh)

  def set_attrs(self,attrs):
    self.default_attrs.update(attrs)
    attrs = self.default_attrs

    
    if not hasattr(self.model,'attrs'):# is None:
        self.model.attrs = {}
        self.model.attrs.update(attrs)
    else:
        self.model.attrs.update(attrs)
    
    self.model_ = AdExPopulation(name='Charly the Neuron',N=1)
    self.neuron = Network(populations=[self.model_],attrs = attrs)
   
    return self

  def get_membrane_potential(self):
    """Must return a neo.core.AnalogSignal.
    And must destroy the hoc vectors that comprise it.
    """
    return self.vM
  def set_stop_time(self, stop_time = 650*pq.ms):
      """Sets the simulation duration
      stopTimeMs: duration in milliseconds
      """
      self.tstop = float(stop_time.rescale(pq.ms))


  def inject_square_current(self,current):#, section = None, debug=False):
    """Inputs: current : a dictionary with exactly three items, whose keys are: 'amplitude', 'delay', 'duration'
    Example: current = {'amplitude':float*pq.pA, 'delay':float*pq.ms, 'duration':float*pq.ms}}
    where \'pq\' is a physical unit representation, implemented by casting float values to the quanitities \'type\'.
    Description: A parameterized means of applying current injection into defined
    Currently only single section neuronal models are supported, the neurite section is understood to be simply the soma.
    """
    try:
      assert len(self.model.attrs)
    except:
      print("this means you didnt instance a model and then add in model parameters")
    temp_attrs = copy.copy(self.model.attrs)
    assert len(temp_attrs)

    #self.init_backend()
    if len(temp_attrs):
      self.set_attrs(temp_attrs)

    if 'injected_square_current' in current.keys():
        c = current['injected_square_current']
    else:
        c = current
    amplitude = float(c['amplitude'])#/1000.0#*10.0#*1000.0 #this needs to be in every backends
    duration = float(c['duration'])#/dt#/dt.rescale('ms')
    delay = float(c['delay'])#/dt#.rescale('ms')
    tMax = delay + duration + 200.0#/dt#*pq.ms

    self.set_stop_time(stop_time = tMax*pq.ms)
    tMax = float(self.tstop)
    # from pyN import Network
    stim = [{'start':delay,'stop':duration+delay,'mV':amplitude,'neurons':[0]}]
    #self.neuron = self.set_attrs(self.attrs)
   
    #self.neuron = self.set_attrs(self.attrs)
    self.neuron._setup(
        experiment_name='Single Neuron exhibiting tonic spiking',\
        T=tMax,\
        dt=0.25,\
        I_ext={'Charly the Neuron':stim})
    
    #v_rest = temp_attrs['v_rest']
    vm = self.neuron.simulate(
        attrs=temp_attrs,\
        T=tMax,\
        dt=0.25,\
        I_ext={'Charly the Neuron':stim})#,v_rest=v_rest)
    vM = AnalogSignal(vm,
                          units = voltage_units,
                          sampling_period = 0.25*pq.ms)
    self.vM = vM
    #self.get_spike_count()
    return self.vM
