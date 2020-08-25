

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

try:
    import asciiplotlib as apl
    fig = apl.figure()
    ascii_plot = True
    import gc

except:
    ascii_plot = False
#ascii_plot = False


import numpy as np
from numba import jit
import time
def timer(func):
    def inner(*args, **kwargs):
        t1 = time.time()
        f = func(*args, **kwargs)
        t2 = time.time()
        print('time taken on block {0} '.format(t2-t1))
        return f
    return inner
#@jit

# code is a very aggressive
# hack on this repository:
# https://github.com/ericjang/pyN
@jit
def update_currents(stim, i, t, dt):
  I_ext_out = 0
  if stim['start'] <= t <= stim['stop']:
    I_ext_out = stim['pA']#*0.001
  return I_ext_out
#@jit


kB = 1.3806503*10.0**23.0#; % Boltzmann constant in m^2 kg s^(-2) K^(-1)
Temperature = 310.65#; % Room temperature in Kelvin
thermoBeta = 1/(kB*Temperature)#;  % thermodynamic beta



import numpy as np
from pylab import *
#from scipy import *
from matplotlib import *
#import pyentropy
import copy

# Adaptive exponential integrate-and-fire neuron
@jit
def aEIF(inputCurrent,V_peak = 20,
          tau_w = 144,
          V_T=-50.4, delta_T = 2,
          g_L=30, E_L=-70.6 ,a=1.0, 
          vr=-70.6,C=281,b=0.0805  ):


#def aEIF(inputCurrent, a=1, vr=-65.0,C=281,g_L = 30,V_peak=20.0, b=0.0805, ):
    """
    Adaptive exponential integrate-and-fire neuron from Gerstner 2005 
    Parameters
    ----------
    adaptationIndex : degree to which neuron adapts                    
    inputCurrent : np array of the stimulus with size (M,N)
    v0 : specify the membrane potential at time 0 
    Returns
    -------
    V : membrane voltage for the simulation
    w : adaptation variable for the simulation
    spikes : 0 or 1 for each time bin
    sptimes : list of spike times
    """
    # https://github.com/lmcintosh/masters-thesis/blob/master/mastersFunctions.py
    # keep in mind that inputCurrent needs to start at 0 for sys to be in equilibrium
    
    # Physiologic neuron parameters from Gerstner et al.
    #C       =     # capacitance in pF ... this is 281*10^(-12) F
         # leak conductance in nS
    #E_L     = -70.6  # leak reversal potential in mV ... this is -0.0706 V
    #delta_T = 2      # slope factor in mV
    #V_T     = -50.4  # spike threshold in mV
    #tau_w   = 144    # adaptation time constant in ms
    #V_peak  = 20     # when to call action potential in mV
    #b       = # spike-triggered adaptation
    #a       = adapatationIndex
    
    # Simulation parameters
    delta = 1.0                     # dt
    M     = 1    # number of neurons
    N     = len(inputCurrent)  # number of simulation points is determined by size of inputCurrent
    T     = np.linspace(0,N*delta,N) # time points corresponding to inputCurrent (same size as V, w, I)
    
    # Thermodynamic parameters
    kB   = 1.3806503*10**(-23.0)   # Boltzmann's constant
    beta = 1.0/(kB*310.65)        # kB times T where T is in Kelvin
    
    # Initialize variables
    V       = np.zeros((N,M))
    V[0] = [ vr for i in V[0] ]#      = np.zeros((N,M))
    w       = np.zeros((N,M))
    w[0] = [ b for i in V[0] ]#      = np.zeros((N,M))


    spikes  = np.zeros((N,M))
    sptimes = [[] for _ in range(0,M)]
    V[0]    = vr                # this gives us a chance to say what the membrane voltage starts at
                                # so we can draw initial conditions from the Boltzmann dist. later
    '''
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
    spike_delta=30,
    '''
    # Run model
    for i in range(0,N-1):
        
        V[i+1,:] = V[i,:] + (delta/C)*( -g_L*(V[i,:] - E_L) + \
          g_L*delta_T*np.exp((V[i,:] - V_T)/delta_T) - w[i,:] + inputCurrent[i+1]) 
        w[i+1,:] = w[i,:] + (delta/tau_w)*(a*(V[i,:] - E_L) - w[i,:])
        # spiking mechanism
        ind = where(V[i+1,:] >= V_peak)
        if size(ind[0]) > 0:
            V[i+1,ind]      = E_L
            w[i+1,ind]      = w[i,ind] + b
            spikes[i+1,ind] = 1
            [sptimes[j].append(T[i+1]) for j in ind[0]]
    
    return [V,sptimes]    
'''
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
  p.I_ext = 0# 
  return p
'''
@jit
def evaluate_vm(attrs,ie):#,dt,T):
  #vm = []
  #ie = []
    
  #ie.append(I_ext)
  #p = update_state(p,i=i, T=T, t=t, dt=dt)
  #vm.append(p.v[0])
  '''
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
  '''

  vm,spikes = aEIF(ie,V_peak = attrs['V_peak'],
          tau_w = attrs['tau_w'],
          V_T=attrs['V_T'],delta_T = attrs['delta_T'],
          g_L=attrs['g_L'], E_L=attrs['E_L'],a=attrs['a'], 
          vr=attrs['v_rest'],C=attrs['C'],b=attrs['b'])#,V_peak=p.)
  #vm,spikes = aEIF(ie)#, a=0.1, vr=p.v_rest,C=p.cm,b=p.b)#,V_peak=p.)

  return vm,spikes

class Base_Population():
  def __init__(self, N=1,spike_delta=100, v_rest=-70, scale=1.0):
    self.N         = 1           # Number of neurons in population
    self.spike_delta = spike_delta
    self.T = None #needs to be set upon starting simulation.
    self.v = np.ones(N) * v_rest #voltage trace
    self.I_ext = 0#np.zeros(N) #externally injected current
    self.I_rec = 0#np.zeros(N) #current from recurrent connections + other populations


  def initialize(self, T, len_time_trace, integration_time, dt):#, stdp_t_window=50):
    self.T = T
    self.spike_raster = np.zeros((1, len_time_trace))
    #self.integrate_window = np.int(np.ceil(integration_time/dt))
    self.dt = dt

    #if self.integrate_window > len_time_trace:
    #  self.integrate_window = len_time_trace #if we are running a short simulation then integrate window will overflow available time slots!
 

class Container():
    def __init__(self,model,attrs = None):
      self.attrs = attrs
      self.spikes = None
 
    def setup(self, T=50,dt=1,integration_time=30, I_ext={},spike_delta=50):
      self.T          = T
      self.dt         = dt
      self.time_trace = np.arange(0,T+dt,dt)#time array
      self.I_ext = I_ext
      self.params = {
        'T' : self.T,
        'dt' : self.dt,
        'I_ext' : I_ext,
      }
      return self.params
      
    def _setup(self,
	    experiment_name='My Experiment',
	    T=50,dt=0.25,integration_time=30,
	    I_ext={},spike_delta=50):
      _ = self.setup(T,dt,
			integration_time,
			I_ext,
			spike_delta)

    @jit
    def simulate(self, attrs=None, T=50,dt=1,integration_time=30, stim=[],spike_delta=50):#,=None):#,v_rest = v_rest):
      self.time_trace = np.arange(0,T+dt,dt)#time array

      '''
      self.spike_delta = spike_delta
      N = 1
      self.I_rec = np.zeros(N) #current from recurrent connections + other populations
      self.T          = T
      self.dt         = dt
      
      #self.I_ext = I_ext
      len_time_trace = len(self.time_trace)
      integration_time = 30.0
      self.spike_raster = np.zeros((1, len_time_trace))
      #self.integrate_window = np.int(np.ceil(integration_time/dt))

      #if self.integrate_window > len_time_trace:
      #  self.integrate_window = len_time_trace #if we are running a short simulation then integrate window will overflow available time slots!
      p = AdExPopulation()
      p.initialize(T,
		    len(self.time_trace),
		    integration_time,
        dt)

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
      '''
      self.attrs = attrs

      ie = [update_currents(stim, i=i, t=t, dt=dt) for i, t in enumerate(self.time_trace[1:],1) ]

      vm,spikes = evaluate_vm(self.attrs,ie)#,self.dt,T)
      self.spikes = spikes
      
      return vm
    
class AdExPopulation(Base_Population):
  def __init__(self,
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
    spike_delta=30,
    scale=0.5):
    '''
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
    self.tau_syn_E  = tau_syn_E  # Decay time constant lof excitatory synaptic conductance in ms.
    self.e_rev_I    = e_rev_I    # Inhibitory reversal potential in mV.
    self.tau_syn_I  = tau_syn_I  # Decay time constant of the inhibitory synaptic conductance in ms.
    self.w = np.ones(1)
  
    Base_Population.__init__(self, 1, spike_delta, v_reset,scale)
    '''

class ADEXPBackend(Backend):
  name = 'ADEXP'
  def init_backend(self, attrs=None, DTC=None):
    self.model._backend.use_memory_cache = False
    self.vM = None
    self.temp_attrs = None

    default_attrs = {}
    '''
    default_attrs['cm']=0.281
    default_attrs['tau_refrac']=0.1
    default_attrs['v_spike']=-40.0
    default_attrs['v_reset']=-70.6
    default_attrs['v_rest']=-70.6
    default_attrs['tau_m']=9.3667
    default_attrs['i_offset']=0.0
    default_attrs['a']=4.0
    default_attrs['b']=0.0805
    default_attrs['delta_T']=2.0
    default_attrs['tau_w']=144.0
    default_attrs['v_thresh']=-50.4
    default_attrs['e_rev_E']=0.0
    default_attrs['tau_syn_E']=5.0
    default_attrs['e_rev_I']=-80.0
    default_attrs['tau_syn_I']=5.0
    default_attrs['spike_delta']=30
    default_attrs['scale']=0.5
    '''
    default_attrs['C']       = 281    # capacitance in pF ... this is 281*10^(-12) F
    default_attrs['g_L']     = 30     # leak conductance in nS
    default_attrs['E_L']     = -70.6  # leak reversal potential in mV ... this is -0.0706 V
    default_attrs['delta_T'] = 2      # slope factor in mV
    default_attrs['V_T']     = -50.4  # spike threshold in mV
    default_attrs['tau_w']   = 144    # adaptation time constant in ms
    default_attrs['V_peak']  = 300     # when to call action potential in mV
    default_attrs['b']       = 0.00805 # spike-triggered adaptation
    default_attrs['a']       = 4
    default_attrs['v_rest'] = -75
    self.default_attrs = default_attrs
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
    self.cont = Container(self,attrs = attrs)   
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
    stim = {'start':delay,'stop':duration+delay,'pA':amplitude}
    #self.neuron._setup(
    #    experiment_name='Single Neuron exhibiting tonic spiking',\
    #    T=tMax,\
    #    dt=1,\
    #    I_ext=stim)
    
    vm = self.cont.simulate(
        attrs=temp_attrs,\
        T=tMax,\
        dt=1,\
        stim=stim)#,v_rest=v_rest)
    vM = AnalogSignal(vm,
                          units = voltage_units,
                          sampling_period = 1*pq.ms)
    self.vM = vM
    vm = self.vM

    self.model.spikes = self.cont.spikes 
    def get_spike_count(self):
      return len(model.spikes)
    self.model.get_spike_count = self.get_spike_count

    return self.vM
