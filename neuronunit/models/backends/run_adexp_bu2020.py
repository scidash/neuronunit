

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

#try:
#import asciiplotlib as apl
#fig = apl.figure()
#ascii_plot = True
#import gc
#print('never gets here')
#ascii_plot = False
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

# Adaptive exponential integrate-and-fire neuron
'''
@jit
def aEIF(inputCurrent,V_peak = 40,
          tau_w = 144,
          V_T=-50.4, delta_T = 2,
          g_L=30, E_L=-70.6 ,a=1.0, 
          vr=-70.6/1000.0,C=0.281,b=0.0805):
    # Simulation parameters
    delta = 0.001                     # dt
    M     = 1    # number of neurons
    N     = len(inputCurrent)  # number of simulation points is determined by size of inputCurrent
    T     = np.linspace(0,N*delta,N) # time points corresponding to inputCurrent (same size as V, w, I)    
    # Thermodynamic parameters
    #kB   = 1.3806503*10**(-23.0)   # Boltzmann's constant
    #beta = 1.0/(kB*310.65)        # kB times T where T is in Kelvin    
    # Initialize variables
    V       = np.zeros((N,M))
    V[0] = [ vr/1000.0 for i in V[0] ]#      = np.zeros((N,M))
    w       = np.zeros((N,M))
    w[0] = [ b for i in V[0] ]#      = np.zeros((N,M))
    sptimes = [[] for _ in range(0,M)]
    V[0]    = vr/1000.0                # this gives us a chance to say what the membrane voltage starts at
                                # so we can draw initial conditions from the Boltzmann dist. later
    # Run model
    V_peak = 20/1000.0 
    E_L = E_L/1000.0
    for i in range(0,N-1):
        
        #V'=1/Cm*(-gL*(V-EL)+gL*DeltaT*exp((V-VT)/DeltaT)-w+muI+sqrt(2*DI)*eta)
        
        #w'=(-w+a*(V-EL))/tauw 
        V[i+1,:] = V[i,:] + (delta/C)*( -g_L*(V[i,:] - E_L) + \
          g_L*delta_T*np.exp((V[i,:] - V_T)/delta_T) - w[i,:] + inputCurrent[i+1]/1000.0) 
        w[i+1,:] = w[i,:] + (delta/tau_w)*(a*(V[i,:] - E_L) - w[i,:])
        # spiking mechanism
        ind = where(V[i+1,:] >= V_peak)
        if size(ind[0]) > 0:
            V[i+1,ind]      = E_L
            w[i+1,ind]      = w[i,ind] + b
            spikes[i+1,ind] = 1
            [sptimes[j].append(T[i+1]) for j in ind[0]]
    #V = [v_+V_peak for v_ in V]
    return [V,sptimes]    
'''
#https://github.com/ericjang/pyN/blob/master/pyN/Populations/AdExPopulation.py
@jit
def aEIF(inputCurrent,
    vr=-70.6,
    b=0.0805,
    delta_T = 2.0,
    V_T=-50.4,
    tau_w = 144,
    C=0.281,
    tau_m=9.3667,
    spike_delta=30,
    v_reset=-70.6):
  a = 4 # change me
  #compute v and adaptation resets
    # Simulation parameters
  dt = 0.00010                     # dt
  M     = 1    # number of neurons
  N     = len(inputCurrent)  # number of simulation points is determined by size of inputCurrent
  T     = np.linspace(0,N*dt,N) # time points corresponding to inputCurrent (same size as V, w, I)
  w = np.ones(N)
  v = np.ones(N)
  v[0] = -70.6/1000.0
  spike_raster = np.zeros((N, len(T)))
  spike_cnt=0
  #inputCurrent = np.array([i_/1000.0 for i_ in inputCurrent])

  for i in range(0,N-1):
    if i>0:
      #if np.nonzero(spike_raster[:,i-1] == True):
      prev_spiked = np.nonzero(spike_raster[:,i-1] == True)
      v[prev_spiked] = v_reset
      w[prev_spiked] += b
      #compute deltas and apply to state variables
    
    dv  = (((vr-v) + \
          delta_T*np.exp((v - V_T)/delta_T))/tau_m + \
          (inputCurrent - w)/C) *dt
    
    v += dv
    w += dt * (a*(v - V_T) - w)/tau_w * dt
    #decide whether to spike or not
    spiked = np.nonzero(v > V_T)
    if spiked:
      spike_cnt +=1
      v[spiked] = spike_delta
      spike_raster[spiked,i] = 1
    #inputCurrent[i] = 0
    #inputCurrent = np.zeros(inputCurrent.shape[0])
  v = [v_/1000.0 for v_ in v]
  return v,spike_cnt

  old
  for i in range(0,N-1):

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
  #p.I_ext = 0# 
  return p
  '''
@jit
def evaluate_vm(attrs,ie):#,dt,T):

  vm,spikes = aEIF(ie,
          V_T=attrs['V_T'],
          delta_T = attrs['delta_T'],
          vr=attrs['v_rest'],
          C=attrs['C'],
          b=attrs['b'])
  return vm,spikes
#def aEIF(inputCurrent,
#    vr=-70,b=0.0805,delta_T = 2,V_T=-50.4,tau_w = 144,
#    C=281,tau_m=9.3667,spike_delta=50,v_reset=-70.6):

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
    self.spike_rasterspike_raster = np.zeros((1, len_time_trace))
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
      self.attrs = attrs

      ie = [update_currents(stim, i=i, t=t, dt=dt) for i, t in enumerate(self.time_trace[1:],1) ]

      vm,spike_cnt = evaluate_vm(self.attrs,ie)#,self.dt,T)
      self.spike_cnt = spike_cnt
      
      return vm
'''    
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

class ADEXPBackend(Backend):
  name = 'ADEXP'
  def init_backend(self, attrs=None, DTC=None):
    self.model._backend.use_memory_cache = False
    self.vM = None
    self.attrs = attrs
    self.temp_attrs = None
    self.spike_cnt = 0
    default_attrs = {}
    default_attrs['C']       = 186    # capacitance in pF ... this is 281*10^(-12) F
    default_attrs['g_L']     = 30     # leak conductance in nS
    default_attrs['E_L']     = -70.6  # leak reversal potential in mV ... this is -0.0706 V
    default_attrs['delta_T'] = 2      # slope factor in mV
    default_attrs['V_T']     = -50.4  # spike threshold in mV
    default_attrs['tau_w']   = 144    # adaptation time constant in ms
    default_attrs['V_peak']  = 20     # when to call action potential in mV
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
    #thresh = threshold_detection(self.vM)
    #return len(thresh)
    return self.spike_cnt
  def set_attrs(self,attrs):
    self.default_attrs.update(attrs)
    attrs = self.default_attrs
    #if not hasattr(self.model,'attrs'):# is None:
    #    self.model.attrs = {}
    self.model.attrs.update(attrs)
    #else:
    #    self.model.attrs.update(attrs)    
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
    temp_attrs = copy.copy(self.model.attrs)

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
    #print(temp_attrs,'model params exist here')
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
    import matplotlib.pyplot as plt
    plt.plot(vm.times,vm.magnitude)
    plt.show()
    #fig = apl.figure()
    #fig.plot([float(t) for t in vm.times],[float(v) for v in vm], label=str('spikes: '), width=100, height=20)

    self.spike_cnt = self.cont.spike_cnt 
    #def get_spike_count(self):
    #  return len(model.spikes)
    #self.model.get_spike_count = self.get_spike_count

    return self.vM
