

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
import cython

from elephant.spike_train_generation import threshold_detection


import numpy as np
from numba import jit

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
def update_currents(amp, i, t, dt,start,stop):   
  scalar = 0
  if start <= t <= stop:
    scalar = amp
  return scalar

#@jit
#def I_inj(t,delay,duration,amplitude):#
#	return 0*(t<delay) +amplitude*(t>delay) +0*(t>delay+duration) # a scalar value.

'''
V[0]    = v0                # this gives us a chance to say what the membrane voltage starts at
                                # so we can draw initial conditions from the Boltzmann dist. later
# Run model
for i in xrange(N-1):
    V[i+1,:] = V[i,:] + (delta/C)*( -g_L*(V[i,:] - E_L) + g_L*delta_T*np.exp((V[i,:] - V_T)/delta_T) - w[i,:] + inputCurrent[:,i+1]) + np.sqrt(noiseT)*randn(1,M)
    w[i+1,:] = w[i,:] + (delta/tau_w)*(a*(V[i,:] - E_L) - w[i,:])

    # spiking mechanism
    ind = where(V[i+1,:] >= V_peak)
    if size(ind[0]) > 0:
        V[i+1,ind]      = E_L
        w[i+1,ind]      = w[i,ind] + b
        spikes[i+1,ind] = 1
        [sptimes[j].append(T[i+1]) for j in ind[0]]

return [V.transpose(),w.transpose(),spikes.transpose(),sptimes]  
'''
'''
@jit
def update_state(i, T, t, dt,I_ext,v,w,spike_raster,v_reset,b,a,spike_delta,v_rest,tau_m,tau_w,v_thresh,delta_T,cm):
 
  prev_spiked = np.nonzero(spike_raster[:,i-1] == True)
  v[prev_spiked] = v_reset
  w[prev_spiked] += b
  #compute deltas and apply to state variables
  dv  = (((v_rest-v) + \
        delta_T*np.exp((v - v_thresh)/delta_T))/tau_m + \
        (I_ext - w)/cm) *dt
  v += dv
  w += dt * (a*(v - v_rest) - w)/tau_w * dt
  #decide whether to spike or not
  spiked = np.nonzero(v > v_thresh)
  ### https://github.com/lmcintosh/masters-thesis/blob/master/mastersFunctions.py
  #ind = np.where(v >= v_thresh)
  #if  np.size(ind[0]) > 0:
  #pass
    #print(ind)
    #spikes[i+1,ind] = 1
  ###
  v[spiked] = spike_delta
  spike_raster[spiked,i] = 1

  return v,w,spike_raster,len(spike_raster)
'''  
#@timer
#
@cython.boundscheck(False)
@cython.wraparound(False)
@jit(nopython=True)
def update_state_new_new(T, dt,v,w,
                        v_reset,b,a,spike_raster,
                        spike_delta,v_rest,tau_m,
                        tau_w,v_thresh,delta_T,cm,time_trace,amp,start,stop):
  #print(v,w)
  i = 0
  spike_raster = [0 for ix in range(0,len(time_trace))]
  vm = []
  
  #v = v[0]
  #w = w[0]
  for t_ind in range(0,len(time_trace)):
    t = time_trace[t_ind]
    if i!=0:
      I_scalar = 0
      if start <= t <= stop:
        I_scalar = amp
      #else:
      #  I_scalar = 0

      #for indx in range(0,i-1):
      if spike_raster[i-1]:
        v = v_reset
        w += b
      dv  = (((v_rest-v) + \
            delta_T*np.exp((v - v_thresh)/delta_T))/tau_m + \
            (I_scalar - w)/cm) *dt
      v += dv
      w += dt * (a*(v - v_rest) - w)/tau_w * dt
      if v>v_thresh:
        v = spike_delta
        spike_raster[i] = 1
      else:
        spike_raster[i] = 0
      vm.append(v)
    i+=1
  #print(vm)
  return len(spike_raster),vm
'''
#@timer
#@jit(nopython=True)
#@cython.boundscheck(False)
#@cython.wraparound(False)
def update_state_new(T, dt,v,w,spike_raster,v_reset,b,a,spike_delta,v_rest,tau_m,tau_w,v_thresh,delta_T,cm,time_trace,amp,start,stop):
  i = 0
  vm = []#np.array([0 for i in time_trace])
  for t_ind in range(0,len(time_trace)):
    t = time_trace[t_ind]
    if i!=0:
      I_scalar = 0
      if start <= t <= stop:
        I_scalar = amp

      prev_spiked = np.nonzero(spike_raster[:,i-1] == True)
      v[prev_spiked] = v_reset
      w[prev_spiked] += b
      #compute deltas and apply to state variables
      dv  = (((v_rest-v) + \
            delta_T*np.exp((v - v_thresh)/delta_T))/tau_m + \
            (I_scalar - w)/cm) *dt
      v += dv
      w += dt * (a*(v - v_rest) - w)/tau_w * dt
      #decide whether to spike or not
      spiked = np.nonzero(v > v_thresh)

      ### https://github.com/lmcintosh/masters-thesis/blob/master/mastersFunctions.py
      v[spiked] = spike_delta
      spike_raster[spiked,i] = 1
      vm.append(v[0])
    i+=1


  return v,w,spike_raster,len(spike_raster),vm
'''
@jit#(nopython=True)
#@cython.boundscheck(False)
#@cython.wraparound(False)
def evaluate_vm_new(time_trace,dt,T,v,w,b,a,spike_delta,spike_raster,v_reset,v_rest,tau_m,tau_w,v_thresh,delta_T,cm,amp,start,stop):
  n_spikes,vm = update_state_new_new(T=T, 
                                    dt=dt,
                                    v=v,
                                    w=w,
                                    spike_raster=spike_raster,
                                    v_reset=v_reset,
                                    b=b,a=a,
                                    spike_delta=spike_delta,
                                    v_rest=v_rest,tau_m=tau_m,tau_w=tau_w,
                                    v_thresh=v_thresh,
                                    delta_T =delta_T,cm=cm,time_trace=time_trace,
                                    amp = amp,start = start,stop = stop)

  
  return vm,n_spikes
'''
@jit(nopython=True)
@cython.boundscheck(False)
@cython.wraparound(False)
def evaluate_vm(time_trace,dt,T,v,w,b,a,spike_delta,spike_raster,v_reset,v_rest,tau_m,tau_w,v_thresh,delta_T,cm,amp,start,stop):
  vm = []
  i = 0
  #for t in time_trace[1:]:
  for t in time_trace:
    if i!=0:
      I_scalar = update_currents(amp, i=i, t=t, dt=dt,start=start,stop=stop)
      v,w,spike_raster,n_spikes = update_state(i=i, T=T, t=t, dt=dt,
                                        I_ext=I_scalar,v=v,
                                        w=w,spike_raster=spike_raster,
                                        v_reset=v_reset,
                                        b=b,a=a,spike_delta=spike_delta,
                                        v_rest=v_rest,tau_m=tau_m,tau_w=tau_w,
                                        v_thresh=v_thresh, delta_T =delta_T,cm=cm)
      vm.append(v[0])
    i+=1
  return vm,n_spikes
'''



class ADEXPBackend(Backend):
  name = 'ADEXP'
  def init_backend(self, attrs={}, DTC=None):
    self.model._backend.use_memory_cache = False
    self.attrs = attrs
    #print(self.attrs)
    #self.current_src_name = current_src_name
    #self.cell_name = cell_name
    self.vM = None
    #self.attrs = attrs
    #self.debug = debug
    self.temp_attrs = None
    #def __init__(self, name, cm=0.281,
    # v_spike=-40.0, v_reset=-70.6, v_rest=-70.6, tau_m=9.3667, a=4.0, b=0.0805, delta_T=2.0,tau_w=144.0,
    # v_thresh=-50.4, spike_delta=30):

    BAE1 = {}
    BAE1['cm']=0.281
    BAE1['v_spike']=-40.0
    BAE1['v_reset']=-70.6
    BAE1['v_rest']=-70.6
    BAE1['tau_m']=9.3667
    BAE1['a']=4.0
    BAE1['b']=0.0805

    BAE1['delta_T']=2.0
    BAE1['tau_w']=144.0
    BAE1['v_thresh']=-50.4
    BAE1['spike_delta']=30

    self.default_attrs = BAE1
    super(ADEXPBackend,self).init_backend()

    if type(DTC) is not type(None):
        if type(DTC.attrs) is not type(None):
            print('gets here')
            #self.set_attrs(DTC.attrs)
            self.attrs = attrs
        if type(DTC.attrs) is type(None):
          self.attrs = self.default_attrs
          #self.set_attrs(self.default_attrs)

  def simulate(self, attrs={}, T=50,dt=0.25,integration_time=30, I_ext={},spike_delta=50):
    spike_delta = spike_delta
    N = 1
    w = 1#np.ones(N)

    dt         = dt
    time_trace = np.arange(0,T+dt,dt)#time array
    #I_ext
    len_time_trace = len(time_trace)
    #self.T = T
    integration_time = 30.0
    spike_raster = np.zeros((1, len_time_trace))
    #self.psc = np.zeros((self.N, len_time_trace))
    integrate_window = np.int(np.ceil(integration_time/dt))
    #dt = dt

    if integrate_window > len_time_trace:
      integrate_window = len_time_trace #if we are running a short simulation then integrate window will overflow available time slots!
    attrs = self.default_attrs
    attrs.update(self.model.attrs)
    v_rest =  attrs['v_rest']
    v = v_rest 
    #v = v_rest #voltage trace
    v_reset = attrs['v_reset']
    tau_m = attrs['tau_m']
    delta_T = attrs['delta_T']
    spike_delta = attrs['spike_delta']

    a = attrs['a']
    b = attrs['b']
    v_thresh = attrs['v_thresh']
    cm = attrs['cm']
    tau_w = attrs['tau_w']

    
    amp = I_ext['pA']
    start = I_ext['start']
    stop = I_ext['stop']


    #vm,n_spikes = evaluate_vm(time_trace,dt,T,v,w,b,a,
    #                  spike_delta,spike_raster,v_reset,v_rest,
    #                  tau_m,tau_w,v_thresh,delta_T,cm,amp,start,stop)

    vm,n_spikes = evaluate_vm_new(time_trace,dt,T,v,w,b,a,
                      spike_delta,spike_raster,v_reset,v_rest,
                      tau_m,tau_w,v_thresh,delta_T,cm,amp,start,stop)
    #print(vm)
    return vm,n_spikes


  def get_spike_count(self):
    return self.n_spikes
    #thresh = threshold_detection(self.vM)
    #assert self.n_spikes == len(thresh)
    #return len(thresh)
  '''
    def set_attrs(self,attrs):
      self.default_attrs.update(attrs)
      attrs = self.default_attrs
      self.attrs = attrs
      
      #if not hasattr(self.model,'attrs'):# is None:
      #    self.model.attrs = {}
      #    self.model.attrs.update(attrs)
      #else:
      self.model.attrs.update(attrs)
      
      #self.model_ = ADEXPopulation(name='Charly the Neuron',N=1)
      #self.neuron = Network(populations=[self.model_],attrs = attrs)
    
      return self
  '''
 # def set_attrs(self, attrs):
 #     self.attrs = attrs
 #     self.model.attrs.update(attrs)

  def set_attrs(self,attrs):
      self.default_attrs.update(attrs)
      attrs = self.default_attrs
      self.attrs = attrs
      if not hasattr(self.model,'attrs'):
        self.model.attrs = {}
      self.model.attrs.update(attrs)
      #print(self.model._backend.attrs)
      #import pdb
      #pdb.set_trace()


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

  #@timer
  #@cython.boundscheck(False)
  #@cython.wraparound(False)
  #@jit
  def inject_square_current(self,current):#, section = None, debug=False):
    """Inputs: current : a dictionary with exactly three items, whose keys are: 'amplitude', 'delay', 'duration'
    Example: current = {'amplitude':float*pq.pA, 'delay':float*pq.ms, 'duration':float*pq.ms}}
    where \'pq\' is a physical unit representation, implemented by casting float values to the quanitities \'type\'.
    Description: A parameterized means of applying current injection into defined
    Currently only single section neuronal models are supported, the neurite section is understood to be simply the soma.
    """

    temp_attrs =  self.attrs

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

    vm,n_spikes = self.simulate(
        attrs=temp_attrs,\
        T=tMax,\
        dt=0.25,\
        I_ext=stim)#,v_rest=v_rest)
    vM = AnalogSignal(vm,
                          units = voltage_units,
                          sampling_period = 0.25*pq.ms)
    #import asciiplotlib as apl
    #fig = apl.figure()
    #fig.plot([float(f) for f in vM.times], [float(f) for f in vM], width=100, height=20)
    #fig.show()

    self.vM = vM
    self.n_spikes = n_spikes
    return vM
