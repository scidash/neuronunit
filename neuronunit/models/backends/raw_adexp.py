import numpy as np
import numba
from numba import jit
# https://github.com/ericjang/pyN/blob/master/pyN/Populations/AdExPopulation.py
class AdEx():
  def __init__(self, cm=0.281, tau_refrac=0.1, v_spike=-40.0, v_reset=-70.6, v_rest=-70.6, tau_m=9.3667, i_offset=0.0, a=4.0, b=0.0805, delta_T=2.0,tau_w=144.0,v_thresh=-50.4,e_rev_E=0.0, tau_syn_E=5.0, e_rev_I=-80.0, tau_syn_I=5.0, N=1, tau_psc=5.0, connectivity=None, spike_delta=30,scale=0.5):
    """
    AdEx Constructor
    """
    def __init__(self, N, tau_psc, connectivity, spike_delta, v_reset,scale):
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
    @jit
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

        #self.update_psc(i)
        #reset i_ext
        self.I_ext = np.zeros(self.I_ext.shape[0])

    def Isyn(self,postsyn,t_diff):
        #this function is pretty much the same but tau_syn_E not necessarily == tau_syn_I
        t[np.nonzero(t < 0)] = 0
        #in order for a good modicum of current to even be applied, t must be negative!
        #note that these currents are positive but once applied to synapses, can become negative (inhibitory)
        return t*np.exp(-t/self.tau_syn_E)