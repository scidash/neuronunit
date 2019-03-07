import numpy as np
import os
from collections import OrderedDict
from numpy import sqrt, pi
import collections
import numpy as np
from neuronunit.optimization import get_neab

import collections
from collections import OrderedDict
import numpy as np
'''
import pyNN
from pyNN import neuron
from pyNN.neuron import EIF_cond_exp_isfa_ista
import pickle
import pdb
import pyNN
from pyNN import neuron
from pyNN.neuron import EIF_cond_exp_isfa_ista
cell = neuron.create(EIF_cond_exp_isfa_ista())

'''
THIS_DIR = os.path.dirname(os.path.realpath(__file__))

path_params = {}
path_params['model_path'] = os.path.realpath(os.path.join(THIS_DIR,'..','models','NeuroML2','LEMS_2007One.xml'))

try:
    GLIF = pickle.load(open('glif_params.p','rb'))
except:
    from neuronunit.optimization import get_neab



MODEL_PARAMS = {}

from neurodynex.adex_model import AdEx

BAE = {}

# Parameters
BAE['C'] = 281 * AdEx.b2.units.pF
BAE['gL'] = 30 * AdEx.b2.units.nS
BAE['taum'] = C / gL
BAE['EL'] = -70.6 * AdEx.b2.units.mV
BAE['VT'] = -50.4 * AdEx.b2.units.mV
BAE['DeltaT'] = 2 * AdEx.b2.units.mV
BAE['Vcut'] = VT + 5 * DeltaT

# Pick an electrophysiological behaviour
RSBrian = [tauw, a, b, Vr = 144*AdEx.b2.units.ms, 4*AdEx.b2.units.nS, 0.0805*AdEx.b2.units.nA, -70.6*AdEx.b2.units.mV] # Regular spiking (as in the paper)
BurstBrian = [tauw,a,b,Vr=20*AdEx.b2.units.ms,4*AdEx.b2.units.ns,0.5*AdEx.b2.units.nA,VT+5*AdEx.b2.units.mV] # Bursting
FSBrian = [tauw,a,b,Vr=144*AdEx.b2.units.ms,2*C/(144*AdEx.b2.units.ms),0*AdEx.b2.units.nA,-70.6*AdEx.b2.units.mV] # Fast spiking


# https://github.com/NeuroML/NML2_LEMS_Examples/blob/master/PyNN.xml
EIF = {}
EIF_dic = cell[0].get_parameters()
EIF['cm'] = (EIF_dic['cm']-EIF_dic['cm']/2,EIF_dic['cm']+EIF_dic['cm']/2)
EIF['tau_m'] = (EIF_dic['tau_m']-EIF_dic['tau_m']/2,EIF_dic['tau_m']+EIF_dic['tau_m']/2)
EIF['b'] = (EIF_dic['b']-EIF_dic['b']/2,EIF_dic['b']+EIF_dic['b']/2)
EIF['a'] = (EIF_dic['a']-EIF_dic['a']/2,EIF_dic['a']+EIF_dic['a']/2)
EIF['v_spike'] = (25.0,-45.0)
EIF['v_thresh'] = (EIF_dic['v_thresh']-EIF_dic['v_thresh']/2,EIF_dic['v_thresh']+EIF_dic['v_thresh']/2)
EIF['v_rest'] = (EIF_dic['v_rest']-EIF_dic['v_rest']/2,EIF_dic['v_rest']+EIF_dic['v_rest']/2)
EIF['e_rev_E'] = (EIF_dic['e_rev_E']-EIF_dic['e_rev_E']/2,EIF_dic['e_rev_E']+EIF_dic['e_rev_E']/2)
#http://neuralensemble.org/docs/PyNN/_modules/pyNN/standardmodels/cells.html
EIF_cond_exp_isfa_ista_parameters = {
    'cm':         0.281,   # Capacitance of the membrane in nF
    'tau_refrac': 0.1,     # Duration of refractory period in ms.
    'v_spike':  30.0,     # Spike detection threshold in mV.
    'v_reset':  -70.6,     # Reset value for V_m after a spike. In mV.
    'v_rest':   -70.6,     # Resting membrane potential (Leak reversal potential) in mV.
    'tau_m':      9.3667,  # Membrane time constant in ms
    'i_offset':   0.0,     # Offset current in nA
    'a':          4.0,     # Subthreshold adaptation conductance in nS.
    'b':          0.0805,  # Spike-triggered adaptation in nA
    'delta_T':    2.0,     # Slope factor in mV
    'tau_w':    144.0,     # Adaptation time constant in ms
    'v_thresh': -50.4,     # Spike initiation threshold in mV
    'e_rev_E':    0.0,     # Excitatory reversal potential in mV.
    'tau_syn_E':  5.0,     # Decay time constant of excitatory synaptic conductance in ms.
    'e_rev_I':  -80.0,     # Inhibitory reversal potential in mV.
    'tau_syn_I':  5.0}     # Decay time constant of the inhibitory synaptic conductance in ms.


#recordable = ['spikes', 'v', 'w', 'gsyn_exc', 'gsyn_inh']
EIF_cond_exp_isfa_ista_initial_values = {
    'v': -70.6,  # 'v_rest',
    'w': 0.0,
    'gsyn_exc': 0.0,
    'gsyn_inh': 0.0,
}
MODEL_PARAMS['PYNN'] = EIF







GLIF_RANGE = {'El_reference': [-0.08569469261169435, -0.05463626766204832], 'C': [3.5071610042390286e-11, 7.630189223327981e-10], 'asc_amp_array': [[-6.493692083311101e-10, 1.224690033604069e-09], [1.0368081669092888e-08, -4.738879134819112e-08]], 'init_threshold': [0.009908733642683513, 0.04939040414685865], 'threshold_reset_method': {'params': {}, 'name': 'inf'}, 'th_inf': [0.009908733642683513, 0.04939040414685865], 'spike_cut_length': [20, 199], 'init_AScurrents': [[0.0, 0.0], [0.0, 0.0]], 'init_voltage': [-70.0, 0.0], 'threshold_dynamics_method': {'params': {}, 'name': 'inf'}, 'voltage_reset_method': {'params': {}, 'name': 'zero'}, 'extrapolation_method_name': ['endpoints', 'endpoints'], 'dt': [5e-05, 5e-05], 'voltage_dynamics_method': {'params': {}, 'name': 'linear_forward_euler'}, 'El': [0.0, 0.0], 'asc_tau_array': [[0.01, 0.0033333333333333335], [0.3333333333333333, 0.1]], 'R_input': [27743752.593817078, 1792774179.3647704], 'AScurrent_dynamics_method': {'params': {}, 'name': 'none'}, 'AScurrent_reset_method': {'params': {}, 'name': 'none'}, 'dt_multiplier': [10, 10], 'th_adapt': None, 'coeffs': {'a': 1, 'C': 1, 'b': 1, 'G': 1, 'th_inf': 1.0212937371199788, 'asc_amp_array': [1.0, 1.0]}, 'type': ['GLIF', 'GLIF']}

MODEL_PARAMS['GLIF'] = GLIF_RANGE
MODEL_PARAMS['GLIF']['init_AScurrents'] = [0,0]

# Which Parameters
# https://www.izhikevich.org/publications/spikes.htm
type2007 = collections.OrderedDict([
  #              C    k     vr  vt vpeak   a      b   c    d  celltype
  ('RS',        (100, 0.7,  -60, -40, 35, 0.03,   -2, -50,  100,  1)),
  ('IB',        (150, 1.2,  -75, -45, 50, 0.01,   5, -56,  130,   2)),
  ('LTS',       (100, 1.0,  -56, -42, 40, 0.03,   8, -53,   20,   4)),
  ('TC',        (200, 1.6,  -60, -50, 35, 0.01,  15, -60,   10,   6)),
  ('TC_burst',  (200, 1.6,  -60, -50, 35, 0.01,  15, -60,   10,   6))])
  #('CH',        (50,  1.5,  -60, -40, 25, 0.03,   1, -40,  150,   3)),
  #('FS',        (20,  1.0,  -55, -40, 25, 0.2,   -2, -45,  -55,   5)),

  #('RTN',       (40,  0.25, -65, -45,  0, 0.015, 10, -55,   50,   7)),
  #('RTN_burst', (40,  0.25, -65, -45,  0, 0.015, 10, -55,   50,   7))])


# http://www.physics.usyd.edu.au/teach_res/mp/mscripts/
# ns_izh002.m

'''
temp = {k:[] for k in ['C','k','vr','vt','vPeak','a','b','c','d']  }
for i,k in enumerate(temp.keys()):
    for v in type2007.values():
        temp[k].append(v[i])

explore_param = {k:(np.min(v),np.max(v)) for k,v in temp.items()}
#IZHI_PARAMS = OrderedDict(explore_param)
'''


# Fast spiking cannot be reproduced as it requires modifications to the standard Izhi equation,
# which are expressed in this mod file.
# https://github.com/OpenSourceBrain/IzhikevichModel/blob/master/NEURON/izhi2007b.mod

trans_dict = OrderedDict([(k,[]) for k in ['C','k','vr','vt','vPeak','a','b','c','d']])

#OrderedDict
for i,k in enumerate(trans_dict.keys()):
    for v in type2007.values():
        trans_dict[k].append(v[i])


reduced_cells = OrderedDict([(k,[]) for k in ['RS','IB','LTS','TC','TC_burst']])

for index,key in enumerate(reduced_cells.keys()):
    reduced_cells[key] = {}
    for k,v in trans_dict.items():
        reduced_cells[key][k] = v[index]

IZHI_PARAMS = {k:(np.min(v),np.max(v)) for k,v in trans_dict.items()}
IZHI_PARAMS = OrderedDict(IZHI_PARAMS)
MODEL_PARAMS['RAW'] = IZHI_PARAMS
'''
# page 1
# http://www.rctn.org/vs265/izhikevich-nn03.pdf
'''
def transcribe_units(input_dic):
    '''
    Move between OSB unit conventions and NEURON unit conventions.
    '''
    # From OSB models
    mparams = {}
    mparams['a'] = 0.03
    mparams['b'] = -2
    mparams['C'] = 100
    mparams['c'] = -50
    mparams['vr'] = -60
    mparams['vt'] = -40
    mparams['vpeak'] = 35
    mparams['k'] = 0.7
    mparams['d'] = 100
    # FROM the MOD file.
    vanilla_NRN = {}
    #vanilla_NRN['v0'] = -60# (mV)
    vanilla_NRN['k'] = 7.0E-4# (uS / mV)
    vanilla_NRN['vr'] = -60# (mV)
    vanilla_NRN['vt'] = -40# (mV)
    vanilla_NRN['vpeak'] = 35# (mV)
    vanilla_NRN['a'] = 0.03# (kHz)
    vanilla_NRN['b'] = -0.002# (uS)
    vanilla_NRN['c'] = -50# (mV)
    vanilla_NRN['d'] = 0.1# (nA)
    vanilla_NRN['C'] = 1.0E-4# (microfarads)

    m2m = {}
    for k,v in vanilla_NRN.items():
        m2m[k] = vanilla_NRN[k]/mparams[k]

    input_dic['vpeak'] = input_dic['vPeak']
    input_dic.pop('vPeak', None)
    input_dic.pop('dt', None)
    for k,v in input_dic.items():
        input_dic[k] = v * m2m[k]
    return input_dic
