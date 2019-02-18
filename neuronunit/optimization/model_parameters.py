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

import pyNN
from pyNN import neuron
from pyNN.neuron import EIF_cond_exp_isfa_ista

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
path_params = {}
path_params['model_path'] = os.path.realpath(os.path.join(THIS_DIR,'..','models','NeuroML2','LEMS_2007One.xml'))


cell = neuron.create(EIF_cond_exp_isfa_ista())

# https://github.com/NeuroML/NML2_LEMS_Examples/blob/master/PyNN.xml
EIF = {}
EIF_dic = cell[0].get_parameters()
EIF['cm'] = (EIF_dic['cm']-EIF_dic['cm']/2,EIF_dic['cm']+EIF_dic['cm']/2)
EIF['tau_m'] = (EIF_dic['tau_m']-EIF_dic['tau_m']/2,EIF_dic['tau_m']+EIF_dic['tau_m']/2)
EIF['b'] = (EIF_dic['b']-EIF_dic['b']/2,EIF_dic['b']+EIF_dic['b']/2)
EIF['a'] = (EIF_dic['a']-EIF_dic['a']/2,EIF_dic['a']+EIF_dic['a']/2)
EIF['v_spike'] = (25.0-45.0)
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
    'tau_syn_I':  5.0,     # Decay time constant of the inhibitory synaptic conductance in ms.

#recordable = ['spikes', 'v', 'w', 'gsyn_exc', 'gsyn_inh']
EIF_cond_exp_isfa_ista_initial_values = {
    'v': -70.6,  # 'v_rest',
    'w': 0.0,
    'gsyn_exc': 0.0,
    'gsyn_inh': 0.0,
}

    # Which Parameters
# https://www.izhikevich.org/publications/spikes.htm
type2007 = collections.OrderedDict([
  #              C    k     vr  vt vpeak   a      b   c    d  celltype
  ('RS',        (100, 0.7,  -60, -40, 35, 0.03,   -2, -50,  100,  1)),
  ('IB',        (150, 1.2,  -75, -45, 50, 0.01,   5, -56,  130,   2)),
  ('CH',        (50,  1.5,  -60, -40, 25, 0.03,   1, -40,  150,   3)),
  ('LTS',       (100, 1.0,  -56, -42, 40, 0.03,   8, -53,   20,   4)),
  ('FS',        (20,  1.0,  -55, -40, 25, 0.2,   -2, -45,  -55,   5)),
  ('TC',        (200, 1.6,  -60, -50, 35, 0.01,  15, -60,   10,   6)),
  ('TC_burst',  (200, 1.6,  -60, -50, 35, 0.01,  15, -60,   10,   6)),
  ('RTN',       (40,  0.25, -65, -45,  0, 0.015, 10, -55,   50,   7)),
  ('RTN_burst', (40,  0.25, -65, -45,  0, 0.015, 10, -55,   50,   7))])


# http://www.physics.usyd.edu.au/teach_res/mp/mscripts/
# ns_izh002.m


'''
temp = {k:[] for k in ['C','k','vr','vt','vpeak','a','b','c','d']  }
for i,k in enumerate(temp.keys()):
    for v in type2007.values():
        temp[k].append(v[i])

explore_param = {k:(np.min(v),np.max(v)) for k,v in temp.items()}
model_params = OrderedDict(explore_param)

'''


# Fast spiking cannot be reproduced as it requires modifications to the standard Izhi equation,
# which are expressed in this mod file.
# https://github.com/OpenSourceBrain/IzhikevichModel/blob/master/NEURON/izhi2007b.mod
reduced_dict = OrderedDict([(k,[]) for k in ['C','k','vr','vt','vPeak','a','b','c','d']])

#OrderedDict
for i,k in enumerate(reduced_dict.keys()):
    for v in type2007.values():
        reduced_dict[k].append(v[i])

reduced_cells = OrderedDict([(k,[]) for k in ['RS','IB','LTS','TC','TC_burst']])

for index,key in enumerate(reduced_cells.keys()):
    reduced_cells[key] = {}
    for k,v in reduced_dict.items():
        reduced_cells[key][k] = v[index]

explore_param = {k:(np.min(v),np.max(v)) for k,v in reduced_dict.items()}
model_params = OrderedDict(explore_param)

# page 1
# http://www.rctn.org/vs265/izhikevich-nn03.pdf




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

#print(pred0,pred1)

# General parameters

SEED_LTS = 428577
SEED_CONN = 193566
SEED_GEN = 983651

DT = 0.1                                        # (ms) Time step
TSTART  = 0
TSTOP   = 5000
V_INIT  = -60.0

# Cell parameters

LENGTH          = sqrt(20000/pi)                # in um
DIAMETER        = sqrt(20000/pi)                # in um
AREA            = 1e-8 * pi * LENGTH * DIAMETER # membrane area in cm2
TAU             = 20                            # time constant in ms
CAPACITANCE     = 1                             # capacitance in muF/cm2
G_L             = 1e-3 * CAPACITANCE / TAU      # leak conductance in S/cm2
V_REST          = -60                           # resting potential

a_RS            = 0.001
b_RS            = 0.1   # full adaptation
b_RS            = 0.005 # weaker adaptation
a_LTS           = 0.02
b_LTS           = 0.0
a_FS            = 0.001
b_FS            = 0.0

TAU_W           = 600
DELTA           = 2.5

# Spike parameters

VTR             = -50           # threshold in mV
VTOP            = 40            # top voltage during spike in mV
VBOT            = -60           # reset voltage in mV
REFRACTORY      = 5.0/2         # refractory period in ms (correction for a bug in IF_CG4)

# Synapse parameters

RS_parameters = {
    'cm': 1000*AREA*CAPACITANCE, 'tau_m': TAU, 'v_rest': V_REST,
    'v_thresh': VTR, 'tau_refrac': REFRACTORY+DT,
    'v_reset': VBOT, 'v_spike': VTR+1e-6, 'a': 1000.0*a_RS, 'b': b_RS,
    'tau_w': TAU_W, 'delta_T': DELTA
}

LTS_parameters = RS_parameters.copy()
LTS_parameters.update({'a': 1000.0*a_LTS, 'b': b_LTS}) # 1000 is for uS --> nS
FS_parameters = RS_parameters.copy()
FS_parameters.update({'a': 1000.0*a_FS, 'b': b_FS})
#print(model_params)
'''

https://www.izhikevich.org/publications/izhikevich.m

Model parameters and dimensions:
t                      time     [ms]
C                    membrane capacitance     [pF = pA.ms.mV-1]
v                     membrane potential     [mV]
         rate of change of membrane potential     [mV.ms-1 = V.s-1]
       capacitor current     [pA]
vr                   resting membrane potential     [mV]
vt                   instantaneous threshold potential     [mV]
k                    constant (“1/R”)     [pA.mV-1   (10-9 Ω-1)]
u                   recovery variable     [pA]
S                   stimulus (synaptic: excitatory or inhibitory, external, noise)     [pA]
       rate of change of recovery variable     [pA.ms-1]
a                     recovery time constant     [ms-1]
b                  constant  (“1/R”)    [pA.mV-1   (10-9 Ω-1) ]
c                  potential reset value     [mV]
d                 outward minus inward currents activated during the spike and affecting the after-spike behavior     [pA]
vpeak          spike cutoff value     [mV]
+model_params['vr'] = np.linspace(-95.0,-30.0,9)

''';
