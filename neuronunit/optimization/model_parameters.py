import numpy as np
import os
from collections import OrderedDict

THIS_DIR = os.path.dirname(os.path.realpath(__file__))

path_params = {}
path_params['model_path'] = os.path.realpath(os.path.join(THIS_DIR,'..','models','NeuroML2','LEMS_2007One.xml'))
# Which Parameters
# https://www.izhikevich.org/publications/spikes.htm

import collections
import numpy as np

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





temp = {k:[] for k in ['C','k','vr','vt','vpeak','a','b','c','d']  }
for i,k in enumerate(temp.keys()):
    for v in type2007.values():
        temp[k].append(v[i])

explore_param = {k:(np.min(v),np.max(v)) for k,v in temp.items()}
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
