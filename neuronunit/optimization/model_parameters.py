import numpy as np
import os
from collections import OrderedDict
import collections
import numpy as np

THIS_DIR = os.path.dirname(os.path.realpath(__file__))

path_params = {}
path_params['model_path'] = os.path.realpath(os.path.join(THIS_DIR,'..','models','NeuroML2','LEMS_2007One.xml'))
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

temp = collections.OrderedDict()
temp['C'] =[]
temp['k'] = []
temp['vr'] = []
temp['vt'] = []
temp['vpeak'] = []
temp['a'] = []
temp['b'] = []
temp['c'] = []
temp['d'] = [] 

temp_list = list(temp.keys())

for i,k in enumerate(temp_list):
    for v in type2007.values():
        temp[k].append(v[i])


model_params = OrderedDict()
for k,v in temp.items():
    model_params[k] = (np.min(v),np.max(v)) 


remap = ['C','k','b','d' ]
for k in model_params.keys():
    if k in remap:
        model_params[k] = np.linspace(float(model_params[k][0])*(10**-4),float(model_params[k][1])*(10**-4),9)
    else:
        model_params[k] = np.linspace(float(model_params[k][0]),float(model_params[k][1]),9)
    if str('vr') in remap:
        model_params['vr'] = np.linspace(float(model_params[k][0]-30.0), 0.0,9)
    if str('b') in remap:
        model_params['b'] = np.linspace(float(-1.0*0.02),float(np.abs(model_params[k][1]))*(10),9)
    if str('a') in remap:
        model_params['a'] = np.linspace(float(-1.0*np.abs(model_params[k][1]))*(10**-1),float(model_params[k][1])*(10**-2),9)


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

+model_params['a'] = np.linspace(0.0,0.945,9)
+model_params['b'] = np.linspace(-3.5*10E-10,-0.5*10E-9,9)
+model_params['vpeak'] =np.linspace(0.0,80.0,9)

+model_params['k'] = np.linspace(7.0E-4-+7.0E-5,7.0E-4+70E-5,9)
+model_params['C'] = np.linspace(1.00000005E-4-1.00000005E-5,1.00000005E-4+1.00000005E-5,9)
+model_params['c'] = np.linspace(-55,-60,9)
+model_params['d'] = np.linspace(0.050,0.2,9)
+model_params['v0'] = np.linspace(-85.0,-15.0,9)
+model_params['vt'] =  np.linspace(-70.0,0.0,9)
'''
