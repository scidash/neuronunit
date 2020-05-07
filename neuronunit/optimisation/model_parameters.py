import numpy as np
import os
from collections import OrderedDict
from numpy import sqrt, pi
import collections
import numpy as np

import numpy as np
import pickle


def check_if_param_stradles_boundary(opt,model_type):
    for k,v in MODEL_PARAMS[model_type].items(): 
        print(v,opt.attrs[k],k)

MODEL_PARAMS = {}
THIS_DIR = os.path.dirname(os.path.realpath(__file__))
path_params = {}
path_params['model_path'] = os.path.realpath(os.path.join(THIS_DIR,'..','models','NeuroML2','LEMS_2007One.xml'))
OSB_HH_attrs = {
        'E_L' : -68.9346,
        'E_K' : -90.0,
        'E_Na' : 120.0,
        'g_L' : 0.1,
        'g_K' : 36,
        'g_Na' : 200,
        'C_m' : 1.0,
        'vr':-68.9346
        }
OSB_HH_attrs['Vr'] = OSB_HH_attrs['E_L']
HH_dic1 = { k:(float(v)-0.25*float(v),float(v)+0.25*float(v)) for k,v in OSB_HH_attrs.items() }
MODEL_PARAMS['OSB_HH_attrs'] = HH_dic1



MODEL_PARAMS['NEURONHH'] = {'gnabar':0.12,\
                            'gkbar':0.036,\
                            'Ra':100,\
                            'L':12.6157,\
                            'diam':12.6157,\
                            'gkbar':0.036,\
                            'el':-54.3,\
                            'gk':0.0,\
                            'gl':0.0003,\
                            'ena':50.0,\
                            'ek':-77.0,\
                            'vr':-65,\
                            'cm':1.0,\
                            'ena':50.0,\
                            'ek':-77}
MODEL_PARAMS['NEURONHH'] = { k:(float(v)-0.95*float(v),float(v)+0.95*float(v)) for k,v in MODEL_PARAMS['NEURONHH'].items() }
MODEL_PARAMS['NEURONHH']['gkbar'] = (0.0018,0.1)
MODEL_PARAMS['NEURONHH']['diam'] = (0.630785,50.0)
MODEL_PARAMS['NEURONHH']['gnabar'] = (0.006,0.2539)

# Which Parameters

# https://www.izhikevich.org/publications/spikes.htm
type2007 = collections.OrderedDict([
  #              C    k     vr  vt vpeak   a      b   c    d  celltype
  ('RS',        (100, 0.7,  -60, -40, 35, 0.03,   -2, -50,  100,  1)),
  ('IB',        (150, 1.2,  -75, -45, 50, 0.01,   5, -56,  130,   2)),
  ('TC',        (200, 1.6,  -60, -50, 35, 0.01,  15, -60,   10,   6)),
  ('TC_burst',  (200, 1.6,  -60, -50, 35, 0.01,  15, -60,   10,   6)),
  ('LTS',       (100, 1.0,  -56, -42, 40, 0.03,   8, -53,   20,   4)),
  ('CH',        (50,  1.5,  -60, -40, 25, 0.03,   1, -40,  150,   3))])

# http://www.physics.usyd.edu.au/teach_res/mp/mscripts/
# ns_izh002.m


temp = {k:[] for k in ['C','k','vr','vt','vPeak','a','b','c','d']  }
for i,k in enumerate(temp.keys()):
    for v in type2007.values():
        temp[k].append(v[i])
explore_param = {k:(np.min(v),np.max(v)) for k,v in temp.items()}
#explore_param['b'] = [-2,8]


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

IZHI_PARAMS['C'] = (40,670)
IZHI_PARAMS['k'] = (0.7, 2.5)
IZHI_PARAMS['vt'] = (-70, -40)
IZHI_PARAMS['a'] = (0.001, 0.03)
IZHI_PARAMS['b'] = (-2, 28)
IZHI_PARAMS['c'] = (-90, -40)
IZHI_PARAMS['vPeak'] = (20,100)

IZHI_PARAMS = OrderedDict(IZHI_PARAMS)
MODEL_PARAMS['RAW'] = IZHI_PARAMS
