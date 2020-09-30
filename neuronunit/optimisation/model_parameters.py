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
HH_dic1 = { k:(float(v)-0.75*float(v),float(v)+0.75*float(v)) for k,v in OSB_HH_attrs.items() }
HH_dic1 = { k:sorted(v) for k,v in HH_dic1.items() }

MODEL_PARAMS['OSB_HH_attrs'] = HH_dic1

 #12.6157,\
#gnabar_hh=0.12 gkbar_hh=0.036 gl_hh=0.0003 el_hh=-54.3
MODEL_PARAMS['NEURONHH'] = {'gnabar':0.12,\
                            'gkbar':0.036,\
                            'Ra':35,\
                            'L':100,\
                            'diam':500,\
                            'el':-54.3,\
                            'gl':0.0003,\
                            'ena':50.0,\
                            'ek':-77.0,\
                            'cm':1,\
                            'ena':50.0,\
                            'ek':-77}

MODEL_PARAMS['NEURONHH'] = { k:(float(v)-0.75*float(np.abs(v)),float(v)+0.75*float(v)) for k,v in MODEL_PARAMS['NEURONHH'].items() }
MODEL_PARAMS['NEURONHH']['L'] = [1,250]
MODEL_PARAMS['NEURONHH']['diam'] = [1,850]
MODEL_PARAMS['NEURONHH']['el'] = [-100,-40]

MODEL_PARAMS['NEURONHH']['ek'] = [-100,-40]
MODEL_PARAMS['NEURONHH'] = { k:sorted(v) for k,v in MODEL_PARAMS['NEURONHH'].items() }

# Which Parameters

HH_attrs = {
        'E_L' : -68.9346,
        'E_K' : -90.0,
        'E_Na' : 120.0,
        'g_L' : 0.1,
        'g_K' : 36,
        'g_Na' : 200,
        'C_m' : 1.0,
        'vr':-68.9346
        }

HH_attrs['Vr'] = HH_attrs['E_L']

HH_dic1 = { k:sorted([float(v)-0.25*float(v),float(v)+0.25*float(v)]) for k,v in HH_attrs.items() }

MODEL_PARAMS['HH'] = HH_dic1

# https://www.izhikevich.org/publications/spikes.htm
type2007 = collections.OrderedDict([
  #              C    k     vr  vt vpeak   a      b   c    d  celltype
  ('RS',        (100, 0.7,  -60, -20, 35, 0.01,   -2, -50,  100,  3)),
  ('IB',        (150, 1.2,  -75, -45, 50, 0.1,   5, -56,  130,   3)),
  ('CH',        (50,  1.5,  -60, -40, 25, 0.01,   1, -40,  150,   3)),
  ('LTS',       (100, 1.0,  -56, -42, 40, 0.01,   8, -53,   20,   4)),
  ('FS',        (20,  1,    -55, -40, 25, 0.2,    2, -45,   55,   5)),
  ('TC',        (200, 1.6,  -60, -50, 35, 0.1,  15, -60,   10,   6)),
  ('TC_burst',  (200, 1.6,  -60, -60, 35, 0.1,  15, -60,   10,   7))])

# http://www.physics.usyd.edu.au/teach_res/mp/mscripts/
# ns_izh002.m


"""
<izhikevichCell id="izTonicSpiking" v0 = "-70mV" thresh = "30mV" a ="0.02" b = "0.2" c = "-65" d = "6"/>
<izhikevichCell id="izPhasicSpiking" v0 = "-64mV" thresh = "30mV" a ="0.02" b = "0.25" c = "-65" d = "6"/>
<izhikevichCell id="izTonicBursting" v0 = "-70mV" thresh = "30mV" a ="0.02" b = "0.2" c = "-50" d = "2"/>
<izhikevichCell id="izPhasicBursting" v0 = "-64mV" thresh = "30mV" a ="0.02" b = "0.25" c = "-55" d = "0.05"/>
<izhikevichCell id="izMixedMode" v0 = "-70mV" thresh = "30mV" a ="0.02" b = "0.2" c = "-55" d = "4"/>
<izhikevichCell id="izSpikeFreqAdapt" v0 = "-70mV" thresh = "30mV" a ="0.01" b = "0.2" c = "-65" d = "8"/>
<generalizedIzhikevichCell id="izClass1Exc" v0 = "-60mV" thresh = "30mV" a ="0.02" b = "-0.1" c = "-55" d = "6" X="0.04" Y="4.1" Z="108"/>
<izhikevichCell id="izClass2Exc" v0 = "-64mV" thresh = "30mV" a ="0.2" b = "0.26" c = "-65" d = "0"/>
<izhikevichCell id="izSpikeLatency" v0 = "-70mV" thresh = "30mV" a ="0.02" b = "0.2" c = "-65" d = "6"/>
<izhikevichCell id="izSubthreshOsc" v0 = "-62mV" thresh = "30mV" a ="0.05" b = "0.26" c = "-60" d = "0"/>
"""

temp = {k:[] for k in ['C','k','vr','vt','vPeak','a','b','c','d','celltype']  }
for i,k in enumerate(temp.keys()):
    for v in type2007.values():
        temp[k].append(v[i])
explore_param = {k:(np.min(v),np.max(v)) for k,v in temp.items()}
IZHI_PARAMS = {k:sorted(v) for k,v in explore_param.items()}

IZHI_PARAMS = OrderedDict(IZHI_PARAMS)
IZHI_PARAMS['d'] = [0,150]
IZHI_PARAMS['c'] = [-65,-20]
IZHI_PARAMS['a'] = [0.01,0.2]
IZHI_PARAMS['vr'] = [-85,-50]


MODEL_PARAMS['IZHI'] = IZHI_PARAMS


#IZHI_PARAMS['c'] = [10,150]
#IZHI_PARAMS = explore_param#{k:(np.mean(v)-np.abs(np.mean(v))*0.1,np.mean(v)*0.1+np.mean(v)) for k,v in trans_dict.items()}

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




BAE1 = {}


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
BAE1 = {k:(np.mean(v)-np.abs(np.mean(v)*2.5),np.mean(v)+np.mean(v)*2.5) for k,v in BAE1.items()}
BAE1 = {k:sorted(v) for k,v in BAE1.items()}
BAE1['v_spike']=[-60.0,-20]
BAE1['v_reset'] = [1, 983.5]
BAE1['v_rest'] = [-100, -35]
BAE1['v_thresh'] = [-65, -15]


BAE1['b']=[0.01,20]
BAE1['a']=[0.01,20]
BAE1['delta_T']=[0.01,10]
BAE1['cm'] = [0.1, 8]
BAE1['delta_T'] = [0, 27]
BAE1['v_reset'] = [-100, -25]
BAE1['tau_m'] = [0.01, 62.78345]
BAE1['tau_w']= [0.01,344]
BAE1['cm'] = [1, 983.5]
#BAE1['cm'] = [0.1, 50]

for v in BAE1.values():
    assert v[1] -v[0] !=0


MODEL_PARAMS['ADEXP'] = BAE1



l5_pc_keys = ['gNaTs2_tbar_NaTs2_t.apical', 'gSKv3_1bar_SKv3_1.apical', 'gImbar_Im.apical', 'gNaTa_tbar_NaTa_t.axonal', 'gNap_Et2bar_Nap_Et2.axonal', 'gK_Pstbar_K_Pst.axonal', 'gK_Tstbar_K_Tst.axonal', 'gSK_E2bar_SK_E2.axonal', 'gSKv3_1bar_SKv3_1.axonal', 'gCa_HVAbar_Ca_HVA.axonal', 'gCa_LVAstbar_Ca_LVAst.axonal', 'gamma_CaDynamics_E2.axonal', 'decay_CaDynamics_E2.axonal', 'gNaTs2_tbar_NaTs2_t.somatic', 'gSKv3_1bar_SKv3_1.somatic', 'gSK_E2bar_SK_E2.somatic', 'gCa_HVAbar_Ca_HVA.somatic', 'gCa_LVAstbar_Ca_LVAst.somatic', 'gamma_CaDynamics_E2.somatic', 'decay_CaDynamics_E2.somatic']
l5_pc_values = [0.0009012730575340265, 0.024287352056036934, 0.0008315987398062784, 1.7100532387472567, 0.7671786030824507, 0.47339571930108143, 0.0025715065622581644, 0.024862299158354962, 0.7754822886266044, 0.0005560440082771592, 0.0020639185209852568, 0.013376906273759268, 207.56154268835758, 0.5154365543590191, 0.2565961138691978, 0.0024100296151316754, 0.0007416593834676707, 0.006240529502225737, 0.028595343511797353, 226.7501580822364]

L5PC = OrderedDict()
for k,v in zip(l5_pc_keys,l5_pc_values):
    L5PC[k] = sorted((v-0.1*v,v+0.1*v))

MODEL_PARAMS['L5PC'] = L5PC



GLIF_RANGE = {'El_reference': [-0.08569469261169435, -0.05463626766204832], \
              'C': [3.5071610042390286e-11-0.5*3.5071610042390286e-11, 10*7.630189223327981e-10], \
              'init_threshold': [0.009908733642683513, 0.06939040414685865], \
                   
              'th_inf': [0.009908733642683513, 0.04939040414685865], \
              'spike_cut_length': [20, 199],
              'init_AScurrents': [0.0, 0.0], \
              'init_voltage': [-0.09, -0.01], 
              'spike_cut_length':[4,94],
              'El': [-0.08569469261169435, -0.05463626766204832], 'asc_tau_array': [[0.01, 0.0033333333333333335],[0.3333333333333333, 0.1]], \
              'R_input': [27743752.593817078-0.5*27743752.593817078, 2*1792774179.3647704] }
              #'th_adapt': None, 'coeffs': {'a': 1, 'C': 1, 'b': 1, 'G': 1, \
              #                             'th_inf': 1.0212937371199788, 'asc_amp_array': [1.0, 1.0]}}

#'init_threshold': 0.02964956889477108,

GLIF_RANGE['th_adapt']=[0,0.1983063518904063]
GLIF_RANGE.pop('init_AScurrents',None)
GLIF_RANGE.pop('dt',None)
GLIF_RANGE.pop('asc_tau_array',None)
GLIF_RANGE.pop('El',None)
GLIF_RANGE = {k:sorted(v) for k,v in GLIF_RANGE.items()}

MODEL_PARAMS['GLIF'] = GLIF_RANGE