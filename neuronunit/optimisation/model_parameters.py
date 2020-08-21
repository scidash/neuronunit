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
HH_dic1 = { k:sorted(v) for k,v in HH_dic1.items() }

MODEL_PARAMS['OSB_HH_attrs'] = HH_dic1


#gnabar_hh=0.12 gkbar_hh=0.036 gl_hh=0.0003 el_hh=-54.3
MODEL_PARAMS['NEURONHH'] = {'gnabar':0.12,\
                            'gkbar':0.036,\
                            'Ra':100,\
                            'L':12.6157,\
                            'diam':12.6157,\
                            'gkbar':0.036,\
                            'el':-54.3,\
                            'gl':0.0003,\
                            'ena':50.0,\
                            'ek':-77.0,\
                            'cm':1.0,\
                            'ena':50.0,\
                            'ek':-77}
MODEL_PARAMS['NEURONHH'] = { k:(float(v)-0.95*float(v),float(v)+0.95*float(v)) for k,v in MODEL_PARAMS['NEURONHH'].items() }
MODEL_PARAMS['NEURONHH']['gkbar'] = (0.0018,0.1)
MODEL_PARAMS['NEURONHH']['diam'] = (0.630785,50.0)
MODEL_PARAMS['NEURONHH']['gnabar'] = (0.006,0.2539)
MODEL_PARAMS['NEURONHH']['L'] = (5.0,1000.0)
MODEL_PARAMS['NEURONHH']['diam'] = (5.0,1000.0)
MODEL_PARAMS['NEURONHH']['cm'] = (0.5,1.0)

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
IZHI_PARAMS = explore_param#{k:(np.mean(v)-np.abs(np.mean(v))*0.1,np.mean(v)*0.1+np.mean(v)) for k,v in trans_dict.items()}


IZHI_PARAMS = OrderedDict(IZHI_PARAMS)
MODEL_PARAMS['IZHI'] = IZHI_PARAMS


BAE1 = {}

'''
BAE1['cm']=0.281
BAE1['tau_refrac']=0.1
BAE1['v_spike']=-40.0 
BAE1['v_reset']=-70.6 
BAE1['v_rest']=-70.6 
BAE1['tau_m']=9.3667 
BAE1['i_offset']=0.0 
BAE1['a']=4.0
BAE1['b']=0.0805
BAE1['delta_T']=2.0
BAE1['tau_w']=144.0
BAE1['v_thresh']=-50.4
BAE1['e_rev_E']=0.0
BAE1['tau_syn_E']=5.0
BAE1['e_rev_I']=-80.0 
BAE1['tau_syn_I']=5.0
#BAE1['N']=1
#BAE1['tau_psc']=5.0
#BAE1['connectivity']=None
BAE1['spike_delta']=30
BAE1['scale']=0.5
'''

'''
C = 281; % capacitance in pF ... this is 281*10^(-12) F
g_L = 30; % leak conductance in nS
E_L = -70.6; % leak reversal potential in mV ... this is -0.0706 V
delta_T = 2; % slope factor in mV
V_T = -50.4; % spike threshold in mV
tau_w = 144; % adaptation time constant in ms
V_peak = 20; % when to call action potential in mV
b = 0.0805; % spike-triggered adaptation
'''
'''
# Physiologic neuron parameters from Gerstner et al.
BAE1['C']       = 238.96205752321424  # capacitance in pF ... this is 281*10^(-12) F
BAE1['g_L']     = 52.33847145920891     # leak conductance in nS
BAE1['E_L']     = -128.24969985688054  # leak reversal potential in mV ... this is -0.0706 V
BAE1['delta_T'] =  3.3496462271892624    # slope factor in mV
BAE1['V_T']     = -50.4  # spike threshold in mV
BAE1['tau_w']   = 144    # adaptation time constant in ms
BAE1['V_peak']  = 30.0     # when to call action potential in mV
BAE1['b']       = 0.16705627559842062,# spike-triggered adaptation
BAE1['a']       = 7.198877237743017
BAE1['v_rest'] = -70
'''

BAE1 = {}
BAE1['cm']=281
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
BAE1 = {k:(np.mean(v)-np.mean(v)*0.5,np.mean(v)*0.5+np.mean(v)) for k,v in BAE1.items()}
BAE1 = {k:sorted(v) for k,v in BAE1.items()}

MODEL_PARAMS['ADEXP'] = BAE1



l5_pc_keys = ['gNaTs2_tbar_NaTs2_t.apical', 'gSKv3_1bar_SKv3_1.apical', 'gImbar_Im.apical', 'gNaTa_tbar_NaTa_t.axonal', 'gNap_Et2bar_Nap_Et2.axonal', 'gK_Pstbar_K_Pst.axonal', 'gK_Tstbar_K_Tst.axonal', 'gSK_E2bar_SK_E2.axonal', 'gSKv3_1bar_SKv3_1.axonal', 'gCa_HVAbar_Ca_HVA.axonal', 'gCa_LVAstbar_Ca_LVAst.axonal', 'gamma_CaDynamics_E2.axonal', 'decay_CaDynamics_E2.axonal', 'gNaTs2_tbar_NaTs2_t.somatic', 'gSKv3_1bar_SKv3_1.somatic', 'gSK_E2bar_SK_E2.somatic', 'gCa_HVAbar_Ca_HVA.somatic', 'gCa_LVAstbar_Ca_LVAst.somatic', 'gamma_CaDynamics_E2.somatic', 'decay_CaDynamics_E2.somatic']
l5_pc_values = [0.0009012730575340265, 0.024287352056036934, 0.0008315987398062784, 1.7100532387472567, 0.7671786030824507, 0.47339571930108143, 0.0025715065622581644, 0.024862299158354962, 0.7754822886266044, 0.0005560440082771592, 0.0020639185209852568, 0.013376906273759268, 207.56154268835758, 0.5154365543590191, 0.2565961138691978, 0.0024100296151316754, 0.0007416593834676707, 0.006240529502225737, 0.028595343511797353, 226.7501580822364]

L5PC = OrderedDict()
for k,v in zip(l5_pc_keys,l5_pc_values):
    L5PC[k] = sorted((v-0.1*v,v+0.1*v))

MODEL_PARAMS['L5PC'] = L5PC