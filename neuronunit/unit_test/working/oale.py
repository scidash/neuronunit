#!/usr/bin/env python
# coding: utf-8

# In[1]:
"""
import sys
import traceback

class TracePrints(object):
    def __init__(self):
        self.stdout = sys.stdout
    def write(self,s):
        self.stdout.write("Writing %r\n" % s)
        traceback.print_stack(file=self.stdout)
sys.stdout = TracePrints()
"""
#get_ipython().run_line_magic('matplotlib', 'inline')
#plt.plot([0,1],[1,0])
#plt.show()


# In[2]:


import hide_imports

import collections
from IPython.display import HTML, display

# Fast spiking cannot be reproduced as it requires modifications to the standard Izhi equation,
# which are expressed in this mod file.
# https://github.com/OpenSourceBrain/IzhikevichModel/blob/master/NEURON/izhi2007b.mod

from collections import OrderedDict
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

import numpy as np
param_dict = OrderedDict([(k,[]) for k in ['C','k','vr','vt','vPeak','a','b','c','d']])
#OrderedDict
for i,k in enumerate(param_dict.keys()):
    for v in type2007.values():
        param_dict[k].append(v[i])

explore_param = {k:(np.min(v),np.max(v)) for k,v in param_dict.items()}
param_ranges = OrderedDict(explore_param)


#IB = mparams[param_dict['IB']]
RS = {}
IB = {}
TC = {}
CH = {}
RTN_burst = {}
for k,v in param_dict.items():
    RS[k] = v[0]
    IB[k] = v[1]
    CH[k] = v[2]
    TC[k] = v[5]
    RTN_burst[k] = v[-2]


    


# In[3]:


from neuronunit.optimisation import get_neab
tests = get_neab.process_all_cells()

#score = tests['Neocortex pyramidal cell layer 5-6'].tests[1].judge(model)
#score = tests['Neocortex pyramidal cell layer 5-6'].tests[0].judge(model)


# In[4]:


def get_table(tests):
    import pdb
    pdb.set_trace()
    temp = {t.name:t for t in tests}
    from neuronunit.optimisation.optimization_management import OptMan
    import pandas as pd
    obs = {}
    pred = {}
    similarity,lps,rps =  OptMan.closeness(OptMan,temp,temp)
    for k,p,o in zip(list(similarity.keys()),lps,rps):
        obs[k] = o
        pred[k] = p

    obs_preds = pd.DataFrame([obs,pred],index=['observations','predictions'])

    return obs_preds


# In[8]:


#from neuronunit.models.very_reduced_sans_lems import VeryReducedModel
#from neuronunit.optimisation import get_neab
model = None
import copy
from neuronunit.models.very_reduced_sans_lems import VeryReducedModel

model = VeryReducedModel(backend = str('HH'))
edges = hide_imports.model_parameters.MODEL_PARAMS['HH']
import numpy as np
params = {k:np.mean(v) for k,v in edges.items()}
model.set_attrs(params)

tests = get_neab.process_all_cells()

#score = tests['Neocortex pyramidal cell layer 5-6'].tests[1].judge(model)
#score = tests['Neocortex pyramidal cell layer 5-6'].tests[0].judge(model)
HH_tables = []
#for t in tests.values():
    #print(t.params)
    #print(dir(t))
    #import pdb
    #pdb.set_trace()

for t in tests.values():
    SA = t.judge(model)
    temp = copy.copy(t)
    table = get_table(temp)
    HH_tables.append(table)
    #break
table
print(SA)


# In[ ]:


#for mparams in [RS,IB,CH,TC,RTN_burst]:
model = None
model = VeryReducedModel(backend = str('RAW'))
model.set_attrs(RS)
SA = tests['Neocortex pyramidal cell layer 5-6'].judge(model)
temp = copy.copy(tests['Neocortex pyramidal cell layer 5-6'])

#display(tables)


# In[ ]:


import copy
tables = []
for mparams in [RS,IB,CH,TC,RTN_burst]:
    model = None
    model = VeryReducedModel(backend = str('RAW'))
    model.set_attrs(mparams)
    SA = tests['Neocortex pyramidal cell layer 5-6'].judge(model)
    temp = copy.copy(tests['Neocortex pyramidal cell layer 5-6'])
    try:
        table = get_table(temp)
        tables.append(table)
    except:
        tables.append(SA)


# In[ ]:


#print(tables[0])

for t in tables:
    try:
        display(t)
    except:
        print(t)
    #print(t)


# In[ ]:





# In[ ]:


for table in HH_tables:
    display(table)


# In[ ]:



#plt.clf()
import copy
#get_ipython().run_line_magic('matplotlib', 'inline')

def permutations(use_test,backend):
    use_test = hide_imports.TSD(use_test)
    use_test.use_rheobase_score = True
    edges = hide_imports.model_parameters.MODEL_PARAMS[backend]
    ga_out0 = use_test.optimize(edges,backend=backend,        protocol={'allen': False, 'elephant': True}, MU=8,NGEN=1)
    ga_out1 =  use_test.optimize(edges,backend=backend,        protocol={'allen': False, 'elephant': True},            MU=8,NGEN=8,seed_pop=ga_out0['pf'][0])

    
    dtc = ga_out1['pf'][0].dtc
    vm,plt = hide_imports.inject_and_plot_model(dtc.attrs,dtc.backend)
    plt.show()
    return dtc, ga_out1['DO'], vm


# In[ ]:


test_frame = hide_imports.get_neab.process_all_cells()
test_frame.pop('Olfactory bulb (main) mitral cell',None)
OMObjects = []
backends = ["ADEXP","BHH","RAW","HH"]
t = test_frame['Neocortex pyramidal cell layer 5-6']


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:



backends = ["RAW","HH"]#,"ADEXP","BHH"]


for t in test_frame.values():
    b = backends[0]
    (dtc,DO,vm) = permutations(copy.copy(t),b)
    display(dtc.SM)
    display(dtc.obs_preds)
    #plt.plot(vm.times,vm.magnitude)
    #plt.show()

    break
    



# In[ ]:


for t in test_frame.values():
    b = backends[1]
    (dtc,DO,vm) = permutations(copy.copy(t),b)
    display(dtc.SM)
    display(dtc.obs_preds)
    plt.plot(vm.times,vm.magnitude)
    plt.show()
    break


# In[ ]:


for t in test_frame.values():
    #for b in backends:
    b = backends[2]
    (dtc,DO,vm) = permutations(copy.copy(t),b)
    display(dtc.SM)
    display(dtc.obs_preds)
    plt.plot(vm.times,vm.magnitude)
    plt.show()


# In[ ]:


for t in test_frame.values():
    #for b in backends:
    b = backends[3]
    (dtc,DO,vm) = permutations(copy.copy(t),b)
    display(dtc.SM)
    display(dtc.obs_preds)
    plt.plot(vm.times,vm.magnitude)
    plt.show()


# In[ ]:


(dtc,DO) = permutations(test_frame['Neocortex pyramidal cell layer 5-6'],"ADEXP")
display(dtc.SM)
display(dtc.obs_preds)


# In[ ]:



backends = ["RAW","HH","BHH"]  



#for b in backends:
(dtc,DO) = permutations(test_frame['Neocortex pyramidal cell layer 5-6'],"RAW")


# In[ ]:


backends = iter(["RAW","HH","BHH"])


from IPython.display import HTML, display

for b in backends:
    (dtc,DO) = permutations(test_frame['Neocortex pyramidal cell layer 5-6'],b)
    display(dtc.SM)
    display(dtc.obs_preds)


# In[ ]:





# In[ ]:


#test_frame['Neocortex pyramidal cell layer 5-6']


# In[ ]:





# In[ ]:




