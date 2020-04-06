#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set_context('notebook')
#get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
#import matplotlib.pyplot as plt
#plt.plot([0,1],[1,0])
#plt.show()
import pickle


tests = pickle.load(open('processed_multicellular_constraints.p','rb'))
testsr = tests['Neocortex pyramidal cell layer 5-6'].tests
from neuronunit.optimisation.data_transport_container import DataTC



dtc = DataTC(backend=str("RAW"))
backend=str("RAW")
dtc.backend = backend


dtc.tests = testsr

#model = dtc.dtc_to_model()
#testsr[-3].judge(model)





print(dtc.attrs)
dtc.tests = testsr
from neuronunit.optimisation.model_parameters import MODEL_PARAMS
dtc.attrs = dtc.dtc_to_model().default_attrs

dtc.attrs['C'] = testsr[3].observation['mean']
dtc.attrs['vr'] = testsr[4].observation['mean']
ranges = MODEL_PARAMS[backend]
#print(ranges)
#ranges.pop('C',None)
#ranges.pop('vr',None)
#ranges.pop('vPeak',None)
#ranges.pop('vt',None)
print(dtc.tests)


# In[11]:


#%%capture
import numpy as np
print(dtc.tests)
OM = dtc.dtc_to_opt_man()
OM.boundary_dict = ranges
OM.backend = backend
OM.td = list(ranges.keys())
out = OM.random_search(dtc,500)
print(out)
#pop,dtcpop = OM.boot_new_genes(3,dtc)
'''
scores = []
for d in dtcpop:
    d.attrs['C'] = testsr[3].observation['mean']
    d.attrs['vr'] = testsr[4].observation['mean']
    vt = testsr[-1].observation['mean']
    vPeak = testsr[-2].observation['mean']
    d.attrs['vt'] = vt
    d.attrs['vPeak'] = float(vPeak)
    #print(vPeak)


# In[ ]:





# In[ ]:





# In[ ]:


#from collections import OrderedDict
import numpy as np
print(dtc.tests)
OM = dtc.dtc_to_opt_man()
OM.boundary_dict = ranges
OM.backend = backend
OM.td = list(ranges.keys())
pop,dtcpop = OM.boot_new_genes(3,dtc)
scores = []
for d in dtcpop:
    d.attrs['C'] = testsr[3].observation['mean']
    d.attrs['vr'] = testsr[4].observation['mean']
    vt = testsr[-1].observation['mean']
    #vPeak = testsr[-2].observation['mean']
    #print(vt,vPeak)
    d.attrs['vt'] = vt
    #d.attrs['vPeak'] = vPeak

    #print(sum_error)


# In[ ]:





# In[ ]:


len(dtcpop)



#from collections import OrderedDict
import numpy as np
print(dtc.tests)
OM = dtc.dtc_to_opt_man()
OM.boundary_dict = ranges
OM.backend = backend
OM.td = list(ranges.keys())

pop,dtcpop = OM.boot_new_genes(10,dtc)
scores = []
for d in dtcpop:
    d.attrs['C'] = testsr[3].observation['mean']
    d.attrs['vr'] = testsr[4].observation['mean']
    vt = testsr[-1].observation['mean']
    #vPeak = testsr[-2].observation['mean']
    #print(vt,vPeak)
    d.attrs['vt'] = vt
    #d.attrs['vPeak'] = vPeak


for d in dtcpop:
    d.tests = testsr
    d = d.self_evaluate()
    for s in d.SA:
        print(s.score)
    #print(d.SA)
    #import pdb
    #pdb.set_trace()
    #sum_error = np.sum([i.score for i in d.SA if type(i) is not type(None)])
    #sum_error = np.sum(d.SA.values)
    #print(sum_error)


# In[ ]:


#from collections import OrderedDict
import numpy as np
print(dtc.tests)
OM = dtc.dtc_to_opt_man()
OM.boundary_dict = ranges
OM.backend = backend
OM.td = list(ranges.keys())

pop,dtcpop = OM.boot_new_genes(10,dtc)
scores = []
for d in dtcpop:
    d.attrs['C'] = testsr[3].observation['mean']
    d.attrs['vr'] = testsr[4].observation['mean']
    vt = testsr[-1].observation['mean']
    #vPeak = testsr[-2].observation['mean']
    #print(vt,vPeak)
    d.attrs['vt'] = vt
    #d.attrs['vPeak'] = vPeak


for d in dtcpop:
    d.tests = testsr
    d = d.self_evaluate()
    for s in d.SA:
        print(s.score)
    #print(d.SA)
    #import pdb
    #pdb.set_trace()
    #sum_error = np.sum([i.score for i in d.SA if type(i) is not type(None)])
    #sum_error = np.sum(d.SA.values)
    #print(sum_error)

#print(dir())stats
# In[ ]:



#dtc.tests = {t.name:t for t in dtc.tests}
dtc = OM.get_agreement(dtc)
hide_imports.display(dtc.SA)
hide_imports.display(dtc.obs_preds)
dtc.obs_pred


# In[ ]:


hide_imports.inject_and_plot_model(dtc)
plt.plot(vm.times,vm.magnitude)

plt.show()


# In[ ]:


dtc = OM.get_agreement(dtc)

dtc.obs_pred


# In[ ]:


dtc = dtc.self_evaluate()
dtc.SA
dtc = OM.get_agreement(dtc)


# In[ ]:





# In[ ]:





# In[ ]:


#test_frame['Neocortex pyramidal cell layer 5-6']


# In[ ]:





# In[ ]:




'''
