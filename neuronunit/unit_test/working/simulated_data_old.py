#!/usr/bin/env python
# coding: utf-8

# # Set up the environment

# In[1]:


import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
#get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot([0,1],[1,0])
plt.show()
import hide_imports
from neuronunit.optimisation.optimization_management import inject_and_plot_model


# # Design simulated data tests

# In[2]:


def jrt(use_test,backend):
    use_test = hide_imports.TSD(use_test)
    use_test.use_rheobase_score = True
    edges = hide_imports.model_parameters.MODEL_PARAMS[backend]

    OM = hide_imports.OptMan(use_test,                backend=backend,                boundary_dict=edges,                protocol={'allen': False, 'elephant': True})

    return OM


# In[3]:


test_frame = hide_imports.get_neab.process_all_cells()
test_frame.pop('Olfactory bulb (main) mitral cell',None)
stds = {}
for k,v in hide_imports.TSD(test_frame['Neocortex pyramidal cell layer 5-6']).items():
    temp = hide_imports.TSD(test_frame['Neocortex pyramidal cell layer 5-6'])[k]
    stds[k] = temp.observation['std']
    print((temp.name,temp.observation))
OMObjects = []
backends = ["RAW"]
import copy
cloned_tests = copy.copy(test_frame['Neocortex pyramidal cell layer 5-6'])
import copy
for b in backends:        
    OM = jrt(cloned_tests,b)
    OMObjects.append(OM)
rt_outs = []


# In[4]:


for OM in OMObjects:
   print(OM.backend)
   x= {k:v for k,v in OM.tests.items() if 'mean' in v.observation.keys() or 'value' in v.observation.keys()}
   cloned_tests = copy.copy(OM.tests)
   OM.tests = hide_imports.TSD(cloned_tests)
   rt_out = OM.simulate_data(OM.tests,OM.backend,OM.boundary_dict)
   #print(rt_out)


# In[5]:


penultimate_tests = hide_imports.TSD(test_frame['Neocortex pyramidal cell layer 5-6'])
for k,v in penultimate_tests.items():
    temp = penultimate_tests[k]

    v = rt_out[1][k].observation
    v['std'] = stds[k]
simulated_data_tests = hide_imports.TSD(penultimate_tests)


# # Show what the randomly generated target waveform the optimizer needs to find actually looks like

# In[6]:


target = rt_out[0]
target.rheobase
inject_and_plot_model(target)


# # Commence optimization of models on simulated data sets

# # first lets just optimize over single objective functions at a time.

# In[7]:


ga_out_rh = hide_imports.TSD([simulated_data_tests["RheobaseTest"]]).optimize(OMObjects[0].boundary_dict,backend=OMObjects[0].backend,        protocol={'allen': False, 'elephant': True},            MU=10,NGEN=10)
opt_rh = ga_out_rh['pf'][0].dtc
opt_rh.obs_preds


# In[8]:


ga_out_time = hide_imports.TSD([simulated_data_tests["TimeConstantTest"]]).optimize(OMObjects[0].boundary_dict,backend=OMObjects[0].backend,        protocol={'allen': False, 'elephant': True},            MU=10,NGEN=10)
opt_time = ga_out_time['pf'][0].dtc
opt_time.obs_preds


# In[9]:


both = hide_imports.TSD([simulated_data_tests["TimeConstantTest"],simulated_data_tests["RheobaseTest"]]).optimize(OMObjects[0].boundary_dict,backend=OMObjects[0].backend,        protocol={'allen': False, 'elephant': True},            MU=20,NGEN=5)
both = both['pf'][0].dtc
both.obs_preds


# In[10]:


three = hide_imports.TSD([simulated_data_tests["InjectedCurrentAPWidthTest"],simulated_data_tests["TimeConstantTest"],simulated_data_tests["RheobaseTest"]]).optimize(OMObjects[0].boundary_dict,backend=OMObjects[0].backend,        protocol={'allen': False, 'elephant': True},            MU=25,NGEN=6)


# In[ ]:


three = three['pf'][0].dtc
three.obs_preds

ga_out = simulated_data_tests.optimize(OMObjects[0].boundary_dict,backend=OMObjects[0].backend, protocol={'allen': False, 'elephant': True}, MU=100,NGEN=100)
opt = ga_out['pf'][0].dtc
front = [ind.dtc for ind in ga_out['pf']]


opt.rheobase



inject_and_plot_model(opt)



opt.obs_preds

from neuronunit.optimisation.optimization_management import check_match_front
check_match_front(target,front[0:10])



from neuronunit.optimisation.algorithms import cleanse
seed_pop = cleanse(copy.copy(ga_out['pf']))
