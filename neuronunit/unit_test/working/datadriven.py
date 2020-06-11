#!/usr/bin/env python
# coding: utf-8

# In[1]:


import seaborn as sns
from neuronunit.optimisation.optimization_management import test_all_objective_test
import numpy as np
from IPython.display import display, HTML
from neuronunit.plottools import plot_score_history1, plot_score_history_standardized
from neuronunit.optimisation.optimization_management import check_binary_match, TSD, feature_mine
import pandas as pd
from collections import OrderedDict
import pickle
from deap import creator
from deap import base
import array
#creator.create("FitnessMin", base.Fitness, weights=tuple(-1.0 for i in range(0,10)))
#creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)

import pickle
cells = pickle.load(open("processed_multicellular_constraints.p","rb"))

purk = TSD(cells['Cerebellum Purkinje cell'])#.tests
purk_vr = purk["RestingPotentialTest"].observation['mean']

ncl5 = TSD(cells["Neocortex pyramidal cell layer 5-6"])
ncl5.name = str("Neocortex pyramidal cell layer 5-6")
ncl5_vr = ncl5["RestingPotentialTest"].observation['mean']

ca1 = TSD(cells['Hippocampus CA1 pyramidal cell'])
ca1_vr = ca1["RestingPotentialTest"].observation['mean']
#cap = ca1["CapacitanceTest"].observation['mean']/1000.0
#rin = ca1["InputResistanceTest"].observation['mean']

#from SALib.test_functions import Ishigami
import numpy as np
from neuronunit.optimisation.model_parameters import MODEL_PARAMS
print(MODEL_PARAMS.keys())

fps_izhi = list(MODEL_PARAMS['IZHI'].keys())
import pickle
rt = pickle.load(open("contains_mitral.p","rb"))
olf = TSD(rt[1]['olf_mit'])
olf_vr = olf["RestingPotentialTest"].observation['mean']
olf.use_rheobase_score = False

# # Simulated Data 
# and select model parameters that are free to vary

# In[4]:
from neuronunit.optimisation.optimization_management import inject_and_plot_model, inject_and_plot_passive_model
from neuronunit.optimisation.data_transport_container import DataTC, DataTCModel
from neuronunit.optimisation.model_parameters import MODEL_PARAMS

import numpy as np

import shelve
model_type = "RAW"
fps_izhi = list(MODEL_PARAMS['IZHI'].keys())
#model_type="NEURONHH"
#fps_hh = list(MODEL_PARAMS['NEURONHH'].keys())
'''
results = olf.optimize(backend=model_type,protocol={'allen': False, 'elephant': True}
    ,MU=200,NGEN=1,
    free_parameters=fps_izhi,
    ignore_cached=False,hc={'vr':olf_vr})#,seed_pop=saltelli_values)


results = olf.optimize(backend=model_type,protocol={'allen': False, 'elephant': True}
    ,MU=10,NGEN=200,
    free_parameters=fps_izhi,
    ignore_cached=False,hc={'vr':olf_vr},seed_pop=results['pf'][0:9])

opt = results['pf'][0].dtc
opt = results['pf'][0].dtc
with open('../../data/dump_izhi_olf.p','wb') as f:
    pickle.dump(opt,f)

'''
#opt.plt = None
#opt.plt = plt
#import pdb
#pdb.set_trace()

model_type = 'NEURONHH'
fps_hh = list(MODEL_PARAMS['NEURONHH'].keys())

'''
rand = olf.optimize(backend=model_type,\
    protocol={'allen': False, 'elephant': True}
    ,MU=80,NGEN=1,
    free_parameters=fps_hh,
    ignore_cached=False)#,hc={'vr':olf_vr})#,seed_pop=saltelli_values)
'''    

with open('simulated_tests.p','rb') as f:
    sim_tests = pickle.load(f)
MU = 10
NGEN = 100

for k,v in olf.items(): 
    olf[k].observation['std'] = sim_tests[k].observation['std']
results = olf.optimize(backend=model_type,\
    protocol={'allen': False, 'elephant': True}
    ,MU=MU,NGEN=NGEN,
    free_parameters=fps_hh,
    ignore_cached=False)#,seed_pop=rand['pf'][0:MU])

opt = [ results['min_gene'], results['pf'] ]
with open('../../data/dump_neuron_olf.p','wb') as f:
    pickle.dump(opt,f)



model_type = 'ADEXP'
fps_hh = list(MODEL_PARAMS['ADEXP'].keys())


rand = olf.optimize(backend=model_type,\
    protocol={'allen': False, 'elephant': True}
    ,MU=80,NGEN=1,
    free_parameters=fps_hh,
    ignore_cached=False)#,hc={'vr':olf_vr})#,seed_pop=saltelli_values)
MU = 20
NGEN = 20
results = olf.optimize(backend=model_type,\
    protocol={'allen': False, 'elephant': True}
    ,MU=MU,NGEN=NGEN,
    free_parameters=fps_hh,
    ignore_cached=False,seed_pop=rand['pf'][0:MU])

model_type = 'NEURONHH'
fps_hh = list(MODEL_PARAMS['NEURONHH'].keys())


rand = olf.optimize(backend=model_type,\
    protocol={'allen': False, 'elephant': True}
    ,MU=80,NGEN=1,
    free_parameters=fps_hh,
    ignore_cached=False)#,hc={'vr':olf_vr})#,seed_pop=saltelli_values)
MU = 20
NGEN = 20
results = olf.optimize(backend=model_type,\
    protocol={'allen': False, 'elephant': True}
    ,MU=MU,NGEN=NGEN,
    free_parameters=fps_hh,
    ignore_cached=False,seed_pop=rand['pf'][0:MU])

opt = results['pf'][0].dtc #  [ results['min_gene'], results['pf'] ]
with open('../../data/dump_neuron_olf.p','wb') as f:
    pickle.dump(opt,f)

with open('../../data/dump_neuron_olf_results.p','wb') as f:
    pickle.dump(results,f)

'''
rand_sample = olf.optimize(backend=model_type,protocol={'allen': False, 'elephant': True}
    ,MU=100,NGEN=1,plot=True,
    free_parameters=fps_hh,
    ignore_cached=False,hc={'vr':olf_vr})
'''
MU = 10
NGEN =100

results = olf.optimize(backend=model_type,protocol={'allen': False, 'elephant': True}
    ,MU=MU,NGEN=NGEN,plot=True,
    free_parameters=fps_hh,
    ignore_cached=False)#,seed_pop=rand_sample['pf'][0:9])

opt = [ results['min_gene'], results['pf'] ]
with open('../../data/dump_hh_olf.p','wb') as f:
    pickle.dump(opt,f)

import numpy as np
MU = 10 #2**len(fps)
NGEN = 100
import shelve
model_type = "RAW"
fps_izhi = list(MODEL_PARAMS['RAW'].keys())

results = ncl5.optimize(backend=model_type,protocol={'allen': False, 'elephant': True}
    ,MU=MU,NGEN=NGEN,plot=True,
    free_parameters=fps_izhi,
    ignore_cached=False,hc={'vr':ncl5_vr})#,seed_pop=[gene,gene,gene,gene,gene])
plot_score_history1(results,figname=str('ncl5')+model_type+str('.png'))
opt = [ results['min_gene'], results['pf'] ]


import numpy as np
MU = 10 #2**len(fps)
NGEN = 100
import shelve
model_type = "RAW"
fps_izhi = list(MODEL_PARAMS['RAW'].keys())

results = ncl5.optimize(backend=model_type,protocol={'allen': False, 'elephant': True}
    ,MU=MU,NGEN=NGEN,plot=True,
    free_parameters=fps_izhi,
    ignore_cached=False,hc={'vr':ncl5_vr})#,seed_pop=[gene,gene,gene,gene,gene])
plot_score_history1(results,figname=str('ncl5')+model_type+str('.png'))
opt = [ results['min_gene'], results['pf'] ]
#with open('dump_izhi_ncl5.p','wb') as f:
#    pickle.dump(opt,f)

import numpy as np
MU = 10 #2**len(fps)
NGEN = 100
import shelve
model_type = "RAW"
fps_izhi = list(MODEL_PARAMS['RAW'].keys())

# # Meta Parameters
# IZHI
results = ncl5.optimize(backend=model_type,protocol={'allen': False, 'elephant': True}
    ,MU=MU,NGEN=NGEN,plot=True,
    free_parameters=fps_izhi,
    ignore_cached=False,hc={'vr':ncl5_vr})#,seed_pop=[gene,gene,gene,gene,gene])
#plot_score_history_standardized(results,figname=str('ncl5')+model_type+str('.png'))
opt = [ results['min_gene'], results['pf'] ]
with open('../../data/dump_izhi_ncl5.p','wb') as f:
    pickle.dump(opt,f)


results = ca1.optimize(backend=model_type, \
    protocol={'allen': False, 'elephant': True},  \
    MU=MU,NGEN=NGEN,plot=True,free_parameters=fps_izhi, 
    ignore_cached=True,hc={'vr':ca1_vr})
opt = results['min_gene']
with open('../../data/dump_izhi_ca1.p','wb') as f:
    pickle.dump(opt,f)

results = purk.optimize(backend=model_type, \
    protocol={'allen': False, 'elephant': True},  \
    MU=MU,NGEN=NGEN,plot=True,free_parameters=fps_izhi, 
    ignore_cached=True)
opt = [ results['min_gene'], results['pf'] ]
with open('../../data/dump_izhi_purk.p','wb') as f:
    pickle.dump(opt,f)
# HH
MU = 90
NGEN = 20
model_type="NEURONHH"
fps_hh = list(MODEL_PARAMS['NEURONHH'].keys())

results = ncl5.optimize(backend=model_type,protocol={'allen': False, 'elephant': True}
    ,MU=MU,NGEN=NGEN,plot=True,
    free_parameters=fps_hh,
    ignore_cached=False,hc={'vr':ncl5_vr})#,seed_pop=[gene,gene,gene,gene,gene])
opt = [ results['min_gene'], results['pf'] ]
with open('../../data/dump_hh_ncl5.p','wb') as f:
    pickle.dump(opt,f)


results = ca1.optimize(backend=model_type, \
    protocol={'allen': False, 'elephant': True},  \
    MU=MU,NGEN=NGEN,plot=True,free_parameters=fps_hh, 
    ignore_cached=True,hc={'vr':ca1_vr})
opt = [ results['min_gene'], results['pf'] ]
with open('../../data/dump_hh_ca1.p','wb') as f:
    pickle.dump(opt,f)

results = purk.optimize(backend=model_type, \
    protocol={'allen': False, 'elephant': True},  \
    MU=MU,NGEN=NGEN,plot=True,free_parameters=fps_hh, 
    ignore_cached=True,hc={'vr':purk_vr})
opt = [ results['min_gene'], results['pf'] ]
with open('../../data/dump_hh_purk.p','wb') as f:
    pickle.dump(opt,f)



#single_objective(results,figname='with_seed_larger_pop.png')
#dtc1 = results['hof'][0].dtc
#x1,vm1 = inject_and_plot_model(dtc1,plotly=False,figname="opt_NEURONHH")

#inject_and_plot_passive_model(dtc1,second=None,figname="passive_opt_NEURONHH",plotly=False)
#with open('opt_hh.p','wb') as f:
    #import pdb
    #pdb.set_trace()

#pickle.dump(f,opt)

#results = ca1.optimize(backend=model_type, protocol={'allen': False, 'elephant': True}, MU=30,NGEN=5,plot=True,free_parameters=fps, hold_constant={'vr':vr},ignore_cached=False,seed_pop = results['pf'])
#single_objective(results,figname='with_seed_very_larger_pop.png')



#import pdb
#pdb.set_trace()
#results = plot_score_history1(results)
#from neuronunit.plottools import plot_score_history_SA
#results = plot_score_history_SA(results)

#opt = feature_mine(results['pf'][0].dtc)
# In[ ]:
'''

opts = []
for v in results.values():
    if v is not None:
        import pdb
        pdb.set_trace()
        opt = v['pf'][0].dtc
        opts.append(opt)
        agreement = opt.get_agreement().agreement
        display(agreement)
'''
opt = results['min_gene'][0]

opt = feature_mine(opt)


# # Analyse Results

# In[ ]:


with open('neo.p','wb') as f:
    pickle.dump(opt,f)
hof = results['hof'][0].dtc
agreement = opt.get_agreement().agreement

display(agreement)


# # Compare scores above to random scores below

# In[ ]:


# This should be more succint than above.
#display(out['frame'])
MU = 10 
NGEN = 50
model_type = "HH"
from neuronunit.optimisation import model_parameters
fps = model_parameters.MODEL_PARAMS["HH"].keys()

print(fps)
results = ncl5.optimize(backend=model_type,        protocol={'allen': False, 'elephant': True},            MU=MU,NGEN=NGEN,plot=True,free_parameters=fps,hold_constant={'Vr':vr})


# In[ ]:


out['best_random_sum_total']


# # Look at evolution History

# In[ ]:


plt = plot_score_history1(results)
from neuronunit.optimisation.optimization_management import check_binary_match


# In[ ]:


model = opt.dtc_to_model()
#check_binary_match(opt,target,snippets=True)
#target = OM.format_test(target)
#simulated_data_tests = target.tests


# In[ ]:


#check_binary_match(opt,target,snippets=False)


# In[ ]:


try:
    opt.attrs.pop('dt',None)
    opt.attrs.pop('Iext',None)
except:
    pass


# In[ ]:


display(pd.DataFrame([opt.attrs]))
import copy
temp = {}
for k in opt.attrs.keys():
    temp[k] = target.attrs[k]
display(pd.DataFrame([temp]))


# In[ ]:


display(pd.DataFrame([{k.name:v for k,v in opt.SA.items()}]))


# What where the values of model parameters that where held constant?
# 

# In[ ]:


df0 = opt.dtc_to_model().default_attrs
df1 = target.dtc_to_model().default_attrs
hc = {}

try:
    df0.attrs.pop('dt',None)
    df0.attrs.pop('Iext',None)
    opt.attrs.pop('dt',None)
    opt.attrs.pop('Iext',None)

except:
    pass

for k,v in df0.items():
    if k not in opt.attrs.keys():
        assert df0[k] == df1[k]
        hc[k] = v        
display("Held constant:")
display(pd.DataFrame([hc]))        


# If the Pareto Front encircles the best solution without sampling directly on top of it.
# Does piercing the center get us closer to the hall of fame?
# Below, plot HOF[0]/PF[0] are they the same model? 

# In[ ]:


# check_binary_match(opt,hof,snippets=True)


# # Exploring the neighbourhood of 
# the Optimal solution is now syntatically easy
# 
# ## Make ranges to explore:

# In[ ]:


from neuronunit.optimisation.model_parameters import MODEL_PARAMS
a_range = MODEL_PARAMS["RAW"]['a']
grid_a = np.linspace(a_range[0],a_range[1],25)

import copy


# ## Mutate a parameter in a dimension of interest.

# In[ ]:


opt_sum0 = np.sum(opt.SA.values)
for_scatter0 = (opt.attrs['a'],opt_sum0)
from tqdm import tqdm
collect = []
mutant = copy.copy(opt)
for a in tqdm(grid_a):
    # non random mutation
    mutant.attrs['a'] = a
    # Evaluate NU test suite
    mutant.self_evaluate()
    # sum components (optional)
    fit = np.sum(mutant.SA.values)
    collect.append(fit)
plt.plot(grid_a,collect)
plt.scatter(for_scatter0[0],for_scatter0[1],label='optima')
    


# In[ ]:





# The above plot seemed to have multiple steep wells of low error about the optima.
# 
# It might not be reasonable to expect to sample every such well, as the stochastic and non exhuastive sampling in the GA means it is not garunteed to sample small and and focused pockets of error change.

# In[ ]:





# In[ ]:


b_range = MODEL_PARAMS["RAW"]['b']
grid_b = np.linspace(b_range[0],b_range[1],30);


# In[ ]:





# In[ ]:


opt_sum = np.sum(opt.SA.values)
for_scatter1 = (opt.attrs['b'],opt_sum)
from tqdm import tqdm
collect = []
mutant = copy.copy(opt)
for b in tqdm(grid_b):
    # non random mutation
    mutant.attrs['b'] = b
    # Evaluate NU test suite
    mutant.self_evaluate()
    # sum components (optional)
    fit = np.sum(mutant.SA.values)
    collect.append(fit)
plt.plot(grid_b,collect)
plt.scatter(for_scatter1[0],for_scatter1[1],label='optima')


# # Compare the match of passive waveforms

# In[ ]:



#from neuronunit.optimisation.optimization_management import inject_and_plot_passive_model
#inject_and_plot_passive_model(opt)
import quantities as pq
tm = target.dtc_to_model()

model = opt.dtc_to_model()
uc = {'amplitude':-10*pq.pA,'duration':500*pq.ms,'delay':100*pq.ms}
model.inject_square_current(uc)
vm1 = model.get_membrane_potential()
tm.inject_square_current(uc)
vm0 = tm.get_membrane_potential()
plt.plot(vm1.times, vm1.magnitude, c='b',label=str('opt ADEXP'))#+str(model.attrs['a']))
plt.plot(vm0.times, vm0.magnitude, c='r',label=str('target HH'))#+str(model.attrs['a']))

plt.show()


# In[ ]:




