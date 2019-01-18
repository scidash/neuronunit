
# coding: utf-8

# Assumptions, the environment for running this notebook was arrived at by building a dedicated docker file.
#
# https://cloud.docker.com/repository/registry-1.docker.io/russelljarvis/nuo
#
# You can run use dockerhub to get the appropriate file, and launch this notebook using Kitematic.


# This is code, change cell type to markdown.
# ![alt text](plan.jpg "Pub plan")


# # Import libraries
# To keep the standard running version of minimal and memory efficient, not all available packages are loaded by default. In the cell below I import a mixture common python modules, and custom developed modules associated with NeuronUnit (NU) development


#!pip install dask distributed seaborn
#!bash after_install.sh
import numpy as np
import os
import pickle
import pandas as pd
from neuronunit.tests.fi import RheobaseTestP
from neuronunit.optimization.model_parameters import reduced_dict, reduced_cells
from neuronunit.optimization import optimization_management as om
from sciunit import scores# score_type

from neuronunit.optimization.data_transport_container import DataTC
from neuronunit.tests.fi import RheobaseTestP# as discovery
from neuronunit.optimization.optimization_management import dtc_to_rheo, format_test, nunit_evaluation
import quantities as pq
from neuronunit.models.reduced import ReducedModel
from neuronunit.optimization.model_parameters import model_params, path_params
LEMS_MODEL_PATH = path_params['model_path']
list_to_frame = []
from neuronunit.tests.fi import RheobaseTestP


# from IPython.display import HTML, display
# import seaborn as sns





# # The Izhiketich model is instanced using some well researched parameter sets.
#

# First lets get the points in parameter space, that Izhikich himself has published about. These points are often used by the open source brain project to establish between model reproducibility. The itial motivating factor for choosing these points as constellations, of all possible parameter space subsets, is that these points where initially tuned and used as best guesses for matching real observed experimental recordings.

# In[ ]:

explore_param = {k:(np.min(v),np.max(v)) for k,v in reduced_dict.items()}


# ## Get the experimental Data pertaining to four different classes or neurons, that can constrain models.
# Next we get some electro physiology data for four different classes of cells that are very common targets of scientific neuronal modelling. We are interested in finding out what are the most minimal, and detail reduced, low complexity model equations, that are able to satisfy

# Below are some of the data set ID's I used to query neuroelectro.
# To save time for the reader, I prepared some data earlier to save time, and saved the data as a pickle, pythons preferred serialisation format.
#
# The interested reader can find some methods for getting cell specific ephys data from neuroelectro in a code file (neuronunit/optimization/get_neab.py)
#

# In[ ]:

purkinje ={"id": 18, "name": "Cerebellum Purkinje cell", "neuron_db_id": 271, "nlex_id": "sao471801888"}
fi_basket = {"id": 65, "name": "Dentate gyrus basket cell", "neuron_db_id": None, "nlex_id": "nlx_cell_100201"}
pvis_cortex = {"id": 111, "name": "Neocortex pyramidal cell layer 5-6", "neuron_db_id": 265, "nlex_id": "nifext_50"}
#does not have rheobase
olf_mitral = {"id": 129, "name": "Olfactory bulb (main) mitral cell", "neuron_db_id": 267, "nlex_id": "nlx_anat_100201"}
ca1_pyr = {"id": 85, "name": "Hippocampus CA1 pyramidal cell", "neuron_db_id": 258, "nlex_id": "sao830368389"}
pipe = [ fi_basket, ca1_pyr, purkinje,  pvis_cortex]


# In[ ]:

electro_tests = []
obs_frame = {}
test_frame = {}

try:

    electro_path = str(os.getcwd())+'all_tests.p'

    assert os.path.isfile(electro_path) == True
    with open(electro_path,'rb') as f:
        (obs_frame,test_frame) = pickle.load(f)

except:
    for p in pipe:
        p_tests, p_observations = get_neab.get_neuron_criteria(p)
        obs_frame[p["name"]] = p_observations#, p_tests))
        test_frame[p["name"]] = p_tests#, p_tests))
    electro_path = str(os.getcwd())+'all_tests.p'
    with open(electro_path,'wb') as f:
        pickle.dump((obs_frame,test_frame),f)


# # Cast the tabulatable data to pandas data frame
# There are many among us who prefer potentially tabulatable data to be encoded in pandas data frame.

for k,v in test_frame.items():
    if "olf_mit" not in k:
        obs = obs_frame[k]
        v[0] = RheobaseTestP(obs['Rheobase'])
df = pd.DataFrame.from_dict(obs_frame)
print(test_frame.keys())


# In the data frame below, you can see many different cell types

df['Hippocampus CA1 pyramidal cell']



# # Tweak Izhikitich equations
# with educated guesses based on information that is already encoded in the predefined experimental observations.
#
# In otherwords use information that is readily amenable into hardcoding into equations
#
# Select out the 'Neocortex pyramidal cell layer 5-6' below, as a target for optimization


free_params = ['a','b','k','c','C','d','vPeak','vr']
hc_ = reduced_cells['RS']
hc_['vr'] = -65.2261863636364
hc_['vPeak'] = hc_['vr'] + 86.364525297619
explore_param['C'] = (hc_['C']-20,hc_['C']+20)
explore_param['vr'] = (hc_['vr']-5,hc_['vr']+5)
use_test = test_frame["Neocortex pyramidal cell layer 5-6"]

#for t in use_test[::-1]:
#    t.score_type = scores.RatioScore
test_opt = {}

with open('data_dump.p','wb') as f:
    pickle.dump(test_opt,f)


# In[ ]:


use_test[0].observation
print(use_test[0].name)

rtp = RheobaseTestP(use_test[0].observation)
use_test[0] = rtp
print(use_test[0].observation)


reduced_cells.keys()
test_frame.keys()
test_frame.keys()
test_frame['olf_mit'].insert(0,test_frame['Cerebellum Purkinje cell'][0])
test_frame



#!pip install neo --upgrade

df = pd.DataFrame(index=list(test_frame.keys()),columns=list(reduced_cells.keys()))

for k,v in reduced_cells.items():
    temp = {}
    temp[str(v)] = {}
    dtc = DataTC()
    dtc.tests = use_test
    dtc.attrs = v
    dtc.backend = 'RAW'
    dtc.cell_name = 'vanilla'


    for key, use_test in test_frame.items():
        dtc.tests = use_test
        dtc = dtc_to_rheo(dtc)
        dtc = format_test(dtc)

        if dtc.rheobase is not None:
            if dtc.rheobase!=-1.0:

                dtc = nunit_evaluation(dtc)

        df[k][key] = int(dtc.get_ss())

# A sparse grid sampling over the parameter space, using the published and well corrobarated parameter points, from Izhikitch publications, and the Open Source brain, shows that without optimization, using off the shelf parameter sets to fit real-life biological cell data, does not work so well.


MU = 6
NGEN = 150

for key, use_test in test_frame.items():
    ga_out, _ = om.run_ga(explore_param,NGEN,use_test,free_params=free_params, NSGA = True, MU = MU)

    test_opt =  {str('multi_objective')+str(ga_out):ga_out}
    with open('multi_objective.p','wb') as f:
        pickle.dump(test_opt,f)
'''
MU = 6
NGEN = 200

import pickle

import numpy as np
try:
    with open('multi_objective.p','rb') as f:
        test_opt = pickle.load(f)
except:

for index, use_test in enumerate(test_frame.values()):

    if index % 2 == 0:
        index, DO = om.run_ga(explore_param,NGEN,use_test,free_params=free_params, NSGA = False, MU = MU)
    else:
        index, DO = om.run_ga(explore_param,NGEN,use_test,free_params=free_params, NSGA = False, MU = MU)
    #print(NSGA)

    print('can get as low as 2.70295, 2.70679')

    test_opt =  {str('multi_objective')+str(index):npcl}
    with open('multi_objective.p','wb') as f:
        pickle.dump(test_opt,f)


print(np.sum(list(test_opt['multi_objective']['pf'][2].dtc.scores.values())))
print(np.sum(list(test_opt['multi_objective']['pf'][1].dtc.scores.values())))
#print(np.sum(list(test_opt['multi_objective']['hof'][0].dtc.scores.values())))
print(test_opt['multi_objective']['pf'][2].dtc.scores.items())
print(test_opt['multi_objective']['pf'][1].dtc.scores.items())
'''


# In[ ]:




# In[ ]:

test_opt.keys()
for value in test_opt.values():
    value['stds']
    value['ranges']
    print(value['ranges'])
    print(value['stds'])

    #fig = pl.figure()
    #ax = pl.subplot(111)
    #ax.bar(range(len(value.keys())), values)


# In[ ]:

with open('data_dump.p','rb') as f:
    test_opt = pickle.load(f)


# In[ ]:

test_opt


# In[ ]:

#errorbar
#for
import seaborn as sns
from matplotlib.pyplot import errorbar
import matplotlib.pyplot as plt

fig0,ax0 = plt.subplots(dim,dim,figsize=(10,10))
plt.figure(num=None, figsize=(11, 11), dpi=80, facecolor='w', edgecolor='k')

for v in test_opt.values():
    x = 0
    labels = []
    plt.clf()
    for k_,v_ in v['ranges'].items():
        #print(k_)
        value = v_

        y = np.mean(value)
        err = max(value)-min(value)
        errorbar(x, y, err, marker='s', mfc='red',
                 mec='green', ms=2, mew=4,label='in '+str(k_))
        x+=1
        labels.append(k_)
    plt.xticks(np.arange(len(labels)), labels)
    ax0[i] = plt

    #plt.title(str(v))

plt.show()


# In[ ]:


from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn import datasets
import seaborn as sns; sns.set()  # for plot styling
dfs.replace([np.inf, -np.inf], np.nan)
dfs = dfs.dropna()

#X = dfs[['standard','sp','ss','info_density','gf','standard','uniqueness','info_density','penalty']]
X = dfs[['standard','sp','ss']]

X = X.as_matrix()
#import pdb; pdb.set_trace()

est = KMeans(n_clusters=3)

est.fit(X)

y_kmeans = est.predict(X)
centers = est.cluster_centers_

fignum = 1
fig = plt.figure(fignum, figsize=(4, 3))
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y_kmeans, s=50)
ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c='black', s=200, alpha=0.5);
ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('standard')
ax.set_ylabel('subjectivity')
ax.set_zlabel('sentiment polarity')
#ax.set_title(titles[fignum - 1])
#ax.dist = 12
fignum = fignum + 1
for x,i in enumerate(zip(y_kmeans,dfs['clue_words'])):
    try:
        #print(i[0],i[1],dfs['link'][x],dfs['publication'][x],dfs['clue_links'][x],dfs['sp_norm'][x],dfs['ss_norm'][x],dfs['uniqueness'][x])
    except:
            #print(i)

fig.savefig('3dCluster.png')


# # the parameter 'd' only seems important
# # C does not have to be too precise within a range.

# I consider the final gene populations for each of the eight tests. I compute the variance in each of the converged populations, I see that variance is low in many of the gene populations.
#
# When all variables are used to optomize only against one set of parameters, you expect their would be high variance in parameters, that don't matter much with respect to that error criteria (you expect redundancy of solutions).
#
# I compute std on errors over all the tests in order to estimate how amenable the problem is to multiobjective optimization.

# In[ ]:

from neuronunit.models.reduced import ReducedModel
from neuronunit.optimization.model_parameters import model_params, path_params
LEMS_MODEL_PATH = path_params['model_path']
import quantities as pq
plt.figure(num=None, figsize=(11, 11), dpi=80, facecolor='w', edgecolor='k')

for k,v in test_opt.items():
    model = ReducedModel(LEMS_MODEL_PATH, name= str('vanilla'), backend=('RAW'))
    model.attrs = v['out']['pf'][1].dtc.attrs
    print(str(k), v['out']['pf'][1].dtc.get_ss())#fitness)
    iparams = {}
    iparams['injected_square_current'] = {}
    iparams['injected_square_current']['amplitude'] =v['out']['pf'][1].rheobase['value']*pq.pA
    #['amplitude']  = dtc.vtest[k]['injected_square_current']['amplitude']
    DELAY = 100.0*pq.ms
    DURATION = 1000.0*pq.ms
    iparams['injected_square_current']['delay'] = DELAY
    iparams['injected_square_current']['duration'] = int(DURATION)

    model.inject_square_current(iparams)

    plt.plot(model.get_membrane_potential().times,model.get_membrane_potential(),label=str(k))
    plt.legend()
plt.show()


# In[ ]:

'''
#print([i.fitness.values for i in test_opt['t'][0]['pop']])#.keys()
print(np.std([i[0] for i in test_opt['t'][0]['pop'][0:5]]))#.keys()
print(np.std([i[1] for i in test_opt['t'][0]['pop'][0:5]]))#.keys()
print(np.std([i[2] for i in test_opt['t'][0]['pop'][0:5]]))#.keys()
print(np.std([i[3] for i in test_opt['t'][0]['pop'][0:5]]))#.keys()
print(test_opt['t'][0]['pop'][0][0])
print(test_opt['t'][0]['pop'][0][1])
test_opt['t'][0]['pop'][0].dtc.attrs
'''


# In[ ]:


#values = { k:v for v in npcl['pop'][i].dtc.attrs.items() for i in npcl['pop'] }
#print(values)
#print(stds.keys())
#stds
#dtc.variances[k] for k in dtc.attrs.keys()


# In[ ]:


DO.seed_pop = npcl['pf'][0:MU]
npcl, DO = om.run_ga(explore_param,10,reduced_tests,free_params=free_params,hc = hc, NSGA = False, MU = MU, seed_pop = DO.seed_pop)


# In[ ]:




# In[ ]:

attrs_here = npcl['hardened'][0][0].attrs
attrs_here.update(hc)
attrs_here
scores = npcl['hof'][0].dtc.scores
print(scores)


# In[ ]:

#
use_test = test_frame["Neocortex pyramidal cell layer 5-6"]
reduced_tests = [use_test[0], use_test[-1], use_test[len(use_test)-1]]
bigger_tests = use_test[1:-2]
bigger_tests.insert(0,use_test[0])


# In[ ]:

#bigger_tests = bigger_tests[-1::]
print(bigger_tests)


# In[ ]:

DO.seed_pop = npcl['hof'][0:MU]
reduced_tests = [use_test[0], use_test[-1], use_test[len(use_test)-1]]
npcl, DO = om.run_ga(explore_param,10,bigger_tests,free_params=free_params,hc = hc, NSGA = False, MU = MU)#, seed_pop = DO.seed_pop)


# In[ ]:

print(npcl['hardened'][0][0].attrs)
print(npcl['hardened'][0][0].scores)
print(npcl['pf'][0].fitness.values)
print(npcl['hof'][0].dtc.scores)

#for t in use_test:
#    print(t.name)


pop


# # From the scores printed above, it looks like certain error criteria, are in conflict with each other.
#
# Tests, that are amenable to co-optimization appear to be:
# * Width test
# * Input resistance tets
# * Resting Potential Test,
# * Capicitance test.
# * Time constant
#
# Tests/criteria that seem in compatible with the above include:
# * Rheobase,
# * InjectedCurrentAPThresholdTest
# * InjectedCurrentAPAmplitudeTest
#
# Therefore a reduced set of lists is made to check if the bottom three are at least amenable to optimization togethor.

# In[ ]:

from sklearn.cluster import KMeans
est = KMeans(n_clusters=2)
est.fit(X)
y_kmeans = est.predict(X)

centers = est.cluster_centers_

fig = plt.figure(fignum,figsize=(4,3))
ax = Axes3D(fig,rect=[0,0,.95,1],elav=48,azim=134)
ax.scatter(X[:,0],X[:,1],X[:,2],c=y_kmeans,s=50),
ax.scatter(centres[:,0],centres[:,1],centres[:,2],c='black',s=200,alpha=0.5)


# In[ ]:


print(reduced_tests)
print(bigger_tests)

DO.seed_pop = npcl['pf'][0:MU]
npcl, DO = om.run_ga(explore_param,10,reduced_tests,free_params=free_params,hc = hc, NSGA = True, MU = 12)#, seed_pop = DO.seed_pop)


# In[ ]:

import pickle
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from neuronunit.optimization.optimization_management import wave_measure
from neuronunit.models.reduced import ReducedModel
from neuronunit.optimization.model_parameters import model_params, path_params
LEMS_MODEL_PATH = path_params['model_path']
import neuronunit.optimization as opt
import quantities as pq
fig = plt.figure()

plt.clf()

from neuronunit.optimization.data_transport_container import DataTC
model = ReducedModel(LEMS_MODEL_PATH,name = str('vanilla'),backend = ('RAW'))
for i in npcl['pf'][0:2]:
    iparams = {}
    iparams['injected_square_current'] = {}
    iparams['injected_square_current']['amplitude'] =i.dtc.rheobase
    model = None
    model = ReducedModel(LEMS_MODEL_PATH,name = str('vanilla'),backend = ('RAW'))
    model.set_attrs(i.dtc.attrs)

    #['amplitude']  = dtc.vtest[k]['injected_square_current']['amplitude']
    DELAY = 100.0*pq.ms
    DURATION = 1000.0*pq.ms
    iparams['injected_square_current']['delay'] = DELAY
    iparams['injected_square_current']['duration'] = int(DURATION)
    model.inject_square_current(iparams)
    n_spikes = len(model.get_spike_train())
    if n_spikes:
        print(n_spikes)
        #print(i[0].scores['RheobaseTestP']*pq.pA)
        plt.plot(model.get_membrane_potential().times,model.get_membrane_potential())#,label='ground truth')
        plt.legend()

#gca().set_axis_off()
#subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
#            hspace = 0, wspace = 0)
#margins(0,0)
#gca().xaxis.set_major_locator(NullLocator())
#gca().yaxis.set_major_locator(NullLocator())

plt.subplots_adjust(left=0.0, right=1.0, top=0.9, bottom=0.1)
fig.tight_layout()
plt.show()

fig.savefig("single_trace.png", bbox_inches = 'tight',
    pad_inches = 0)


# In[ ]:

import pickle
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot


from neuronunit.models.reduced import ReducedModel
from neuronunit.optimization.model_parameters import model_params, path_params
LEMS_MODEL_PATH = path_params['model_path']
import neuronunit.optimization as opt
import quantities as pq
fig = plt.figure()

plt.clf()

from neuronunit.optimization.data_transport_container import DataTC
for i in npcl['hardened']:
    iparams = {}
    iparams['injected_square_current'] = {}
    iparams['injected_square_current']['amplitude'] = i[0].rheobase
    model = None
    model = ReducedModel(LEMS_MODEL_PATH,name = str('vanilla'),backend = ('RAW'))
    model.set_attrs(i[0].attrs)

    #['amplitude']  = dtc.vtest[k]['injected_square_current']['amplitude']
    DELAY = 100.0*pq.ms
    DURATION = 1000.0*pq.ms
    iparams['injected_square_current']['delay'] = DELAY
    iparams['injected_square_current']['duration'] = int(DURATION)
    model.inject_square_current(iparams)
    n_spikes = len(model.get_spike_train())
    if n_spikes:
        print(n_spikes)
        print(i[0].scores['RheobaseTestP']*pq.pA)
        plt.plot(model.get_membrane_potential().times,model.get_membrane_potential())#,label='ground truth')
        plt.legend()

#gca().set_axis_off()
#subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
#            hspace = 0, wspace = 0)
#margins(0,0)
#gca().xaxis.set_major_locator(NullLocator())
#gca().yaxis.set_major_locator(NullLocator())

plt.subplots_adjust(left=0.0, right=1.0, top=0.9, bottom=0.1)
fig.tight_layout()
plt.show()

fig.savefig("single_trace.png", bbox_inches = 'tight',
    pad_inches = 0)


'''
hc = {}

#free_params = ['c','k']
for k,v in explore_param.items():
    if k not in free_params:
        hc[k] = v
constants = npcl['hardened'][0][0].attrs
hc.update(constants)
npcl, _ = om.run_ga(explore_param,20,test_frame["Neocortex pyramidal cell layer 5-6"],free_params=free_params,hc = hc, NSGA = True)
'''


# In[ ]:


free_params = ['a','b','k']#vt','c','k','d']#,'vt','k','c','C']#,'C'] # this can only be odd numbers.

##
# Use information that is available
##
hc = reduced_cells['RS']

hc['vr'] = -65.2261863636364

hc['vPeak'] = hc['vr'] + 86.364525297619
hc['C'] = 89.7960714285714
hc.pop('a',0)
hc.pop('b',0)
hc.pop('k',0)
hc.pop('c',0)
hc.pop('d',0)

use_test = test_frame["Neocortex pyramidal cell layer 5-6"]
DO.seed_pop = npcl['pf']
ga_out = DO.run(max_ngen = 15)
'''
hc = {}

free_params = ['C']

for k,v in explore_param.items():
    if k not in free_params:
        hc[k] = v
#,'vt','k','c','C']#,'C'] # this can only be odd numbers
constants = npcl['hardened'][0][0].attrs
hc.update(constants)
npcl, _ = om.run_ga(explore_param,20,test_frame["Neocortex pyramidal cell layer 5-6"],free_params=free_params,hc = hc, NSGA = True)
'''


# In[ ]:

'''
import pandas

try:
    ne_raw = pandas.read_csv('article_ephys_metadata_curated.csv', delimiter='\t')
    !ls -ltr *.csv
except:
    !wget https://neuroelectro.org/static/src/article_ephys_metadata_curated.csv
    ne_raw = pandas.read_csv('article_ephys_metadata_curated.csv', delimiter='\t')

blah = ne_raw[ne_raw['NeuronName'].str.match('Hippocampus CA1 pyramidal cell')]
#ne_raw['NeuronName']
#ne_raw['cell\ capacitance']
#blah = ne_raw[ne_raw['NeuronName'].str.match('Hippocampus CA1 pyramidal cell')]

print([i for i in blah.columns])
#rint(blah['rheobase'])
#print(blah)
#for i in ne_raw.columns:#['NeuronName']:
#    print(i)

#ne_raw['NeuronName'][85]
#blah = ne_raw[ne_raw['TableID'].str.match('85')]
#ne_raw['n'] = 84
#here = ne_raw[ne_raw['Index']==85]
here = ne_raw[ne_raw['TableID']==18]

print(here['rheo_raw'])
#!wget https://neuroelectro.org/apica/1/n/
'''


# In[ ]:

ca1 = ne_raw[ne_raw['NeuronName'].str.match('Hippocampus CA1 pyramidal cell')]
ca1['rheo']


# In[ ]:



test_frame["Dentate gyrus basket cell"][0].observation['std'] = test_frame["Dentate gyrus basket cell"][0].observation['mean']
for t in test_frame["Dentate gyrus basket cell"]:
    print(t.name)

    print(t.observation)



#
# '''
# Inibitory Neuron
# This can't pass the Rheobase test
# '''

# In[ ]:


from neuronunit.optimization import optimization_management as om
import pickle

free_params = ['a','vr','b','vt','vPeak','c','k']
for k,v in explore_param.items():
    if k not in free_params:
        hc[k] = v
use_test = test_frame["Dentate gyrus basket cell"]
bcell, _ = om.run_ga(explore_param,20,use_test,free_params=free_params,hc = hc, NSGA = True, MU = 4)


# In[ ]:



#test_frame["Dentate gyrus basket cell"][0].observation['std'] = test_frame["Dentate gyrus basket cell"][0].observation['mean']
for t in test_frame["Hippocampus CA1 pyramidal cell"]:
    print(t.name)

    print(t.observation)


# In[ ]:

use_test = test_frame["Hippocampus CA1 pyramidal cell"]
bcell, _ = om.run_ga(explore_param,20,use_test,free_params=free_params,hc = hc, NSGA = True, MU = 10)


# In[ ]:

import pickle
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from neuronunit.models.reduced import ReducedModel
from neuronunit.optimization.model_parameters import model_params, path_params
LEMS_MODEL_PATH = path_params['model_path']
import neuronunit.optimization as opt
import quantities as pq
fig = plt.figure()

plt.clf()

from neuronunit.optimization.data_transport_container import DataTC
for i in bcell['hardened'][0:6]:
    iparams = {}
    iparams['injected_square_current'] = {}
    iparams['injected_square_current']['amplitude'] =i[0].rheobase
    model = None
    model = ReducedModel(LEMS_MODEL_PATH,name = str('vanilla'),backend = ('RAW'))
    model.set_attrs(i[0].attrs)

    #['amplitude']  = dtc.vtest[k]['injected_square_current']['amplitude']
    DELAY = 100.0*pq.ms
    DURATION = 1000.0*pq.ms
    iparams['injected_square_current']['delay'] = DELAY
    iparams['injected_square_current']['duration'] = int(DURATION)
    model.inject_square_current(iparams)
    n_spikes = len(model.get_spike_train())
    if n_spikes:
        print(n_spikes)
        print(i[0].scores['RheobaseTestP']*pq.pA)
        plt.plot(model.get_membrane_potential().times,model.get_membrane_potential())#,label='ground truth')
        plt.legend()

#gca().set_axis_off()
#subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
#            hspace = 0, wspace = 0)
#margins(0,0)
#gca().xaxis.set_major_locator(NullLocator())
#gca().yaxis.set_major_locator(NullLocator())

plt.subplots_adjust(left=0.0, right=1.0, top=0.9, bottom=0.1)
fig.tight_layout()
plt.show()

fig.savefig("single_trace.png", bbox_inches = 'tight',
    pad_inches = 0)


# In[ ]:




# In[ ]:

use_test = test_frame["Hippocampus CA1 pyramidal cell"]
bcell, _ = om.run_ga(explore_param,20,use_test,free_params=free_params,hc = hc, NSGA = True, MU = 10)


# # This cell is in markdown, but it won't be later.
# Later optimize a whole heap of cells in a loop.
#
# try:
#     import pickle
#     with open('data_dump.p','rb') as f:
#         test_opt = pickle.load(f)
# except:
#     MU = 12
#     NGEN = 25
#     cnt = 1
#     for t in use_test:
#         if cnt==len(use_test):
#             MU = 12
#             NGEN = 20
#
#             npcl, DO = om.run_ga(explore_param,NGEN,[t],free_params=free_params, NSGA = True, MU = MU)
#         else:
#
#             npcl, DO = om.run_ga(explore_param,NGEN,[t],free_params=free_params, NSGA = True, MU = MU)
#
#         test_opt[str(t)] =  {'out':npcl}
#
#         ranges = {}
#         stds = npcl['pop'][0].dtc.attrs
#         for k in npcl['pop'][0].dtc.attrs.keys():
#             stds[k] = []
#             ranges[k] = []
#
#
#         for i in npcl['pop'][::5]:
#             for k,v in i.dtc.attrs.items():
#                 stds[k].append(v)
#                 ranges[k].append(v)
#
#         for k in npcl['pop'][0].dtc.attrs.keys():
#             ranges[k] = (np.min(ranges[k][1::]),np.max(ranges[k][1::]))
#
#             stds[k] = np.std(stds[k][1::])
#         test_opt[str(t)]['stds'] = stds
#         test_opt[str(t)]['ranges'] = ranges
#
#         cnt+=1
#
#     with open('data_dump.p','wb') as f:
#         pickle.dump(test_opt,f)
#

# In[ ]:
