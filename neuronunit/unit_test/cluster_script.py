
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


explore_param = {k:(np.min(v),np.max(v)) for k,v in reduced_dict.items()}


# ## Get the experimental Data pertaining to four different classes or neurons, that can constrain models.
# Next we get some electro physiology data for four different classes of cells that are very common targets of scientific neuronal modelling. We are interested in finding out what are the most minimal, and detail reduced, low complexity model equations, that are able to satisfy

# Below are some of the data set ID's I used to query neuroelectro.
# To save time for the reader, I prepared some data earlier to save time, and saved the data as a pickle, pythons preferred serialisation format.
#
# The interested reader can find some methods for getting cell specific ephys data from neuroelectro in a code file (neuronunit/optimization/get_neab.py)
#


purkinje ={"id": 18, "name": "Cerebellum Purkinje cell", "neuron_db_id": 271, "nlex_id": "sao471801888"}
fi_basket = {"id": 65, "name": "Dentate gyrus basket cell", "neuron_db_id": None, "nlex_id": "nlx_cell_100201"}
pvis_cortex = {"id": 111, "name": "Neocortex pyramidal cell layer 5-6", "neuron_db_id": 265, "nlex_id": "nifext_50"}
#does not have rheobase
olf_mitral = {"id": 129, "name": "Olfactory bulb (main) mitral cell", "neuron_db_id": 267, "nlex_id": "nlx_anat_100201"}
ca1_pyr = {"id": 85, "name": "Hippocampus CA1 pyramidal cell", "neuron_db_id": 258, "nlex_id": "sao830368389"}
pipe = [ fi_basket, ca1_pyr, purkinje,  pvis_cortex, olf_mitral ]

from neuronunit.optimization import get_neab


# In[ ]:

electro_tests = []
obs_frame = {}
test_frame = {}

try:
    #assert 1==2
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
    if "olf_mitral" not in k:
        obs = obs_frame[k]
    if "olf_mitral" in k:
        v[0] = RheobaseTestP(obs['Rheobase'])
        print('gets here?')
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


free_params = ['a','b','k','c','C','d','vPeak','vr','vt']
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


use_test[0].observation
print(use_test[0].name)

rtp = RheobaseTestP(use_test[0].observation)
use_test[0] = rtp
print(use_test[0].observation)


reduced_cells.keys()
#test_frame.keys()
#test_frame.keys()
#test_frame['Olfactory bulb (main) mitral cell'].insert(0,test_frame['Cerebellum Purkinje cell'][0])
test_frame




df = pd.DataFrame(index=list(test_frame.keys()),columns=list(reduced_cells.keys()))
MU = 6
NGEN = 200


model = ReducedModel(LEMS_MODEL_PATH,name = str('vanilla'),backend = (str('HH')))


explore_ranges = {'E_Na' : (40,70), 'g_Na':(100.0,140.0), 'C_m':(0.5,1.5)}



attrs_hh = { 'g_K' : 36.0, 'g_Na' : 120.0, 'g_L' : 0.3, \
         'C_m' : 1.0, 'E_L' : -54.387, 'E_K' : -77.0, 'E_Na' : 50.0, 'vr':-65.0 }




import copy
from sklearn.model_selection import ParameterGrid



from neuronunit.models.interfaces import glif
gc = glif.GC()

params = gc.glif.to_dict()
grid = ParameterGrid(explore_ranges)
store_hh_results = {}
for local_attrs in grid:
   store_hh_results[str(local_attrs.values())] = {}
   dtc = DataTC()
   dtc.tests = use_test
   print(dtc.attrs)
   dtc.backend = 'glif'
   dtc.cell_name = 'glif'
   for key, use_test in test_frame.items():
      dtc.tests = use_test
      dtc = dtc_to_rheo(dtc)
      dtc = format_test(dtc)
      if dtc.rheobase is not None:
         if dtc.rheobase!=-1.0:
            dtc = nunit_evaluation(dtc)
      print(dtc.get_ss())
      store_hh_results[str(local_attrs.values())][key] = dtc.get_ss()


try:
    #assert 1==2
    with open('HH_seeds.p','rb') as f:
        seeds = pickle.load(f)
    assert seeds is not None

except:

    grid = ParameterGrid(explore_ranges)
    store_hh_results = {}
    for local_attrs in grid:
        store_hh_results[str(local_attrs.values())] = {}
        dtc = DataTC()
        dtc.tests = use_test
        updatable_attrs = copy.copy(attrs_hh)
        updatable_attrs.update(local_attrs)
        dtc.attrs = updatable_attrs
        print(updatable_attrs)

        dtc.backend = 'HH'
        dtc.cell_name = 'Point Hodgkin Huxley'
        for key, use_test in test_frame.items():
            dtc.tests = use_test
            dtc = dtc_to_rheo(dtc)
            dtc = format_test(dtc)
            if dtc.rheobase is not None:
                if dtc.rheobase!=-1.0:
                    dtc = nunit_evaluation(dtc)
            print(dtc.get_ss())
            store_hh_results[str(local_attrs.values())][key] = dtc.get_ss()
    df = pd.DataFrame(store_hh_results)
    best_params = {}
    for index, row in df.iterrows():
        best_params[index] = row == row.min()
        best_params[index] = best_params[index].to_dict()


    seeds = {}
    for k,v in best_params.items():
        for nested_key,nested_val in v.items():
            if True == nested_val:
                seed = nested_key
                seeds[k] = seed
    with open('HH_seeds.p','wb') as f:
        pickle.dump(seeds,f)



attrs_hh = { 'g_K' : 36.0, 'g_Na' : 120.0, 'g_L' : 0.3, \
         'C_m' : 1.0, 'E_L' : -54.387, 'E_K' : -77.0, \
         'E_Na' : 50.0, 'vr':-65.0 }


explore_hh_ranges = {'E_Na' : (30,80), 'E_K': (-90.0,-75.0), 'g_K': (30.0,42.0),\
                    'C_m':(0.5,1.5), 'g_Na':(100.0,140.0),'g_L':(0.1,0.5), \
                    'E_L' : (-64.387,-44.387), 'vr':(-85.0,45.0)}



hold_constant_hh = {}
for k,v in attrs_hh.items():
    if k not in explore_ranges.keys():
        hold_constant_hh[k] = v


MU = 6
NGEN = 150


try:
    with open('multi_objective_HH.p','rb') as f:
        test_opt = pickle.load(f)

except:
    for key, use_test in test_frame.items():
        seed = seeds[key]
        print(seed)
        ga_out, _ = om.run_ga(explore_hh_ranges,NGEN,use_test,free_params=explore_ranges.keys(), NSGA = True, MU = MU, model_type = str('HH'),hc = hold_constant_hh)
        test_opt =  {str('multi_objective_HH')+str(ga_out):ga_out}
        with open('multi_objective_HH.p','wb') as f:
            pickle.dump(test_opt,f)




with open('Izh_seeds.p','rb') as f:
    seeds = pickle.load(f)

try:
    #assert 1==2
    assert seeds is not None

except:
    print('exceptional circumstances pickle file does not exist, rebuilding sparse grid for Izhikich')
    # Below we perform a sparse grid sampling over the parameter space, using the published and well corrobarated parameter points, from Izhikitch publications, and the Open Source brain, shows that without optimization, using off the shelf parameter sets to fit real-life biological cell data, does not work so well.

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
            df[k][key] = dtc.get_ss()

    best_params = {}
    for index, row in df.iterrows():
        best_params[index] = row == row.min()
        best_params[index] = best_params[index].to_dict()


    seeds = {}
    for k,v in best_params.items():
        for nested_key,nested_val in v.items():
            if True == nested_val:
                seed = reduced_cells[nested_key]
                seeds[k] = seed
    with open('Izh_seeds.p','wb') as f:
        pickle.dump(seeds,f)




MU = 6
NGEN = 150


for key, use_test in test_frame.items():

    # use the best parameters found via the sparse grid search above, to inform the first generation
    # of the GA.

    seed = seeds[key]
    print(seed)
    ga_out, _ = om.run_ga(explore_param,NGEN,use_test,free_params=free_params, NSGA = True, MU = MU,seed_pop = seed, model_type = str('RAW'))

    test_opt =  {str('multi_objective_izhi')+str(ga_out):ga_out}
    with open('multi_objective_izhi.p','wb') as f:
        pickle.dump(test_opt,f)



'''
Next  Adaptive Exp.
MU = 6
NGEN = 2
model_type = str('HH')

#Next  Adaptive Exp.
#model = ReducedModel(LEMS_MODEL_PATH,name = str('vanilla'),backend = ('HH'))
'''
