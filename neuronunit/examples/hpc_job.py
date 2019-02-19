
# coding: utf-8

# Assumptions, the environment for running this notebook was arrived at by building a dedicated docker file.
#
# https://cloud.docker.com/repository/registry-1.docker.io/russelljarvis/nuo
#
# You can run use dockerhub to get the appropriate file, and launch this notebook using Kitematic.

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
import copy
from sklearn.model_selection import ParameterGrid
from neuronunit.models.interfaces import glif
from neuronunit.optimization.data_transport_container import DataTC
from neuronunit.optimization.optimization_management import grid_search
import matplotlib.pyplot as plt

# # The Izhiketich model is instanced using some well researched parameter sets.
# First lets get the points in parameter space, that Izhikich himself has published about. These points are often used by the open source brain project to establish between model reproducibility. The itial motivating factor for choosing these points as constellations, of all possible parameter space subsets, is that these points where initially tuned and used as best guesses for matching real observed experimental recordings.

explore_param = {k:(np.min(v),np.max(v)) for k,v in reduced_dict.items()}

# ## Get the experimental Data pertaining to four different classes or neurons, that can constrain models.
# Next we get some electro physiology data for four different classes of cells that are very common targets of scientific neuronal modelling. We are interested in finding out what are the most minimal, and detail reduced, low complexity model equations, that are able to satisfy
# Below are some of the data set ID's I used to query neuroelectro.
# To save time for the reader, I prepared some data earlier to save time, and saved the data as a pickle, pythons preferred serialisation format.
# The interested reader can find some methods for getting cell specific ephys data from neuroelectro in a code file (neuronunit/optimization/get_neab.py)


purkinje ={"id": 18, "name": "Cerebellum Purkinje cell", "neuron_db_id": 271, "nlex_id": "sao471801888"}
fi_basket = {"id": 65, "name": "Dentate gyrus basket cell", "neuron_db_id": None, "nlex_id": "nlx_cell_100201"}
pvis_cortex = {"id": 111, "name": "Neocortex pyramidal cell layer 5-6", "neuron_db_id": 265, "nlex_id": "nifext_50"}
#does not have rheobase
olf_mitral = {"id": 129, "name": "Olfactory bulb (main) mitral cell", "neuron_db_id": 267, "nlex_id": "nlx_anat_100201"}
ca1_pyr = {"id": 85, "name": "Hippocampus CA1 pyramidal cell", "neuron_db_id": 258, "nlex_id": "sao830368389"}
pipe = [ fi_basket, ca1_pyr, purkinje,  pvis_cortex,olf_mitral]

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

# idea something like:
# test_frame['Olfactory bulb (main) mitral cell'].insert(0,test_frame['Cerebellum Purkinje cell'][0])

for k,v in test_frame.items():
   if "Olfactory bulb (main) mitral cell" not in k:
       pass
   if "Olfactory bulb (main) mitral cell" in k:
       pass
       #v[0] = RheobaseTestP(obs['Rheobase'])
df = pd.DataFrame.from_dict(obs_frame)



# In the data frame below, you can see many different cell types
df['Hippocampus CA1 pyramidal cell']
# # Tweak Izhikitich equations
# with educated guesses based on information that is already encoded in the predefined experimental observations.
# In otherwords use information that is readily amenable into hardcoding into equations
# Select out the 'Neocortex pyramidal cell layer 5-6' below, as a target for optimization

#!pip install lazyarray
clustered_tests = pickle.load(open('clustered_tests.p','rb'))
grouped_tests = clustered_tests['gtc']
grouped_testsn = clustered_tests['gtn']
from neuronunit.optimization import model_parameters as model_params
from neuronunit.optimization.optimization_management import inject_and_plot, cluster_tests

import pyNN
from pyNN import neuron
from pyNN.neuron import EIF_cond_exp_isfa_ista
#neurons = pyNN.Population(N_CX, pyNN.EIF_cond_exp_isfa_ista, RS_parameters)

EIF = model_params.EIF
'''
import pdb
pdb.set_trace()

from neuronunit.tests import RheobaseTestP, fi#, RheobaseTest
dtc = DataTC()
dtc.attrs = EIF
dtc.backend = str('PYNN')
dtc.cell_name = str('PYNN')
dtc.tests = test_frame[list(test_frame.keys())[0]]
dtc = dtc_to_rheo(dtc)

#import time
#model = ReducedModel(LEMS_MODEL_PATH,name = str('vanilla'),backend = (str('PYNN')))
#model.set_attrs(cell[0].get_parameters())
rt = fi.RheobaseTest(obs_frame['Dentate gyrus basket cell']['Rheobase'])
dtc.rheobase = rt
#pred1 = rt.generate_prediction(model)
inject_and_plot(dtc,figname='EIF_problem')
for key, use_test in test_frame.items():
    grouped_tests, grouped_tests = cluster_tests(use_test,str('PYNN'),EIF)

dtc = format_test(dtc)
dtc.vtest[0]['injected_square_current']['amplitude']= dtc.vtest[0]['injected_square_current']['amplitude']
model.inject_square_current(dtc.vtest[0])

#model.inject_square_current(pred1)
vm = model.get_membrane_potential()
plt.clf()
plt.plot(vm.times,vm)
plt.savefig('debug.png')

dtc = nunit_evaluation(dtc)
'''




try:
    with open('adexp_seeds.p','rb') as f:
        seeds = pickle.load(f)
    assert seeds is not None

except:
    seeds, df = grid_search(EIF,test_frame,backend=str('PYNN'))

MU = 6
NGEN = 80

try:
    with open('multi_objective_adexp.p','rb') as f:
        test_opt = pickle.load(f)
except:
    for key, use_test in test_frame.items():
        seed = seeds[key]
        print(seed)
        ga_out, _ = om.run_ga(EIF,NGEN,use_test,free_params=explore_ranges.keys(), NSGA = True, MU = MU, model_type = str('PYNN'),seed_pop=seed)
        test_opt =  {str('multi_objective_PYNN')+str(seed):ga_out}
        with open('multi_objective_adexp.p','wb') as f:
            pickle.dump(test_opt,f)

df = pd.DataFrame(index=list(test_frame.keys()),columns=list(reduced_cells.keys()))

MU = 6
NGEN = 90
gc = glif.GC()
glif_dic = gc.glif.to_dict()
explore_ranges = {}
gd = glif_dic
range_dic = {}

# P4pg0k purkinje
# QDKPDQ slice of monkey brain
# YVP 5PD skull

# sketchfab
# autconverted format.


# directly code in observations, that are direct model parameters
for k,v in test_frame.items():
    range_dic['R_input'] = v[1].observation
    range_dic['C'] = v[3].observation
    range_dic['th_inf'] = v[7].observation
explore_ranges['El'] = (glif_dic['El'],glif_dic['El']+10.0)
#explore_ranges['R_input'] = (glif_dic['R_input']-glif_dic['R_input']/2.0,glif_dic['R_input']+glif_dic['R_input']/2.0)
#explore_ranges['C'] = (glif_dic['C']-glif_dic['C']/2.0,glif_dic['C']+glif_dic['C']/2.0)
#explore_ranges['th_inf'] = (glif_dic['th_inf']-glif_dic['th_inf']/4.0,glif_dic['th_inf']+glif_dic['th_inf']/4.0)
model = ReducedModel(LEMS_MODEL_PATH,name = str('vanilla'),backend = (str('GLIF')))

store_glif_results = {}
params = gc.glif.to_dict()
store_glif_results = {}
try:
    with open('glif_seeds.p','rb') as f:
        seeds = pickle.load(f)
    assert seeds is not None

except:
    seeds, df = grid_search(explore_ranges,test_frame,backend=str('GLIF'))

MU = 6
NGEN = 80


try:
    with open('multi_objective_glif.p','rb') as f:
        test_opt = pickle.load(f)

except:
    for key, use_test in test_frame.items():
        seed = seeds[key]
        print(seed)
        # Todo optimize on clustered tests
        print(grouped_tests)
        print(grouped_tests)

        ga_out, _ = om.run_ga(explore_ranges,NGEN,use_test,free_params=explore_ranges.keys(), NSGA = True, MU = MU, model_type = str('GLIF'),seed_pop=seed)
        test_opt =  {str('multi_objective_glif')+str(seed):ga_out}
        with open('multi_objective_glif.p','wb') as f:
            pickle.dump(test_opt,f)




model = ReducedModel(LEMS_MODEL_PATH,name = str('vanilla'),backend = (str('HH')))
explore_ranges = {'E_Na' : (40,70), 'g_Na':(100.0,140.0), 'C_m':(0.5,1.5)}
attrs_hh = { 'g_K' : 36.0, 'g_Na' : 120.0, 'g_L' : 0.3, \
         'C_m' : 1.0, 'E_L' : -54.387, 'E_K' : -77.0, 'E_Na' : 50.0, 'vr':-65.0 }

try:
    with open('HH_seeds.p','rb') as f:
        seeds = pickle.load(f)
    assert seeds is not None

except:
    seeds, df = grid_search(explore_ranges,test_frame,backend=str('HH'))


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
        ga_out, _ = om.run_ga(explore_hh_ranges,NGEN,use_test,free_params=explore_ranges.keys(), NSGA = True, MU = MU, model_type = str('HH'),hc = hold_constant_hh,seed_pop=seed)
        test_opt =  {str('multi_objective_HH')+str(ga_out):ga_out}
        with open('multi_objective_HH.p','wb') as f:
            pickle.dump(test_opt,f)

# # The Izhiketich model is instanced using some well researched parameter sets.
# First lets get the points in parameter space, that Izhikich himself has published about. These points are often used by the open source brain project to establish between model reproducibility. The itial motivating factor for choosing these points as constellations, of all possible parameter space subsets, is that these points where initially tuned and used as best guesses for matching real observed experimental recordings.

explore_param = {k:(np.min(v),np.max(v)) for k,v in reduced_dict.items()}


# # Tweak Izhikitich equations
# with educated guesses based on information that is already encoded in the predefined experimental observations.
# In otherwords use information that is readily amenable into hardcoding into equations
# Select out the 'Neocortex pyramidal cell layer 5-6' below, as a target for optimization

free_params = ['a','b','k','c','C','d','vPeak','vr','vt']
hc_ = reduced_cells['RS']
hc_['vr'] = -65.2261863636364
hc_['vPeak'] = hc_['vr'] + 86.364525297619
explore_param['C'] = (hc_['C']-20,hc_['C']+20)
explore_param['vr'] = (hc_['vr']-5,hc_['vr']+5)
use_test = test_frame["Neocortex pyramidal cell layer 5-6"]
test_opt = {}
with open('data_dump.p','wb') as f:
    pickle.dump(test_opt,f)
use_test[0].observation
rtp = RheobaseTestP(use_test[0].observation)
use_test[0] = rtp
reduced_cells.keys()


with open('Izh_seeds.p','rb') as f:
    seeds = pickle.load(f)

try:

    assert seeds is not None

except:
    print('exceptional circumstances pickle file does not exist, rebuilding sparse grid for Izhikich')
    # Below we perform a sparse grid sampling over the parameter space, using the published and well corrobarated parameter points, from Izhikitch publications, and the Open Source brain, shows that without optimization, using off the shelf parameter sets to fit real-life biological cell data, does not work so well.
    seeds, df = grid_search(explore_ranges,test_frame,backend=str('RAW'))



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




#Next HH, model and Adaptive Exp.
#model = ReducedModel(LEMS_MODEL_PATH,name = str('vanilla'),backend = ('HH'))
