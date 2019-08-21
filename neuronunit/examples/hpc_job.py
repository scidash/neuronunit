
# coding: utf-8

# Assumptions, the environment for running this notebook was arrived at by building a dedicated docker file.
#
# https://cloud.docker.com/repository/registry-1.docker.io/russelljarvis/nuo
# or more recently:
# https://cloud.docker.com/u/russelljarvis/repository/docker/russelljarvis/network_unit_opt
# You can run use dockerhub to get the appropriate file, and launch this notebook using Kitematic.

# # Import libraries
# To keep the standard running version of minimal and memory efficient, not all available packages are loaded by default. In the cell below I import a mixture common python modules, and custom developed modules associated with NeuronUnit (NU) development
# goals.
# given https://www.nature.com/articles/nn1352
# Goal is based on this. Don't optimize to a singular point, optimize onto a cluster.
# Golowasch, J., Goldman, M., Abbott, L.F, and Marder, E. (2002)
# Failure of averaging in the construction
# of conductance-based neuron models. J. Neurophysiol., 87: 11291131.

import numpy as np
import os
import pickle
import pandas as pd
from neuronunit.tests.fi import RheobaseTestP
#from neuronunit.optimisation.model_parameters import reduced_dict, reduced_cells
from neuronunit.optimisation import optimisation_management as om
from sciunit import scores# score_type

from neuronunit.optimisation.data_transport_container import DataTC
from neuronunit.tests.fi import RheobaseTestP# as discovery
from neuronunit.optimisation.optimisation_management import dtc_to_rheo, format_test, nunit_evaluation, grid_search
import quantities as pq
from neuronunit.models.reduced import ReducedModel
from neuronunit.optimisation.model_parameters import path_params
LEMS_MODEL_PATH = path_params['model_path']
list_to_frame = []
import copy
from sklearn.model_selection import ParameterGrid
from neuronunit.models.interfaces import glif
import matplotlib.pyplot as plt
from neuronunit.optimisation import get_neab

import pickle
from neuronunit import tests
from neuronunit import neuroelectro
import neuronunit.optimisation.model_parameters as model_params
MODEL_PARAMS = model_params.MODEL_PARAMS
MODEL_PARAMS['results'] = {}

from neuronunit.optimization import optimization_management as om

from collections import Iterable, OrderedDict
import quantities as qt
rts,complete_map = pickle.load(open('../tests/russell_tests.p','rb'))
#import pdb; pdb.set_trace()
local_tests = [value for value in rts['Hippocampus CA1 pyramidal cell'].values() ]
ga_out, DO = om.run_ga(MODEL_PARAMS['RAW'], 4, local_tests, free_params = MODEL_PARAMS['RAW'],
                            NSGA = True, MU = 12, model_type = str('RAW'))#,seed_pop=seeds[key])
=======
from neuronunit.optimisation import model_parameters as model_params
from neuronunit.optimisation.optimisation_management import inject_and_plot, cluster_tests
#from neuronunit.optimisation.optimisation_management import opt_pair


# # The Izhiketich model is instanced using some well researched parameter sets.
# First lets get the points in parameter space, that Izhikich himself has published about. These points are often used by the open source brain project to establish between model reproducibility. The itial motivating factor for choosing these points as constellations, of all possible parameter space subsets, is that these points where initially tuned and used as best guesses for matching real observed experimental recordings.
# ## Get the experimental Data pertaining to four different classes or neurons, that can constrain models.
# Next we get some electro physiology data for four different classes of cells that are very common targets of scientific neuronal modelling. We are interested in finding out what are the most minimal, and detail reduced, low complexity model equations, that are able to satisfy
# Below are some of the data set ID's I used to query neuroelectro.
# To save time for the reader, I prepared some data earlier to save time, and saved the data as a pickle, pythons preferred serialisation format.
# The interested reader can find some methods for getting cell specific ephys data from neuroelectro in a code file (neuronunit/optimisation/get_neab.py)

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
        try:
            tests,observations = get_neab.executable_druckman_tests(p)
        except:
            p_tests, p_observations = get_neab.get_neuron_criteria(p)
        obs_frame[p["name"]] = p_observations#, p_tests))
        test_frame[p["name"]] = p_tests#, p_tests))
    electro_path = str(os.getcwd())+'all_tests.p'
    with open(electro_path,'wb') as f:
        pickle.dump((obs_frame,test_frame),f)


for k,v in test_frame.items():
   if "Olfactory bulb (main) mitral cell" not in k:
       pass
   if "Olfactory bulb (main) mitral cell" in k:
       pass
df = pd.DataFrame.from_dict(obs_frame)



df['Hippocampus CA1 pyramidal cell']
>>>>>>> 51529ae8e9a02874e8b1d050bb812f2aec8d41d9

from neuronunit.optimisation.optimisation_management import stochastic_gradient_descent
import pdb

<<<<<<< HEAD
pdb.set_trace()

stochastic_gradient_descent(ga_out)
import sciunit
for cell,tt in rts.items():
    tests = [ y for x,y in tt.items() ]
    if len(tests):
        ga_out, DO = om.run_ga(MODEL_PARAMS['RAW'], 12 , rts['Hippocampus CA1 pyramidal cell'], free_params = MODEL_PARAMS['RAW'],
                                    NSGA = True, MU = 12, model_type = str('RAW'))#,seed_pop=seeds[key])

        #ga_out, _ = om.run_ga(MODEL_PARAMS['RAW']['EIF'],2 ,tests,free_params=MODEL_PARAMS['PYNN']['EIF'].keys(),
        #                            NSGA = True, MU = 4, model_type = str('PYNN'))#,seed_pop=seed)
        everything = list(zip(pop,dtcpop))

        #import pdb; pdb.set_trace()
        with open('GA_init_for_julia_objective_raw.p','wb') as f:
            pickle.dump(everything,f)



=======
>>>>>>> 51529ae8e9a02874e8b1d050bb812f2aec8d41d9
try:
    with open('Izh_seeds.p','rb') as f:
        seeds = pickle.load(f)

    assert seeds is not None

except:
    print('exceptional circumstances pickle file does not exist, rebuilding sparse grid for Izhikich')
    # Below we perform a sparse grid sampling over the parameter space, using the published and well corrobarated parameter points, from Izhikitch publications, and the Open Source brain, shows that without optimisation, using off the shelf parameter sets to fit real-life biological cell data, does not work so well.
    seeds, df = grid_search(MODEL_PARAMS['RAW'],
                            test_frame, backend=str('RAW'))


try:
    with open('multi_objective_raw.p','rb') as f:
        MODEL_PARAMS = pickle.load(f)
    assert len(MODEL_PARAMS['results']['RAW'].keys())
except:
    MU = 12 # more than six causes a memory leak. I suspect this is PYNN
    NGEN = 12
    test_opt = {}#{str('multi_objective_izhi')+str(ga_out):ga_out}
    MODEL_PARAMS['results'] = {}

    for key, use_test in test_frame.items():
        # use the best parameters found via the sparse grid search above, to inform the first generation
        # of the GA.
        if str('results') in MODEL_PARAMS['RAW'].keys():
            MODEL_PARAMS['RAW'].pop('results', None)
        ga_out, DO = om.run_ga(MODEL_PARAMS['RAW'], NGEN,use_test, free_params = MODEL_PARAMS['RAW'],
                                    NSGA = True, MU = MU, model_type = str('RAW'),seed_pop=seeds[key])
        try:
            print('optimization done, doing extra experimental work beyond the scope of the opt project')
            dtcpop = [ p.dtc for p in ga_out['pf'] ]
            measure_dtc_pop = opt_pair(dtcpop)
            ga_out['measure_dtc_pop'] = measure_dtc_pop
            print(ga_out['measure_dtc_pop'])
        except:
            print('failed on a new development feature, not critical to optimization')
        MODEL_PARAMS['results']['RAW'] = {}
        MODEL_PARAMS['results']['RAW'][key]  = ga_out
        with open('multi_objective_raw.p','wb') as f:
            pickle.dump(MODEL_PARAMS,f)
=======
for key, use_test in test_frame.items():
    # use the best parameters found via the sparse grid search above, to inform the first generation
    # of the GA.
    if str('results') in MODEL_PARAMS['RAW'].keys():
        MODEL_PARAMS['RAW'].pop('results', None)

    ga_out, _ = om.run_ga(MODEL_PARAMS['RAW'], NGEN,use_test, free_params = MODEL_PARAMS['RAW'],
                                NSGA = True, MU = MU, model_type = str('RAW'),seed_pop=seeds[key])

    MODEL_PARAMS['results']['RAW'] = {}
    MODEL_PARAMS['results']['RAW'][key]  = ga_out
>>>>>>> 51529ae8e9a02874e8b1d050bb812f2aec8d41d9



'''
try:
    with open('adexp_seeds.p','rb') as f:
        seeds = pickle.load(f)
    assert seeds is not None

except:
    seeds, df = grid_search(MODEL_PARAMS['PYNN']['EIF'],test_frame,backend=str('PYNN'))
    with open('adexp_seeds.p','rb') as f:
        pickle.dump(seeds,f)
'''
MU = 5
NGEN = 3
try:
    with open('multi_objective_adexp.p','rb') as f:
        test_opt = pickle.load(f)
except:

    MODEL_PARAMS['results']['PYNN'] = {}
    for key, use_test in test_frame.items():
        MODEL_PARAMS['RAW'].pop('results', None)

        ga_out, _ = om.run_ga(MODEL_PARAMS['PYNN']['EIF'],NGEN,use_test,free_params=MODEL_PARAMS['PYNN']['EIF'].keys(), NSGA = True, MU = MU, model_type = str('PYNN'))#,seed_pop=seed)

        MODEL_PARAMS['results']['PYNN'][key]  = ga_out
        with open('multi_objective_adexp.p','wb') as f:
            pickle.dump(MODEL_PARAMS,f)



# directly code in observations, that are direct model parameters
test_keyed_MODEL_PARAMS = {}
for k,v in test_frame.items():
    MODEL_PARAMS['GLIF']['R_input'] = v[1].observation
    MODEL_PARAMS['GLIF']['C'] = v[3].observation
    MODEL_PARAMS['GLIF']['init_AScurrents'] = [0,0]
    test_keyed_MODEL_PARAMS[k] = MODEL_PARAMS['GLIF']

#params = gc.glif.to_dict()
store_glif_results = {}
try:
    with open('glif_seeds.p','rb') as f:
        seeds = pickle.load(f)
    assert seeds is not None

except:
    # rewrite test search, so that model params change as a function of.
    seeds, df = grid_search(MODEL_PARAMS['GLIF'],test_frame,backend=str('GLIF'))

MU = 6
NGEN = 80


try:
    with open('multi_objective_glif.p','rb') as f:
        test_opt = pickle.load(f)

except:
    MODEL_PARAMS['GLIF']['results'] = {}
    for key, use_test in test_frame.items():
        seed = seeds[key]
        print(seed)
        # Todo optimise on clustered tests
        print(grouped_tests)
        print(grouped_tests)
        MODEL_PARAMS['GLIF'] = test_keyed_MODEL_PARAMS[k]
        ga_out, _ = om.run_ga(test_keyed_MODEL_PARAMS[k],NGEN,use_test,free_params=test_keyed_MODEL_PARAMS[k].keys(), NSGA = True, MU = MU, model_type = str('GLIF'),seed_pop=seed)

        MODEL_PARAMS['GLIF']['results'][key] = ga_out
        with open('multi_objective_glif.p','wb') as f:
            pickle.dump(MODEL_PARAMS,f)


try:
    with open('adexp_seeds.p','rb') as f:
        seeds = pickle.load(f)
    assert seeds is not None

except:
    seeds, df = grid_search(MODEL_PARAMS['PYNN'],test_frame,backend=str('PYNN'))

MU = 6
NGEN = 6

try:
    with open('multi_objective_adexp.p','rb') as f:
        test_opt = pickle.load(f)
except:
    MODEL_PARAMS['PYNN']['results'] = {}
    for key, use_test in test_frame.items():
        seed = seeds[key]
        print(seed)
        ga_out, _ = om.run_ga(MODEL_PARAMS['PYNN'],NGEN,use_test,free_params=explore_ranges.keys(), NSGA = True, MU = MU, model_type = str('PYNN'),seed_pop=seed)

        try:
            print('optimization done, doing extra experimental work beyond the scope of the opt project')
            dtcpop = [ p.dtc for p in ga_out['pf'] ]
            measure_dtc_pop = opt_pair(dtcpop)
            ga_out['measure_dtc_pop'] = measure_dtc_pop
            print(ga_out['measure_dtc_pop'])
        except:
            print('failed on a new development feature, not critical to optimization')
        MODEL_PARAMS['PYNN']['results'][key] = ga_out

        with open('multi_objective_adexp.p','wb') as f:
            pickle.dump(MODEL_PARAMS,f)

df = pd.DataFrame(index=list(test_frame.keys()),columns=list(reduced_cells.keys()))

MU = 6
NGEN = 90
gc = glif.GC()
glif_dic = gc.glif.to_dict()
explore_ranges = {}
gd = glif_dic
range_dic = {}

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
    if k not in explore_hh_ranges.keys():
        hold_constant_hh[k] = v


MU = 6
NGEN = 150


try:
    with open('multi_objective_HH.p','rb') as f:
        test_opt = pickle.load(f)

except:
    MODEL_PARAMS['HH']['results'] = {}

    for key, use_test in test_frame.items():
        seed = seeds[key]
        ga_out, _ = om.run_ga(explore_hh_ranges,NGEN,use_test,free_params=explore_ranges.keys(), NSGA = True, MU = MU, model_type = str('HH'),hc = hold_constant_hh,seed_pop=seed)
        MODEL_PARAMS['HH']['results'][key] = ga_out

        #test_opt =  {str('multi_objective_HH')+str(ga_out):ga_out}
        with open('multi_objective_HH.p','wb') as f:
            pickle.dump(test_opt,f)


#Next HH, model and Adaptive Exp.
#model = ReducedModel(LEMS_MODEL_PATH,name = str('vanilla'),backend = ('HH'))