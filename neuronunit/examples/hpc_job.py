
# coding: utf-8

# Assumptions, the environment for running this notebook was arrived at by building a dedicated docker file.
#
# https://cloud.docker.com/repository/registry-1.docker.io/russelljarvis/nuo
# or more recently:
# https://cloud.docker.com/u/russelljarvis/repository/docker/russelljarvis/network_unit_opt
# You can run use dockerhub to get the appropriate file, and launch this notebook using Kitematic.

# # Import libraries
# To keep the standard running version of minimal and memory efficient, not all available packages are loaded by default. In the cell below I import a mixture common python modules, and custom developed modules associated with NeuronUnit (NU) development
#!pip install dask distributed seaborn
#!bash after_install.sh


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
from sciunit import scores# score_type

from neuronunit.optimisation.data_transport_container import DataTC
from neuronunit.tests.fi import RheobaseTestP# as discovery
from neuronunit.optimisation.optimisation_management import dtc_to_rheo, format_test, nunit_evaluation, grid_search
import quantities as pq
from neuronunit.models.reduced import ReducedModel
from neuronunit.optimisation.model_parameters import path_params
LEMS_MODEL_PATH = path_params['model_path']
list_to_frame = []
#from neuronunit.tests.fi import RheobaseTestP
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

from neuronunit.optimisation.optimisation_management import stochastic_gradient_descent
import pdb

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

        try:
            print('optimization done, doing extra experimental work beyond the scope of the opt project')
            dtcpop = [ p.dtc for p in ga_out['pf'] ]
            measure_dtc_pop = opt_pair(dtcpop)
            ga_out['measure_dtc_pop'] = measure_dtc_pop
            print(ga_out['measure_dtc_pop'])
        except:
            print('failed on a new development feature, not critical to optimization')


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

        try:
            print('optimization done, doing extra experimental work beyond the scope of the opt project')
            dtcpop = [ p.dtc for p in ga_out['pf'] ]
            measure_dtc_pop = opt_pair(dtcpop)
            ga_out['measure_dtc_pop'] = measure_dtc_pop
            print(ga_out['measure_dtc_pop'])
        except:
            print('failed on a new development feature, not critical to optimization')

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
