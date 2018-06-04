import os
import sys
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

from neuronunit.models.interfaces import glif
gc = glif.GC()
from neuronunit.optimization import get_neab #import get_neuron_criteria, impute_criteria
from neuronunit.optimization.model_parameters import model_params

import os
import pickle
electro_path = 'pipe_tests.p'

assert os.path.isfile(electro_path) == True
with open(electro_path,'rb') as f:
    electro_tests = pickle.load(f)


MU = 6; NGEN = 6; CXPB = 0.9
USE_CACHED_GA = False

#provided_keys = list(model_params.keys())
USE_CACHED_GS = False
from bluepyopt.deapext.optimisations import DEAPOptimisation
npoints = 2
nparams = 10

from dask import distributed
test = electro_tests[0][0]

#for test, observation in electro_tests:
DO = DEAPOptimisation(error_criterion = test, selection = 'selIBEA', nparams = 10, provided_dict = model_params)
#DO = DEAPOptimisation(error_criterion = test, selection = 'selIBEA', backend = 'glif')
#DO.setnparams(nparams = nparams, provided_keys = provided_keys)
pop, hof_py, log, history, td_py, gen_vs_hof = DO.run(offspring_size = MU, max_ngen = NGEN, cp_frequency=0,cp_filename='ga_dumpnifext_50.p')
