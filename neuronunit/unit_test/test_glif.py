import os
import sys
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

from neuronunit.optimization import get_neab #import get_neuron_criteria, impute_criteria
#from neuronunit.optimization.model_parameters import model_params
import os
import pickle
electro_path = 'pipe_tests.p'

assert os.path.isfile(electro_path) == True
with open(electro_path,'rb') as f:
    electro_tests = pickle.load(f)


MU = 12; NGEN = 12; CXPB = 0.9
USE_CACHED_GA = False

from neuronunit.models.interfaces import glif
gc = glif.GC()

#provided_keys = list(model_params.keys())
USE_CACHED_GS = False
from bluepyopt.deapext.optimisations import DEAPOptimisation
npoints = 10
#nparams = len(gc.nc.keys())


from dask import distributed
test = electro_tests[0][0]
print(gc.nc)
#import pdb; pdb.set_trace()
#for test, observation in electro_tests:

paramd0 = {k:v for k,v in gc.nc.items() if type(v) is type(0.01) }
paramd1 = {k:v for k,v in gc.nc.items() if type(v) is type(1) }
paramdict = {}
paramdict.update(paramd0)
paramdict.update(paramd1)

DO = DEAPOptimisation(error_criterion = test, selection = 'selIBEA', backend = 'glif', nparams = 10, provided_dict = paramdict)
pop, hof_py, log, history, td_py, gen_vs_hof = DO.run(offspring_size = MU, max_ngen = NGEN, cp_frequency=0,cp_filename='ga_dumpnifext_50.p')
import pdb; pdb.set_trace()
