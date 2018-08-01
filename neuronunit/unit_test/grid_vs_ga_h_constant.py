import timeit
import numpy as np
import matplotlib
import shelve
matplotlib.use('Agg')

#import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math as math
from pylab import rcParams
from neuronunit.optimization.optimization_management import run_ga
from neuronunit.optimization.exhaustive_search import run_grid, reduce_params, create_grid
from neuronunit.optimization.model_parameters import model_params


import os
import pickle
from neuronunit.optimization.results_analysis import min_max, error_domination, param_distance


from neuronunit.optimization.results_analysis import make_report
from neuronunit.optimization import get_neab

import copy
import time

from neuronunit.optimization import exhaustive_search as es
import numpy as np

reports = {}


electro_path = str(os.getcwd())+'/pipe_tests.p'
print(os.getcwd())
assert os.path.isfile(electro_path) == True
with open(electro_path,'rb') as f:
    electro_tests = pickle.load(f)

electro_tests = get_neab.replace_zero_std(electro_tests)
electro_tests = get_neab.substitute_parallel_for_serial(electro_tests)
test, observation = electro_tests[0]


tests = copy.copy(electro_tests[0][0][0:2])

#tests = copy.copy(electro_tests[0][0])



with open('dim_3_errs8_ga.p','rb') as f:
    package = pickle.load(f)



# timing performance profiling

import time
start_time = timeit.default_timer()

# plotting tools
try:
    from prettyplotlib import plt
except:
    import matplotlib.pyplot as plt

# import NU tests

from neuronunit.optimization import get_neab
import copy

electro_path = str(os.getcwd())+'/pipe_tests.p'
print(os.getcwd())
assert os.path.isfile(electro_path) == True
with open(electro_path,'rb') as f:
    electro_tests = pickle.load(f)

electro_tests = get_neab.replace_zero_std(electro_tests)
electro_tests = get_neab.substitute_parallel_for_serial(electro_tests)
test, observation = electro_tests[0]

npoints = 10

opt_keys = [str('a'),str('C'),str('b')]
nparams = len(opt_keys)
ga_out = run_ga(model_params,nparams,npoints,tests,provided_keys = opt_keys, use_cache = True, cache_name='simple')
'''
fname = 'dim_{0}_errs{1}_ga.p'.format(len(opt_keys),len(tests))
with open(fname,'wb') as f:
   pickle.dump([ga_out,opt_keys,tests,elapsed_ga],f)


with open(fname,'rb') as f:
    package = pickle.load(f)

'''

pop = ga_out[0]
ga_keys = list(pop[0].dtc.attrs)

grid_results = {}
hof = package[1]

with shelve.open('hcg.p') as db:
    db['grid_results'] = None

import os
assert os.path.isfile('hcg.p') is True




for i in range(len(ga_keys)):
    for j in range(len(ga_keys)):
        if i!=j:
            provided_keys = [ga_keys[j],ga_keys[i]]
            ss = set(provided_keys)
            bs = set(attrs_list)
            # this finds the parameter that you do not have the explicit index for:
            diff = bs.difference(ss)
            bd =  {}
            for i in range(0,len(diff)):
                bd[list(diff)[i]] = hof[0].dtc.attrs[list(diff)[i]]
            provided_keys.append(attrs_list[i])
            provided_keys.append(attrs_list[j])
            gr = run_grid(1,1,tests,provided_keys = provided_keys ,hold_constant = bd)
            assert gr[0] != gr[-1]
            key = str(attrs_list[i])+str(attrs_list[j])
            grid_results[key] = gr


for i in range(len(ga_keys)):
    for j in range(len(ga_keys)):

        if i == j:
            provided_keys = [ ga_keys[i] ]
            ss = set(provided_keys)
            bs = set(ga_keys)
            # this finds the parameter that you do not have the explicit index for:
            diff = bs.difference(ss)
            bd =  {}
            for i in range(0,len(diff)):
                bd[list(diff)[i]] = hof[0].dtc.attrs[list(diff)[i]]

            provided_keys.append(attrs_list[i])
            assert len(bd) > len(provided_keys)
            assert len(bd) == 2 and len(provided_keys) == 1

            gr = run_grid(10,tests,provided_keys = provided_keys ,hold_constant = bd, use_cache = True, cache_name='complex')

            # do a 1D exhaustive search where everything except parameter i is held constant.

        if i>j:

            provided_keys = [ga_keys[j],ga_keys[i]]
            ss = set(provided_keys)
            bs = set(attrs_list)
            # this finds the parameter that you do not have the explicit index for:
            diff = bs.difference(ss)
            bd =  {}
            for i in range(0,len(diff)):
                bd[list(diff)[i]] = hof[0].dtc.attrs[list(diff)[i]]

            provided_keys.append(attrs_list[i])
            provided_keys.append(attrs_list[j])
            gr = run_grid(2,10,tests,provided_keys = provided_keys ,hold_constant = bd, use_cache = True, cache_name='complex')
            key = str(attrs_list[i])+str(attrs_list[j])
        grid_results[key] = gr
        with shelve.open('hcg.p') as db:
            db['grid_results'] = grid_results
