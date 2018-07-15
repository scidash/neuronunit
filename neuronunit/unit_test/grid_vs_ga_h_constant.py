import timeit
import numpy as np
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
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
reports = {}


electro_path = str(os.getcwd())+'/pipe_tests.p'
print(os.getcwd())
assert os.path.isfile(electro_path) == True
with open(electro_path,'rb') as f:
    electro_tests = pickle.load(f)

electro_tests = get_neab.replace_zero_std(electro_tests)
electro_tests = get_neab.substitute_parallel_for_serial(electro_tests)
test, observation = electro_tests[0]

npoints = 6
tests = copy.copy(electro_tests[0][0][0:2])
#tests+= electro_tests[0][0][4:-1]
#tests = copy.copy(electro_tests[0][0])




with open('pre_ga_reports.p','rb') as f:
    package = pickle.load(f)




#opt_keys = list(copy.copy(grid_results)[0].dtc.attrs.keys())
#ga_out = run_ga(model_params,nparams,npoints,tests,provided_keys = opt_keys)
#ga_out = run_ga(model_params,10,npoints,tests)#,provided_keys = opt_keys)

import time
start_time = timeit.default_timer()

try:
    from prettyplotlib import plt
except:
    import matplotlib.pyplot as plt
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

npoints = 6
tests = copy.copy(electro_tests[0][0][0:2])
import numpy as np
ax = None
from neuronunit.optimization import exhaustive_search as es



with open('pre_ga_reports.p','rb') as f:
    package = pickle.load(f)

pop = package[0]
fig,ax = plt.subplots(3,3,figsize=(10,10))
attrs_list = list(pop[0].dtc.attrs)

from neuronunit import plottools
plot_surface = plottools.plot_surface
grid_results = {}
hof = package[1]
for i in range(len(attrs_list)):
    for j in range(len(attrs_list)):
        if i<j:
            ax[i,j].set_title('Param {0} vs Param {1}'.format(attrs_list[i],attrs_list[j]))

            x = [ g.dtc.attrs[attrs_list[i]] for g in pop ]
            y = [ g.dtc.attrs[attrs_list[j]] for g in pop ]
            z = [ sum(list(g.dtc.scores.values())) for g in pop ]
            ax[i,j].scatter(y,x,c=z)
            ax[i,j].set_xlim(np.min(y),np.max(y))
            ax[i,j].set_ylim(np.min(x),np.max(x))


        if i == j:
            ax[i,j].set_title('Param {0} vs score'.format(attrs_list[i]))

            x = [ g.dtc.attrs[attrs_list[i]] for g in pop ]
            z = [ sum(list(g.dtc.scores.values())) for g in pop ]
            ax[i,i].scatter(x,z)
            ax[i,j].set_xlim(np.min(x),np.max(x))


        elif i>j:
            ax[i,j].set_title('Param {0} vs Param {1}'.format(attrs_list[i],attrs_list[j]))
            provided_keys = [attrs_list[j],attrs_list[i]]
            ss = set(provided_keys)
            bs = set(attrs_list)
            # this finds the parameter that you do not have the explicit index for:
            diff = bs.difference(ss)
            bd =  {}
            bd[list(diff)[0]] = hof[0].dtc.attrs[list(diff)[0]]


            #ax_trip,plot_axis = plot_surface(fig,ax[i,j],attrs_list[j],attrs_list[i],history)
            #ax[i,j].plot_axis = plot_axis
            #provided_keys = []
            provided_keys.append(attrs_list[i])
            provided_keys.append(attrs_list[j])
            gr = es.run_grid(2,10,tests,provided_keys = provided_keys ,hold_constant = bd)
            grid_results[str(attrs_list[i])+str(attrs_list[j])] = gr

            with open('held_constant_grid'+str(nparams)+str('.p'),'wb') as f:
                pickle.dump(grid_results,f)
            #print(best)

            # here access the GA's optimum for that parameter
            #ax[i,j].pcolormesh(Z)
