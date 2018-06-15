
import os
import pickle
from dask import distributed
import pickle
import pandas as pd
import timeit

from neuronunit.optimization import get_neab
from neuronunit.optimization.model_parameters import model_params
from bluepyopt.deapext.optimisations import DEAPOptimisation
from neuronunit.optimization.optimization_management import write_opt_to_nml
from neuronunit.optimization import optimization_management
from neuronunit.optimization import optimization_management as om

import numpy as np
import copy


electro_path = 'pipe_tests.p'
purkinje = { 'nlex_id':'sao471801888'}#'NLXWIKI:sao471801888'} # purkinje
fi_basket = {'nlex_id':'100201'}
pvis_cortex = {'nlex_id':'nifext_50'} # Layer V pyramidal cell
olf_mitral = { 'nlex_id':'nifext_120'}
ca1_pyr = { 'nlex_id':'830368389'}
pipe = [ fi_basket, pvis_cortex, olf_mitral, ca1_pyr, purkinje ]
electro_path = 'pipe_tests.p'

try:
    assert os.path.isfile(electro_path) == True
    with open(electro_path,'rb') as f:
        electro_tests = pickle.load(f)
    electro_tests = get_neab.replace_zero_std(electro_tests)
    electro_tests = get_neab.substitute_parallel_for_serial(electro_tests)

except:

    electro_tests = []
    for p in pipe:
       p_tests, p_observations = get_neab.get_neuron_criteria(p)
       electro_tests.append((p_tests, p_observations))

    electro_tests = get_neab.replace_zero_std(electro_tests)
    electro_tests = get_neab.substitute_parallel_for_serial(electro_tests)
    with open('pipe_tests.p','wb') as f:
       pickle.dump(electro_tests,f)

MU = 7; NGEN = 7; CXPB = 0.9

USE_CACHED_GA = False
#provided_keys = list(model_params.keys())
#npoints = 2
#nparams = 10


cnt = 0
pipe_results = {}

def check_dif(pipe_old,pipe_new):
    bool = False
    for key, value in pipe_results.items():
        if value != pipe_new[key]:
            bool = True
        print(value,pipe_new[key])

    return bool

start_time = timeit.default_timer()
# code you want to evaluate
'''
dic_key = str(list(pipe[0].values())[0])
cp_filename = str(dic_key)+'.p'
contents = pickle.load(open(cp_filename,'rb'))
print(contents, 'fail safe')
'''

for test, observation in electro_tests:
    dic_key = str(list(pipe[cnt].values())[0])
    init_time = timeit.default_timer()
    #print(cnt,len(electro_tests))
    DO = DEAPOptimisation(error_criterion = test, selection = 'selIBEA', provided_dict = model_params)
    package = DO.run(offspring_size = MU, max_ngen = 4, cp_frequency=1,cp_filename=str(dic_key)+'.p')
    pop, hof_py, log, history, td_py, gen_vs_hof = package
    finished_time = timeit.default_timer()
    pipe_results[dic_key] = {}
    pipe_results[dic_key]['duration'] = finished_time - init_time
    pipe_results[dic_key]['pop'] = copy.copy(pop)

    pipe_results[dic_key]['hof_py'] = copy.copy(hof_py)
    pipe_results[dic_key]['log'] = copy.copy(log)
    pipe_results[dic_key]['history'] = copy.copy(history)
    pipe_results[dic_key]['td_py'] = copy.copy(td_py)
    pipe_results[dic_key]['gen_vs_hof'] = copy.copy(gen_vs_hof)
    pipe_results[dic_key]['scored_hof'] = copy.copy(om.update_deap_pop(hof_py,test, td_py))
    old = 0
    alist = pipe_results[dic_key]['scored_hof'][::-1] # reverse the list.
    pipe_results[dic_key]['scored_hof'] = sorted(alist,key=lambda sum_val: sum(sum_val.dtc.scores.values()))

    for p in pipe_results[dic_key]['scored_hof']:
        temp = sum(p.dtc.scores.values())
        print(temp>=old)
        old = temp
    file_name = str('nlex_id_')+dic_key
    model_to_write = pipe_results[dic_key]['scored_hof'][0].dtc.attrs

    optimization_management.write_opt_to_nml(file_name,model_to_write)

    cnt += 1
    with open('dump_all_cells','wb') as f: pickle.dump(pipe_results,f)

    elapsed = timeit.default_timer() - start_time
    print('entire duration', elapsed)


    times_list = list(pipe_results.values())
    ts = [ t['duration']/60.0 for t in times_list ]
    mean_time = np.mean(ts)
    total_time = np.sum(ts)
