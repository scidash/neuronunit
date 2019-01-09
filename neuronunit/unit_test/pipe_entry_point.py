
import os
import pickle
from dask import distributed
import pickle
import pandas as pd
import timeit

from neuronunit.optimization import get_neab
from neuronunit.optimization.model_parameters import model_params
from bluepyopt.deapext.optimisations import SciUnitOptimization
from neuronunit.optimization.optimization_management import write_opt_to_nml
from neuronunit.optimization import optimization_management
from neuronunit.optimization import optimization_management as om

import numpy as np
import copy


electro_path = 'pipe_tests.p'
purkinje = { 'nlex_id':'sao471801888'}#'NLXWIKI:sao471801888'} # purkinje
fi_basket = {'nlex_id':'100201'}
#pvis_cortex = {'nlex_id':'nifext_50'} # Layer V pyramidal cell
olf_mitral = { 'nlex_id':'nifext_120'}
ca1_pyr = { 'nlex_id':'830368389'}
pipe = [ fi_basket, olf_mitral, ca1_pyr, purkinje ]


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

#MU = 4; NGEN = 3; CXPB = 0.9
USE_CACHED_GA = False

cnt = 0
pipe_results = {}
##
# TODO move to unit testing
##

start_time = timeit.default_timer()
#sel = [str('selNSGA2'),str('selIBEA'),str('')]
flat_iter = [ (test, observation) for test, observation in electro_tests ]#for s in sel ]
print(flat_iter)
from neuronunit.optimization import optimization_management as om

# http://www.physics.usyd.edu.au/teach_res/mp/mscripts/
# ns_izh002.m
import collections
from collections import OrderedDict

# Fast spiking cannot be reproduced as it requires modifications to the standard Izhi equation,
# which are expressed in this mod file.
# https://github.com/OpenSourceBrain/IzhikevichModel/blob/master/NEURON/izhi2007b.mod

reduced2007 = collections.OrderedDict([
  #              C    k     vr  vt vpeak   a      b   c    d  celltype
  ('RS',        (100, 0.7,  -60, -40, 35, 0.03,   -2, -50,  100,  1)),
  ('IB',        (150, 1.2,  -75, -45, 50, 0.01,   5, -56,  130,   2)),
  ('LTS',       (100, 1.0,  -56, -42, 40, 0.03,   8, -53,   20,   4)),
  ('TC',        (200, 1.6,  -60, -50, 35, 0.01,  15, -60,   10,   6)),
  ('TC_burst',  (200, 1.6,  -60, -50, 35, 0.01,  15, -60,   10,   6)),
  ('RTN',       (40,  0.25, -65, -45,  0, 0.015, 10, -55,   50,   7)),
  ('RTN_burst', (40,  0.25, -65, -45,  0, 0.015, 10, -55,   50,   7))])

import numpy as np
reduced_dict = OrderedDict([(k,[]) for k in ['C','k','vr','vt','vPeak','a','b','c','d']])

#OrderedDict
for i,k in enumerate(reduced_dict.keys()):
    for v in reduced2007.values():
        reduced_dict[k].append(v[i])

explore_param = {k:(np.min(v),np.max(v)) for k,v in reduced_dict.items()}
cnt = 0
for (test, observation) in flat_iter:
    dic_key = str(list(pipe[cnt].values())[0])
    init_time = timeit.default_timer()
    free_params = ['a','b','vr','k','vt','d']
    hc = ['C','c']
    #DO = SciUnitOptimization(error_criterion = test, selection = sel, provided_dict = model_params, elite_size = 3)
    start_time = timeit.default_timer()

    #ga_out, DO = om.run_ga(explore_param,5,TC_tests,free_params=free_params,hc = hc, NSGA = True, MU = 8)
    ga_out, DO = om.run_ga(explore_param,17,test,free_params=free_params,hc = hc, NSGA = True)
    elapsed = timeit.default_timer() - start_time
    ga_out['time_length'] = elapsed
    #package = DO.run(offspring_size = MU, max_ngen = 6, cp_frequency=1,cp_filename=str(dic_key)+'.p')
    #pop, hof_py, pf, log, history, td_py, gen_vs_hof = package
    with open('dump_all_cells'+str(pipe[cnt])+str(cnt),'wb') as f:
         pickle.dump(ga_out,f)
    cnt += 1


    print('entire duration', elapsed)

    '''
    model_to_write = pipe_results[dic_key]['gen_vs_hof'][-1].dtc.attrs

    optimization_management.write_opt_to_nml(file_name,model_to_write)




    times_list = list(pipe_results.values())
    ts = [ t['duration']/60.0 for t in times_list ]
    mean_time = np.mean(ts)
    total_time = np.sum(ts)
    '''
#    return pipe_results
#pipe_results = main_proc(flat_iter)
'''
pipe_results[dic_key] = {}
pipe_results[dic_key]['duration'] = finished_time - init_time
pipe_results[dic_key]['pop'] = copy.copy(pop)
pipe_results[dic_key]['sel'] = sel
pipe_results[dic_key]['hof'] = copy.copy(hof[::-1])
pipe_results[dic_key]['pf'] = copy.copy(pf[::-1])
pipe_results[dic_key]['log'] = copy.copy(log)
pipe_results[dic_key]['history'] = copy.copy(history)
pipe_results[dic_key]['td_py'] = copy.copy(td_py)
pipe_results[dic_key]['gen_vs_hof'] = copy.copy(gen_vs_hof)
pipe_results[dic_key]['sum_ranked_hof'] = [sum(i.dtc.scores.values()) for i in pipe_results[dic_key]['gen_vs_hof'][1:-1]]
pipe_results[dic_key]['componentsh'] = [list(i.dtc.scores.values()) for i in pipe_results[dic_key]['gen_vs_hof'][1:-1]]
pipe_results[dic_key]['componentsp'] = [list(i.dtc.scores.values()) for i in pipe_results[dic_key]['pf'][1:-1]]
file_name = str('nlex_id_')+dic_key
'''
