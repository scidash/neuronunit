from neuronunit.optimization import get_neab #import get_neuron_criteria, impute_criteria
import os
import pickle
from neuronunit.optimization.model_parameters import model_params
from bluepyopt.deapext.optimisations import DEAPOptimisation
from neuronunit.tests.fi import RheobaseTestP
from dask import distributed
import pickle

electro_path = 'pipe_tests.p'
fi_basket = {'nlex_id':'NLXCELL:100201'}
# https://scicrunch.org/scicrunch/interlex/view/ilx_0107386
pvis_cortex = {'nlex_id': 'nifext_50'} # Layer V pyramidal cell
#Hippocampal CA1 Pyramidal Neuron
#ca1_pyr = {'nlex_id': 'ILX:0105031' }
#NLXWIKI:sao830368389
# https://scicrunch.org/scicrunch/interlex/view/ilx_0101974
purkinje = { 'nlex_id':'NLXWIKI:sao471801888'} # purkinje
#https://scicrunch.org/scicrunch/interlex/view/ilx_0107933
olf_mitral = { 'nlex_id':'NLXWIKI:nifext_120'}
#https://scicrunch.org/scicrunch/interlex/view/ilx_0107386
ca1_pyr = { 'nlex_id':'SAO:830368389'}
pipe = [ fi_basket, pvis_cortex, ca1_pyr, purkinje, ca1_pyr ]

def replace_zero_std(electro_tests):
    for test,obs in electro_tests:
        test[0] = RheobaseTestP(obs['Rheobase'])
        for k,v in obs.items():
            if v['std'] == 0:
                obs = get_neab.substitute_criteria(electro_tests[1][1],obs)
    return electro_tests

try:
    assert os.path.isfile(electro_path) == True
    with open(electro_path,'rb') as f:
        electro_tests = pickle.load(f)
    electro_tests = replace_zero_std(electro_tests)
except:

    electro_tests = []
    contents[0][0].observation
    for p in pipe:
       p_tests, p_observations = get_neab.get_neuron_criteria(p)
       electro_tests.append((p_tests, p_observations))
    '''
    Add in the parallel rheobase test.
    '''
    electro_tests = replace_zero_std(electro_tests)
    with open('pipe_tests.p','wb') as f:
       pickle.dump(electro_tests,f)

MU = 6; NGEN = 6; CXPB = 0.9

USE_CACHED_GA = False
provided_keys = list(model_params.keys())

npoints = 2
nparams = 10

import numpy as np

cnt = 0
pipe_results = {}
import pandas as pd

def check_dif(pipe_old,pipe_new):
    bool = False
    for key, value in pipe_results.items():
        if value != pipe_new[key]:
            bool = True
        print(value,pipe_new[key])
    '''
    for index, val in enumerate(pipe_results.values()):
        if index == 0:
            attrs = pd.DataFrame(list(val['pop'][0].dtc.attrs.values())).T
        else:
            attrs = attrs.append(pd.DataFrame(list(val['pop'][0].dtc.attrs.values())).T)
    attrs.columns = val['pop'][0].dtc.attrs.keys()
    '''
    return bool

import pdb; pdb.set_trace()

for test, observation in electro_tests:
    dic_key = str(list(pipe[cnt].values())[0])
    DO = None
    DO = DEAPOptimisation(error_criterion = test, selection = 'selIBEA', provided_dict = model_params)
    pop, hof_py, log, history, td_py, gen_vs_hof = DO.run(offspring_size = MU, max_ngen = NGEN, cp_frequency=1,cp_filename=str(dic_key)+'.p')
    import pdb; pdb.set_trace()

    #with open(str(dic_key)+'.p','rb') as f:

    #    check_point = pickle.load(f)
    pipe_results[dic_key] = {}
    pipe_results[dic_key]['pop'] = pop
    pipe_results[dic_key]['hof_py'] = hof_py
    pipe_results[dic_key]['log'] = log
    pipe_results[dic_key]['history'] = history
    pipe_results[dic_key]['td_py'] = td_py
    pipe_results[dic_key]['gen_vs_hof'] = gen_vs_hof
    if cnt > 1:
        import pdb; pdb.set_trace()
        assert check_dif(pipe_old,pipe_results) == True
        assert np.mean(pipe_results[dic_key]['gen_vs_hof']) != np.mean(pipe_results[previous]['gen_vs_hof'])
        assert np.mean(pipe_results[dic_key]['history']) != np.mean(pipe_results[previous]['history'])
        pipe_old = pipe_results[dic_key]#['pop'][0].attrs
    previous = dic_key
    cnt += 1

with open('dump_all_cells','wb') as f:
   pickle.dump(pipe_results,f)



    #except:
    #pvis_criterion, pvis_observations = get_neab.get_neuron_criteria(p)

    #inh_criterion, inh_observations = get_neab.get_neuron_criteria(p)
#print(type(inh_observations),inh_observations)

#inh_observations = get_neab.substitute_criteria(pvis_observations,inh_observations)

#inh_criterion, inh_observations = get_neab.get_neuron_criteria(fi_basket,observation = inh_observations)
