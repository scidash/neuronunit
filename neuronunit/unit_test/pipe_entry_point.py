from neuronunit.optimization import get_neab #import get_neuron_criteria, impute_criteria
import os
import pickle
electro_path = 'pipe_tests.p'
#try:
assert os.path.isfile(electro_path) == True
with open(electro_path,'rb') as f:
    electro_tests = pickle.load(f)

#except:
print('hello')
'''
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
import pickle
with open('test_pipe.p','wb') as handle:
    pickle.dump(pipe,handle)
contents = pickle.load(open('ne_neuron_criteria.p','rb'))
pvis_criterion, inh_criterion = contents
electro_tests = []
contents[0][0].observation
for p in pipe:
   p_tests, p_observations = get_neab.get_neuron_criteria(p)
   electro_tests.append((p_tests, p_observations))
with open('pipe_tests.p','wb') as f:
   pickle.dump(electro_tests,f)
'''
MU = 6; NGEN = 6; CXPB = 0.9
USE_CACHED_GA = False
print(get_neab)
from neuronunit.optimization.model_parameters import model_params
provided_keys = list(model_params.keys())
USE_CACHED_GS = False
from bluepyopt.deapext.optimisations import DEAPOptimisation
npoints = 2
nparams = 10

from dask import distributed
#c = distributed.Client()
#NCORES = len(c.ncores().values())-2
for test, observation in electro_tests:
    DO = DEAPOptimisation(error_criterion = test, selection = 'selIBEA')
    DO.setnparams(nparams = nparams, provided_keys = provided_keys)
    pop, hof_py, log, history, td_py, gen_vs_hof = DO.run(offspring_size = MU, max_ngen = NGEN, cp_frequency=0,cp_filename='ga_dumpnifext_50.p')

#with open('ga_dump_NLXCELL:100201.p','wb') as f:
#   pickle.dump([pop, log, history, hof_py, td_py],f)
from neuronunit.tests.fi import RheobaseTestP
for test,obs in electro_tests:
    test[0] = RheobaseTestP(obs)



    #except:
    #pvis_criterion, pvis_observations = get_neab.get_neuron_criteria(p)

    #inh_criterion, inh_observations = get_neab.get_neuron_criteria(p)
#print(type(inh_observations),inh_observations)

#inh_observations = get_neab.substitute_criteria(pvis_observations,inh_observations)

#inh_criterion, inh_observations = get_neab.get_neuron_criteria(fi_basket,observation = inh_observations)
