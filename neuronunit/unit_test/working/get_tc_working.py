import pickle
#bad_adexp = pickle.load(open('bad_adexp','rb'))
bad_izhi = pickle.load(open('bad_izhi','rb'))
bad_izhi_tests = pickle.load(open('bad_izhi_tests','rb'))
##from neuronunit.optimisation.optimisations import run_ga
from neuronunit.optimisation.optimisations import run_ga
#from neuronunit.optimisation import model_parameters, MODEL_PARAMS
from neuronunit.optimisation.model_parameters import MODEL_PARAMS
free_params = MODEL_PARAMS[str('RAW')]
#import pdb

#pdb.set_trace()
ga_out = {}
sub_test = {}
for i in bad_izhi:
   sub_test[i] = bad_izhi_tests[i]
   sub_test['protocol'] = 'elephant'
   ga_out[i] = run_ga(free_params, 4, sub_test, \
        free_params = free_params.keys(), MU = 4, seed_pop = None, \
        backend = str('RAW'),protocol={'allen':False,'elephant':True})
