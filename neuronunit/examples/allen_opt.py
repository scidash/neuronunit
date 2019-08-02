import pickle
from neuronunit.tests.allen_tests import test_collection, judge
ga_out = pickle.load(open('adexp_ca1.p','rb'))
from hide_imports import *
df = pd.DataFrame(rts)
ga_outad = {}
ga_outiz = {}
for key,v in rts.items():
    local_tests = [value for value in v.values() ]
    backend = str('BAE1')
    filename = str(key)+backend+str('.p')
    break
try:
    ga_outad[key] = pickle.load(open(filename,'rb'))
except:
    ga_outad[key], DO = om.run_ga(model_params.MODEL_PARAMS['BAE1'],1, local_tests, free_params = model_params.MODEL_PARAMS['BAE1'],
                                  NSGA = True, MU = 6, model_type = str('ADEXP'))
    pickle.dump(ga_outad[key],open(filename,'wb'))


#ga_out = pickle.load(open('izhi_ca1.p','rb'))
#ga_out = pickle.load(open('multi_objective_glif.p','rb'))
model = ga_outad['pf'][0]
model.static = None
model.static = False
scores = [ judge(model,t) for t in test_collection ]
