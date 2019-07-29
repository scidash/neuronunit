
import pickle
#from neuronunit.tests import allen_tests, allen_tests_utils
from neuronunit.tests.allen_tests import test_collection, judge
ga_out = pickle.load(open('adexp_ca1.p','rb'))
#ga_out = pickle.load(open('izhi_ca1.p','rb'))
#ga_out = pickle.load(open('multi_objective_glif.p','rb'))
model = ga_out['pf'][0]
model.static = None
model.static = False
scores = [ judge(model,t) for t in test_collection ]
