from allensdk.ephys.extract_cell_features import extract_cell_features
import pickle
from neuronunit.optimisation.optimization_management import OptMan



try:
    with open('allen_test_data.p','rb') as f:
        pre_obs = pickle.load(f)
except:
    from neuronunit.tests.allen_tests import pre_obs#, test_collection
    with open('allen_test_data.p','wb') as f:
        pickle.dump(pre_obs,f)

local_tests = pre_obs[2][1]
local_tests.update(pre_obs[2][1]['spikes'][0])
local_tests['current_test'] = pre_obs[1][0]
local_tests['spk_count'] = len(pre_obs[2][1]['spikes'])

OM = OptMan(local_tests,protocol={'elephant':False,'allen':True,'dm':False},confident=False)
res = OM.round_trip_test(local_tests,str('RAW'),MU=6,NGEN=6)#,stds = easy_standards)

import pdb
pdb.set_trace()
temp = [results,converged,target,simulated_tests]
print(converged,target)

with open('jd.p','wb') as f:
    pickle.dump(temp,f)

results,converged,target,simulated_tests = res
sim_tests = TSD(simulated_tests)
backend = str('ADEXP')
similar = sim_tests.optimize(model_parameters.MODEL_PARAMS[backend], NGEN=4, \
                             backend=backend, MU=4, protocol={'allen': True, 'elephant': False})


import pdb
pdb.set_trace()