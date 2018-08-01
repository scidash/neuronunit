import pickle


from neuronunit.optimization import get_neab
import copy
import os
from neuronunit.optimization.optimization_management import run_ga


electro_path = str(os.getcwd())+'/pipe_tests.p'
print(os.getcwd())
assert os.path.isfile(electro_path) == True
with open(electro_path,'rb') as f:
    electro_tests = pickle.load(f)

electro_tests = get_neab.replace_zero_std(electro_tests)
electro_tests = get_neab.substitute_parallel_for_serial(electro_tests)
test, observation = electro_tests[0]


import matplotlib.pyplot as plt

from neuronunit.optimization import get_neab
import copy
import os
import pickle
electro_path = str(os.getcwd())+'/pipe_tests.p'
from neuronunit import plottools
import numpy as np
ax = None
from neuronunit.optimization import exhaustive_search as es

plot_surface = plottools.plot_surface
scatter_surface = plottools.plot_surface

with open(electro_path,'rb') as f:
    electro_tests = pickle.load(f)
from matplotlib.colors import LogNorm
from neuronunit.optimization.exhaustive_search import run_grid, reduce_params, create_grid, mock_grid
from neuronunit.optimization import model_parameters as modelp
mp = modelp.model_params

opt_keys = ['C','b']


electro_tests = get_neab.replace_zero_std(electro_tests)
electro_tests = get_neab.substitute_parallel_for_serial(electro_tests)
test, observation = electro_tests[0]

npoints = 6
tests = copy.copy(electro_tests[0][0][0:2])

MU = 6
nparams = len(opt_keys)
package = run_ga(mp,nparams,MU,tests,provided_keys = opt_keys)#, use_cache = True, cache_name='simple')
pop = package[0]
print(pop[0].dtc.attrs.items())
history = package[4]
gen_vs_pop =  package[6]
hof = package[1]

grid_results = {}

plt.clf()
fig,ax = plt.subplots(3,3,figsize=(10,10))

visited = []
one_d = {}
two_d = {}
cnt = 0

from neuronunit.optimization import model_parameters as modelp
mp = modelp.model_params
for i,ki in enumerate(pop[0].dtc.attrs.keys()):
    for j,kj in enumerate(pop[0].dtc.attrs.keys()):

        free_param = set([ki,kj]) # construct a small-set out of the indexed keys 2.
        fp = { k:None for k in list(free_param) }

        bs = set(attrs_list) # construct a full set out of all of the keys available 3.

        diff = bs.difference(free_param) # diff is simply the key that is not indexed.
        visited.append(diff)

        # BD is the dictionary of parameters to be held constant
        # if the plot is 1D then two parameters should be held constant.

        bd =  {}
        for d in diff:
            bd[d] = hof[0].dtc.attrs[d]

        for k,v in hc.items():
            if str(ki) in k and str(kj) in k:
                key = k

        if ki == kj:
            #pass
            assert len(fp) == len(bd) - 1
            assert len(bd) == len(fp) + 1
            gr = run_grid(10,tests,provided_keys = fp ,hold_constant = bd)#, use_cache = True, cache_name='complex')
            one_d[str(fp.keys())] = gr
            ax[i,j].plt()

        if ki != kj:
            reduced = { k:mp[k] for k in fp.keys() }
            gr = None
            assert len(reduced) == 2
            assert len(bd) == 1
            gr = run_grid(6,tests,provided_keys = reduced ,hold_constant = bd)
            if str('kj') not in two_d[ki].keys():
                two_d[ki] = {}
                two_d[ki][kj] = gr
            else:
                two_d[ki][kj] = gr
            z = np.array([ np.sum(list(p.dtc.scores.values())) for p in gr[0] ])

            x = set([ p.dtc.attrs[ki] for p in gr[0] ])
            y = set([ p.dtc.attrs[kj] for p in gr[0] ])

            N = int(np.sqrt(len(z)))
            z = np.array(z).reshape((N, N))
            print(len(x),len(y),len(z), np.shape(z))
            ax[i,j].pcolormesh(x, y, z)

with open('surfaces.p','wb') as f:
    pickle.dump(f,[two_d,one_d,ax])
