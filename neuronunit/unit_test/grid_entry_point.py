#USE_CACHED_GS = False

import pickle
import os

from neuronunit.optimization import exhaustive_search
from neuronunit.optimization import get_neab
from neuronunit.optimization import optimization_management

import dask.bag as db
from neuronunit.optimization import get_neab
from sklearn.grid_search import ParameterGrid

electro_path = str(os.getcwd())+'/pipe_tests.p'
assert os.path.isfile(electro_path) == True
with open(electro_path,'rb') as f:
    electro_tests = pickle.load(f)

electro_tests = get_neab.replace_zero_std(electro_tests)
electro_tests = get_neab.substitute_parallel_for_serial(electro_tests)
#electro_tests = []
purkinje = { 'nlex_id':'sao471801888'}#'NLXWIKI:sao471801888'} # purkinje
fi_basket = {'nlex_id':'100201'}
pvis_cortex = {'nlex_id':'nifext_50'} # Layer V pyramidal cell
olf_mitral = { 'nlex_id':'nifext_120'}
ca1_pyr = { 'nlex_id':'830368389'}

pipe = [ fi_basket, pvis_cortex, olf_mitral, ca1_pyr, purkinje ]


class WSListIndividual(list):
    """Individual consisting of list with weighted sum field"""
    def __init__(self, *args, **kwargs):
        """Constructor"""
        self.rheobase = None
        super(WSListIndividual, self).__init__(*args, **kwargs)


nparams = 10

grid_points = exhaustive_search.create_grid(npoints = 5,nparams = nparams)
tds = [ list(g.keys()) for g in grid_points ]
td = tds[0]

ds = [(i,value) for i,value in enumerate(grid_points)]
# sample the grid at different levels of resolution.
ds10_ = scipy.signal.resample(ds,10))
ds5_ = scipy.signal.resample(ds,5))
ds3_ = scipy.signal.resample(ds,3))
ds1_ = ds

ds10 = ds10_
ds5 = ds5_[list(set(ds10_,ds5_))]
ds3 = ds3_[list(set(ds3_,ds5_,ds10_))]
ds1 = ds1_[list(set(ds3_,ds5_,ds1_,ds10_))]

assert len(ds5_) > ds5
assert len(ds3_) > ds3
assert len(ds3_) > ds3
assert len(ds1_) > ds1

pops = []
for grid in [ds10,ds5,ds3,ds1]:
    pre_pop = [ list(g.values()) for g in grid ]
    pop = [ WSListIndividual(p) for p in pre_pop ]
    pops.extend(pop)


cnt = 0

##
# Crete a
# Consumble iterator.
##

flat_iter = iter((p, test, observation ) for p, test, observation in electro_tests for p in pops)

for p, test, observation in flat_iter:
    pipes[str(pipe[cnt])] = optimization_management.update_exhaust_pop(p, test, td)
    with open('grid_cell'+str(pipe[cnt])+'.p','wb') as f:
        pickle.dump(pop,f)
    with open('iterator_state.p','wb') as f:
        pickle.dump([p, test, observation, flat_iter],f)
    cnt += 1
    print('done cell: ',cnt)
print('done all')
with open('all_grid_cell'+str(pipe[cnt])+'.p','wb') as f:
    pickle.dump(pipes,f)
