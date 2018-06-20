
nparams = 2
MU = 7

import multiprocessing
multiprocessing.cpu_count()

import pickle
import os
from neuronunit.optimization import get_neab

electro_path = str(os.getcwd())+'/pipe_tests.p'
assert os.path.isfile(electro_path) == True
with open(electro_path,'rb') as f:
    electro_tests = pickle.load(f)

electro_tests = get_neab.replace_zero_std(electro_tests)
electro_tests = get_neab.substitute_parallel_for_serial(electro_tests)

from neuronunit.optimization import get_neab
from neuronunit.optimization.model_parameters import model_params
from bluepyopt.deapext.optimisations import DEAPOptimisation
from bluepyopt.deapext import optimisations
from bluepyopt.deapext import algorithms

from neuronunit.optimization.optimization_management import write_opt_to_nml
from neuronunit.optimization import optimization_management
from neuronunit.optimization import optimization_management as om
key_list = list(model_params.keys())
reduced_key_list = key_list[0:nparams]
subset = { k:model_params[k] for k in reduced_key_list }
DO = DEAPOptimisation(error_criterion = electro_tests[0][0], selection = str('selNSGA'), provided_dict = subset, elite_size = 3)
package = DO.run(offspring_size = MU, max_ngen = 6, cp_frequency=1,cp_filename=str('regular.p'))
pop, hof_py, pf, log, history, td_py, gen_vs_hof = package

with open('all_ga_cell.p','wb') as f:
    pickle.dump(package,f)


from neuronunit.optimization import exhaustive_search
from neuronunit.optimization import optimization_management

import dask.bag as db
from neuronunit.optimization import get_neab
from sklearn.grid_search import ParameterGrid
import scipy
import pdb


class WSListIndividual(list):
    """Individual consisting of list with weighted sum field"""
    def __init__(self, *args, **kwargs):
        """Constructor"""
        self.rheobase = None
        super(WSListIndividual, self).__init__(*args, **kwargs)



grid_points = exhaustive_search.create_grid(npoints = 10,nparams = nparams)
tds = [ list(g.keys()) for g in grid_points ]
td = tds[0]


pops = []
for g in grid_points:
    pre_pop = list(g.values())
    pop = [ WSListIndividual(pre_pop) ]
    pops.extend(pop)


def chunks(l, n):
    # For item i in a range that is a length of l,
    ch = []
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        ch.append(l[:][i:i+n])
    return ch

npartitions = multiprocessing.cpu_count()
# divide population into chunks that reflect the number of CPUs.
pops_ =  chunks(pops,npartitions)

##
# Create a
# Consumble iterator.
# That promotes memory friendly lazy evaluation.
##

consumble = [(sub_pop, test, observation ) for test, observation in electro_tests for sub_pop in pops_ ]
try:
    with open('grid_cell_results.p','rb') as f:
        results  = pickle.load(f)
    with open('iterator_state.p','rb') as f:
        consumble_ = [(sub_pop, test, observation ) for test, observation in electro_tests for sub_pop in pops_ ][cnt]
        sub_pop, test, observation, cnt = pickle.load(f)
        if len(consumble_) < len(consumble) and len(consumble_) !=0 :
            consumble = iter(consumble_)
except:
    consumble = iter(consumble)

cnt = 0
pipes = {}
results = []

for sub_pop, test, observation in consumble:
    if len(sub_pop) == 1:
        sub_pop.extend(sub_pop)
    print('{0}, out of {1}'.format(cnt,len(pops_)))
    results.append(optimization_management.update_exhaust_pop(sub_pop, test, td))
    with open('grid_cell_results.p','wb') as f:
        pickle.dump(results,f)
    with open('iterator_state.p','wb') as f:
        pickle.dump([sub_pop, test, observation, cnt],f)
    cnt += 1
    print('done cell: ',cnt)
print('done all')
