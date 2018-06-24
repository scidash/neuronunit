

import multiprocessing
npartitions = multiprocessing.cpu_count()

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
#from bluepyopt.deapext import optimisations
#from bluepyopt.deapext import algorithms

from neuronunit.optimization.optimization_management import write_opt_to_nml

from neuronunit.optimization import exhaustive_search
from neuronunit.optimization import optimization_management

import dask.bag as db
from neuronunit.optimization import get_neab
from sklearn.grid_search import ParameterGrid
import scipy
import pdb

import numpy as np
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math as math
from pylab import rcParams


class WSListIndividual(list):
    """Individual consisting of list with weighted sum field"""
    def __init__(self, *args, **kwargs):
        """Constructor"""
        self.rheobase = None
        self.dtc = None

        super(WSListIndividual, self).__init__(*args, **kwargs)


def min_max(pop):
    garanked = [ (r.dtc.attrs , sum(r.dtc.scores.values()), r.dtc) for r in pop ]
    garanked = sorted(garanked, key=lambda w: w[1])
    miniga = garanked[0]
    maxiga = garanked[-1]
    return miniga, maxiga

def reduce_params(model_params,nparams):
    key_list = list(model_params.keys())
    reduced_key_list = key_list[0:nparams]
    subset = { k:model_params[k] for k in reduced_key_list }
    return subset


def chunks(l, n):
    # For item i in a range that is a length of l,
    ch = []
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        ch.append(l[:][i:i+n])
    return ch

def build_chunk_grid(npoints,nparams):
    grid_points = exhaustive_search.create_grid(npoints = npoints,nparams = nparams)
    tds = [ list(g.keys()) for g in grid_points ]
    td = tds[0]

    pops = []
    for g in grid_points:
        pre_pop = list(g.values())
        pops.extend(pre_pop)
        pop = WSListIndividual(pops)

    # divide population into chunks that reflect the number of CPUs.
    if len(pops) % npartitions != 1:
        pops_ = chunks(pops,npartitions)
    else:
        pops_ = chunks(pops,npartitions-2)
    try:
        assert pops_[0] != pops_[1]
    except:
        import pdb; pdb.set_trace()
    return pops_, td



def run_ga(model_params,nparams):
    grid_points = exhaustive_search.create_grid(npoints = 1,nparams = nparams)
    td = list(grid_points[0].keys())
    subset = reduce_params(model_params,nparams)
    DO = DEAPOptimisation(error_criterion = electro_tests[0][0], selection = str('selNSGA'), provided_dict = subset, elite_size = 3)
    MU = int(np.floor(npoints/2.0))
    max_ngen = int(np.floor(nparams/2.0))
    # make initial samples uniform on grid points
    # 3
    # 3 minimum number of points to define this hypercube.
    # Create a lattice, using the exhaustive

    # make a movie,
    assert (MU * max_ngen) < (npoints * nparams)
    ga_out = DO.run(offspring_size = MU, max_ngen = 6, cp_frequency=1,cp_filename=str('regular.p'))
    # pop, hof_py, pf, log, history, td_py, gen_vs_hof = ga_out
    with open('all_ga_cell.p','wb') as f:
        pickle.dump(ga_out,f)
    return ga_out

def float_to_list(sub_pop):
    if type(sub_pop) is not type(list()):
        if not hasattr(sub_pop,'rheobase'):
            sp = WSListIndividual()
            sp.append(sub_pop)
            sub_pop = sp
    return sub_pop


reports = {}
npoints = 10

def param_distance(dtc_ga_attrs,dtc_grid_attrs,td):
    distances = {}
    mp = reduce_params(model_params,nparams)
    for k,v in dtc_ga_attrs.items():
        dimension_length = np.max(mp[k]) - np.min(mp[k])
        solution_distance_in_1D = np.abs(float(dtc_grid_attrs[k]))-np.abs(float(v))
        try:
            relative_distance = np.abs(dimension_length/solution_distance_in_1D)
        except:
            relative_distance = None
        distances.get(k, relative_distance)

        distances[k] = relative_distance
        print('the difference between brute force candidates model parameters and the GA\'s model parameters:')
        print(float(dtc_grid_attrs[k])-float(v),dtc_grid_attrs[k],v,k)
        print('the relative distance scaled by the length of the parameter dimension of interest:')
        print(relative_distance)
    return distances

def error_domination(dtc_ga,dtc_grid):
    distances = {}
    errors_ga = list(dtc_ga.scores.values())
    print(errors_ga)
    me_ga = np.mean(errors_ga)
    std_ga = np.std(errors_ga)

    errors_grid = list(dtc_grid.scores.values())
    print(errors_grid)
    me_grid = np.mean(errors_grid)
    std_grid = np.std(errors_grid)

    dom_grid = False
    dom_ga = False

    for e in errors_ga:
        if e <= me_ga + std_ga:
            dom_ga = True

    for e in errors_grid:
        if e <= me_grid + std_grid:
            dom_grid= True

    return dom_grid, dom_ga

def run_grid(npoints,nparams):
    consumable_ ,td = build_chunk_grid(npoints,nparams)

    #consumble = [(sub_pop, test, observation ) for test, _ in electro_tests for sub_pop in pops_ ]
    # Create a consumble iterator, that facilitates memory friendly lazy evaluation.
    test, observation = electro_tests[0]
    try:
        assert 1==2

        with open('grid_cell_results'+str(nparams)+str('.p'),'rb') as f:
            results  = pickle.load(f)
        with open('iterator_state'+str(nparams)+str('.p'),'rb') as f:
            sub_pop, test, observation, cnt = pickle.load(f)
            # consumble_ = [(sub_pop, test, observation ) for test, _ in electro_tests for sub_pop in pops_ ][cnt]
            consumable_ = consumable_[cnt]
            if len(consumable_) < len(consumable) and len(consumable_) !=0 :
                consumbale = iter(consumbale_)
    except:
        consumable = iter(consumable_)
    cnt = 0
    grid_results = []

    for sub_pop in consumable:
        print('{0}, out of {1}'.format(cnt,len(sub_pop)))
        grid_results.extend(optimization_management.update_exhaust_pop(sub_pop, test, td))
        with open('grid_cell_results'+str(nparams)+str('.p'),'wb') as f:
            pickle.dump(grid_results,f)
        with open('iterator_state'+str(nparams)+str('.p'),'wb') as f:
            pickle.dump([sub_pop, test, observation, cnt],f)
        cnt += 1
        print('done_block_of_eight_cells: ',cnt)
    return grid_results
#import matplotlib.plot as mpl




for nparams in range(1,3):
    pass

if True:
    nparams = 2
    grid_results = run_grid(npoints,nparams)

    ga_out = run_ga(model_params,nparams)

    plt.clf()
    plt.scatter(grid_results,[ sum(g.dtc.scores.values()) for g in grid_results ] )
    plt.savefig(str(nparams)+str('error_cross_section')+str('.png'))

    miniga = min_max(ga_out[0])[0][1]

    plt.clf()
    plt.scatter(ga_out[0],[ sum(g.dtc.scores.values()) for g in ga_out[0] ] )
    plt.scatter(grid_results,[ sum(g.dtc.scores.values()) for g in grid_results ] )
    plt.scatter(miniga,int(len(grid_results)/2))
    plt.savefig(str(nparams)+str('ga_grid_error_cross_section')+str('.png'))

    plt.clf()
    plt.scatter(ga_out[0],[ sum(g.dtc.score.values()) for g in ga_out[0] ] )
    plt.scatter(grid_results,[ sum(g.dtc.score.values()) for g in grid_results ] )
    plt.savefig(str(nparams)+str('obs_prediction_agreement')+str('.png'))



    plt.clf()
    for j in [ list(g.dtc.scores.values()) for g in grid_results ]:
        plt.scatter([i for i in range(0,len(j))] ,j)
    plt.savefig(str(nparams)+str('error_cross_section_components')+str('.png'))



    mini = min_max(grid_results)[0][1]
    maxi = min_max(grid_results)[1][1]
    quantize_distance = list(np.linspace(mini,maxi,21))
    worked = bool(miniga < quantize_distance[2])
    print('Report: ')
    print('did it work? {0}'.format(worked))
    reports[nparams] = {}
    reports[nparams]['success'] = bool(miniga < quantize_distance[2])
    dtc_ga = min_max(ga_out[0])[0][0]
    attrs_grid = min_max(grid_results)[0][0]
    attrs_ga = min_max(ga_out[0])[0][0]

    grid_points = exhaustive_search.create_grid(npoints = 1,nparams = nparams)#td = list(grid_points[0].keys())
    td = list(grid_points[0].keys())

    reports[nparams]['p_dist'] = param_distance(attrs_ga,attrs_grid,td)
    dtc_grid = dtc_ga = min_max(ga_out[0])[0][2]
    dom_grid, dom_ga = error_domination(dtc_ga,dtc_grid)

    # Was there vindicating domination in grid search but not GA?
    if dom_grid == True and dom_ga == False:
        reports[nparams]['vind_domination'] = True
    elif dom_grid == False and dom_ga == False:
        reports[nparams]['vind_domination'] = True
    # Was there incriminating domination in GA but not the grid, or in GA and Grid
    elif dom_grid == True and dom_ga == True:
        reports[nparams]['inc_domination'] = False
    elif dom_grid == False and dom_ga == True:
        reports[nparams]['inc_domination'] = False
