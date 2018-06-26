
#from neuronunit.optimization import get_neab
#tests = get_neab.tests
import pdb

import multiprocessing
npart = multiprocessing.cpu_count()

from neuronunit.optimization.model_parameters import model_params
from neuronunit.optimization import model_parameters as modelp
from neuronunit.optimization import data_transport_container
from neuronunit.optimization.optimization_management import nunit_evaluation, update_deap_pop
from neuronunit.optimization.optimization_management import update_dtc_pop
import numpy as np


import copy
from copy import deepcopy
import math

import dask.bag as db
from sklearn.grid_search import ParameterGrid
import scipy

import pickle
import os
import numpy as np
npartitions = multiprocessing.cpu_count()
import os


class WSListIndividual(list):
    """Individual consisting of list with weighted sum field"""
    def __init__(self, *args, **kwargs):
        """Constructor"""
        self.rheobase = None
        self.dtc = None

        super(WSListIndividual, self).__init__(*args, **kwargs)



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
    grid_points = create_grid(npoints = npoints,nparams = nparams)
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
        pdb.set_trace()
    return pops_, td


def sample_points(iter_dict, npoints=3):
    replacement = {}
    for k,v in iter_dict.items():
        sample_points = list(np.linspace(v.max(),v.min(),npoints))
        replacement[k] = sample_points
    return replacement


def create_refined_grid(best_point,point1,point2):
    '''
    Can be used for creating a second pass fine grained grid
    '''

    # This function reports on the deltas brute force obtained versus the GA found attributes.
    #from neuronunit.optimization import model_parameters as modelp
    #mp = modelp.model_params
    new_search_interval = {}
    for k,v in point1.attrs.items():
        higher =  max(float(point1.attrs[k]),float(v), point2.attrs[k])
        lower = min(float(point1.attrs[k]),float(v), point2.attrs[k])
        temp = list(np.linspace(lower,higher,10))
        new_search_interval[k] = temp[1:-2] # take only the middle two points
        # discard edge points, as they are already well searched/represented.
    grid = list(ParameterGrid(new_search_interval))
    return grid

def update_dtc_grid(item_of_iter_list):

    dtc = data_transport_container.DataTC()
    dtc.attrs = deepcopy(item_of_iter_list)
    dtc.scores = {}
    dtc.rheobase = None
    dtc.evaluated = False
    dtc.backend = 'NEURON'
    return dtc

def create_grid(npoints=3,nparams=7,provided_keys=None):
    '''
    Description, create a grid of evenly spaced samples in model parameters to search over.
    Inputs: npoints, type: Integer: number of sample points per parameter
    nparams, type: Integer: number of parameters to use, conflicts, with next argument.
    nparams, iterates through a list of parameters, and assigns the nparams to use via stupid counting.
    provided keys: explicitly define state the model parameters that are used to create the grid of samples, by
    keying into an existing of parameters.

    This method needs the user of the method to declare a dictionary of model parameters in a path:
    neuronunit.optimization.model_parameters.

    Miscallenous, once grid created by this function
    has been evaluated using neuronunit it can be used for informing a more refined second pass fine grained grid
    '''


    mp = modelp.model_params
    # smaller is a dictionary thats not necessarily as big
    # as the grid defined in the model_params file. Its not necessarily
    # a smaller dictionary, if it is smaller it is reduced by reducing sampling
    # points.
    smaller = {}
    smaller = sample_points(mp, npoints=npoints)

    if type(provided_keys) is type(None):

        key_list = list(smaller.keys())
        reduced_key_list = key_list[0:nparams]
    else:
        reduced_key_list = list(provided_keys)

    # subset is reduced, by reducing parameter keys.
    subset = { k:smaller[k] for k in reduced_key_list }
    grid = list(ParameterGrid(subset))
    return grid

def run_grid(nparams,npoints,test):

    consumable_ ,td = build_chunk_grid(npoints,nparams)

    #consumble = [(sub_pop, test, observation ) for test, _ in electro_tests for sub_pop in pops_ ]
    # Create a consumble iterator, that facilitates memory friendly lazy evaluation.
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
        for s in sub_pop:
            if math.isnan(s):
                import pdb; pdb.set_trace()

        print('{0}, out of {1}'.format(cnt,len(sub_pop)))
        grid_results.extend(update_deap_pop(sub_pop, test, td))
        with open('grid_cell_results'+str(nparams)+str('.p'),'wb') as f:
            pickle.dump(grid_results,f)
        with open('iterator_state'+str(nparams)+str('.p'),'wb') as f:
            pickle.dump([sub_pop, test, cnt],f)
        cnt += 1
        print('done_block_of_N_cells: ',cnt)
    return grid_results
