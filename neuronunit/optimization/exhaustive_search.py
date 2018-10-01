
#from neuronunit.optimization import get_neab
#tests = get_neab.tests
import pdb

import multiprocessing
from collections import OrderedDict

from neuronunit.optimization.model_parameters import model_params
from neuronunit.optimization import data_transport_container
from neuronunit.optimization.optimization_management import nunit_evaluation, update_deap_pop
from neuronunit.optimization.optimization_management import update_dtc_pop
import numpy as np
from collections import OrderedDict


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
npart = multiprocessing.cpu_count()
import shelve

import os
from numba import jit

class WSListIndividual(list):
    """Individual consisting of list with weighted sum field"""
    def __init__(self, *args, **kwargs):
        """Constructor"""
        self.rheobase = None
        self.dtc = None

        super(WSListIndividual, self).__init__(*args, **kwargs)


@jit
def reduce_params(model_params,nparams):
    key_list = list(model_params.keys())
    reduced_key_list = key_list[0:nparams]
    subset = { k:model_params[k] for k in reduced_key_list }
    return subset

#@jit
def chunks(l, n):
    # For item i in a range that is a length of l,
    return [ l[:][i:i+n] for i in range(0, len(l), n) ]


#@jit
def build_chunk_grid(npoints, free_params, hold_constant=None, mp_in = None):
    grid_points, maps = create_grid(mp_in, npoints = npoints, free_params = provided_keys)

    temp = OrderedDict(grid_points[0]).keys()
    tds = list(temp)
    # old stable approach
    if len(grid_points[0].keys())>1:
        pops = [ WSListIndividual(g.values()) for g in grid_points ]
    else:
        pops = [ val for g in grid_points for val in g.values() ]
    return pops,tds

    # new approach, merited? Older works better
    '''

    pops = []
    for g in grid_points:
        pre_pop = list(g.values())
        pop = WSListIndividual(pre_pop)
        pops.append(pop)
    # divide population into chunks that reflect the number of CPUs.
    # don't want lists of lengths 1 that are awkward to iterate over.
    # so check if there would be a chunk of list length 1, and if so divide by a different numberself.
    # that is still dependant on CPU number.
    if len(pops) % npartitions != 1:
        pops_ = chunks(pops,npartitions)
    else:

        pops_ = chunks(pops,npartitions-2)
    if len(pops_) > 1:
        assert pops_[0] != pops_[1]

    return pops_, tds
    '''

@jit
def sample_points(iter_dict, npoints=3):
    replacement = {}
    for p in range(0,len(iter_dict)):
        k,v = iter_dict.popitem(last=False)
        if len(v) == 2:
            sample_points = list(np.linspace(v[0],v[1],npoints))
        else:
            v = np.array(v)
            sample_points = list(np.linspace(v.max(),v.min(),npoints))
        replacement[k] = sample_points
    return replacement
'''
@jit
def sample_points(iter_dict, npoints=2):
    replacement = {}
    for p in range(0,len(iter_dict)):
        k,v = iter_dict.popitem(last=False)
        v[0] = v[0]*1.0/3.0
        v[1] = v[1]*2.0/3.0
        #if len(v) == 2:
        #    sample_points = list(np.linspace(v[0],v[1],npoints))
        #else:
        #    v = np.array(v)
        sample_points = list(np.linspace(v.max(),v.min(),npoints))
        replacement[k] = sample_points
    return replacement
'''

@jit
def update_dtc_grid(item_of_iter_list):

    dtc = data_transport_container.DataTC()
    dtc.attrs = deepcopy(item_of_iter_list)
    dtc.scores = {}
    dtc.rheobase = None
    dtc.evaluated = False
    dtc.backend = 'NEURON'
    return dtc
@jit
def create_a_map(subset):
    maps = {}
    for k,v in subset.items():
        maps[k] = {}
        for ind,j in enumerate(subset[k]):
            maps[k][j] = ind
    return maps

#@jit
#def create_grid(mp_in=None,npoints=3,free_params=None,ga=None):
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

    # smaller is a dictionary thats not necessarily as big
    # as the grid defined in the model_params file. Its not necessarily
    # a smaller dictionary, if it is smaller it is reduced by reducing sampling
    # points.

    if type(mp_in) is type(None):
        from neuronunit.models.NeuroML2 import model_parameters as modelp
        mp_in = OrderedDict(modelp.model_params)

    pdb.set_trace()

    whole_p_set = {}
    sp = sample_points(copy.copy(mp_in), npoints=2)
    whole_p_set = OrderedDict(sp)

    print(type(free_params), 'free_params')
    if type(free_params) is type(dict):
        subset = OrderedDict( {k:whole_p_set[k] for k in list(free_params.keys())})

    elif len(free_params) == 1 or type(free_params) is type(str('')):
        subset = OrderedDict( {free_params: whole_p_set[free_params] } )

    else:
        subset = OrderedDict( {k:whole_p_set[k] for k in free_params})

    print('subset is wrong')
    pdb.set_trace()

    maps = create_a_map(subset)
    if type(ga) is not type(None):
        if npoints > 1:
            for k,v in subset.items():
                v[0] = v[0]*1.0/3.0
                v[1] = v[1]*2.0/3.0


    '''
def create_grid(mp_in=None,npoints=3,free_params=None,ga=None):
    subset = OrderedDict(free_params)
    if type(mp_in) is not type(None):
        for k,v in mp_in.items():
            if k in free_params.keys():
                subset[k] = ( np.min(free_params[k]),np.max(free_params[k]) )
            else:
                subset[k] = v
    # The function of maps is to map floating point sample spaces onto a  monochromataic matrix indicies.
    grid = list(ParameterGrid(subset))
    return grid

@jit
def tfg2i(x, y, z):
    '''
    translate_float_grid_to_index
    Takes x, y, z values as lists and returns a 2D numpy array
    '''
    dx = abs(np.sort(list(set(x)))[1] - np.sort(list(set(x)))[0])
    dy = abs(np.sort(list(set(y)))[1] - np.sort(list(set(y)))[0])
    i = ((x - min(x)) / dx).astype(int) # Longitudes
    j = ((y - max(y)) / dy).astype(int) # Latitudes
    grid = np.nan * np.empty((len(set(j)),len(set(i))))
    grid[j, i] = z # if using latitude and longitude (for WGS/West)
    return grid

def tfc2i(x, y, z,err):
    '''
    translate_float_cube_to_index

    Takes x, y, z values as lists and returns a 2D numpy array
    '''
    dx = abs(np.sort(list(set(x)))[1] - np.sort(list(set(x)))[0])
    dy = abs(np.sort(list(set(y)))[1] - np.sort(list(set(y)))[0])
    dz = abs(np.sort(list(set(z)))[1] - np.sort(list(set(y)))[0])

    i = ((x - min(x)) / dx).astype(int) # Longitudes
    j = ((y - max(y)) / dy).astype(int) # Latitudes
    k = ((z - max(z)) / dz).astype(int) # Latitudes

    grid = np.nan * np.empty((len(set(i)),len(set(j)), len(set(k)) ))

    grid[i,j,k] = err # if using latitude and longitude (for WGS/West)
    return
@jit
def add_constant(hold_constant,consumable_,td):

    hc = list(hold_constant.values())
    for c in consumable_:
        for i in c:
            for h in hc:
                print(type(i),i)
                i.append(h)
    for k in hold_constant.keys():
        td.append(k)
    return td, hc



@jit
def transdict(dictionaries):
    from collections import OrderedDict
    mps = OrderedDict()
    sk = sorted(list(dictionaries.keys()))
    for k in sk:
        mps[k] = dictionaries[k]
    tl = [ k for k in mps.keys() ]
    return mps, tl


def run_rick_grid(rick_grid, tests,td):
    consumable = iter(rick_grid)
    grid_results = []
    results = update_deap_pop(consumable, tests, td)
    #import pdb
    #pdb.set_trace()

    if type(results) is not None:
        grid_results.extend(results)
    return grid_results

def run_grid(npoints, tests, provided_keys = None, hold_constant = None, mp_in=None):
    consumable_ ,td = build_chunk_grid(npoints,provided_keys,mp_in=mp_in)
    cnt = 0
    grid_results = []
    if type(hold_constant) is not type(None):
        td, hc = add_constant(hold_constant,consumable_,td)
        #assert len(td) == len(provided_keys) + len(hold_constant)
    consumable = iter(consumable_)
    use_cache = None
    s = None

    for sub_pop in consumable:
        results = update_deap_pop(sub_pop, tests, td)
        if type(results) is not None:
            grid_results.extend(results)
        #if type(hold_constant) is not type(None):
        #    assert len(grid_results[0]) == len(provided_keys) + len(hold_constant)

        if type(use_cache) is not type(None):
            if type(s) is not type(None):
                s['consumable'] = consumable
                s['cnt'] = cnt
                s['grid_results'] = grid_results
                s['sub_pop'] = sub_pop
        cnt += 1
        print('done_block_of_N_cells: ',cnt)
    if type(s) is not type(None):
        s.close()
    return grid_results
