##
# Assumption that this file was executed after first executing the bash: ipcluster start -n 8 --profile=default &
##
import matplotlib # Its not that this file is responsible for doing plotting, but it calls many modules that are, such that it needs to pre-empt
# setting of an appropriate backend.
matplotlib.use('agg')
import os
from numpy import random
import numpy as np

import ipyparallel as ipp
rc = ipp.Client(profile='default')
#rc.Client.become_dask()

rc[:].use_cloudpickle()
dview = rc[:]

import dask.bag as db
import pandas as pd

from ipyparallel import depend, require, dependent
# Import get_neab has to happen exactly here. It has to be called only on
# controller (rank0, it has)
from neuronunit import tests
from neuronunit.optimization import get_neab

def get_tests():
    '''
    # Not compatible with cloud pickle
    # Pull tests
    '''
    tests = []
    tests.append(dview.pull('InputResistanceTest',targets=0).get())
    tests.append(dview.pull('TimeConstantTest',targets=0).get())
    tests.append(dview.pull('CapacitanceTest',targets=0).get())
    tests.append(dview.pull('RestingPotentialTest',targets=0).get())
    tests.append(dview.pull('InjectedCurrentAPWidthTest',targets=0).get())
    tests.append(dview.pull('InjectedCurrentAPAmplitudeTest',targets=0).get())
    tests.append(dview.pull('InjectedCurrentAPThresholdTest',targets=0).get())
    return tests

def dtc_to_rheo(dtc):
    from neuronunit.models.reduced import ReducedModel
    from neuronunit.optimization import get_neab
    dtc.model_path = get_neab.LEMS_MODEL_PATH
    dtc.LEMS_MODEL_PATH = get_neab.LEMS_MODEL_PATH

    #from neuronunit.optimization import evaluate_as_module
    model = ReducedModel(dtc.LEMS_MODEL_PATH,name=str('vanilla'),backend='NEURON')

    model.set_attrs(dtc.attrs)
    #if not hasattr(model,'rheobase'):
    model.rheobase = None
    #dtc.cell_name = model._backend.cell_name
    #dtc.current_src_name = model._backend.current_src_name
    dtc.scores = None
    dtc.scores = {}
    dtc.differences = None
    dtc.differences = {}
    get_neab.tests[0].dview = dview
    score = get_neab.tests[0].judge(model,stop_on_error = False, deep_error = True)
    observation = score.observation
    dtc.rheobase = score.prediction
    dtc.scores[str(get_neab.tests[0])] = score.sort_key
    print(dtc.scores)
    return dtc

def dtc_to_plotting(dtc):
    dtc.t = None
    from neuronunit.models.reduced import ReducedModel
    from neuronunit.optimization import get_neab
    from neuronunit.optimization import evaluate_as_module
    model = ReducedModel(get_neab.LEMS_MODEL_PATH,name=str('vanilla'),backend='NEURON')
    model.set_attrs(dtc.attrs)
    model.rheobase = dtc.rheobase['value']
    score = get_neab.tests[-1].judge(model,stop_on_error = False, deep_error = True)
    dtc.vm = list(model.results['vm'])
    dtc.t = list(model.results['time'])
    return dtc

def nunit_evaluation(dtc):
    '''
    NU
    was in evaluate_as_module
    '''
    from neuronunit.models.reduced import ReducedModel
    from neuronunit.optimization import get_neab
    tests = get_neab.tests
    model = ReducedModel(get_neab.LEMS_MODEL_PATH,name=str('vanilla'),backend=('NEURON',{'DTC':dtc}))
    model.set_attrs(dtc.attrs)
    tests[0].prediction = dtc.rheobase
    get_neab.tests[0].dview = dview
    model.rheobase = dtc.rheobase['value']
    dtc = pre_format(dtc)
    for k,t in enumerate(tests[1:-1]):
        t.params = dtc.vtest[k]
        score = t.judge(model,stop_on_error = False, deep_error = False)
        assert bool(model.get_spike_count() == 1 or model.get_spike_count() == 0)
        dtc.scores[str(t)] = 1 - score.sort_key #.sort_key
        print(dtc.scores)#, 'get\'s here')
    return dtc


def evaluate(dtc):

    from neuronunit.optimization import get_neab
    import numpy as np
    fitness = [ 1.0 for i in range(0,7) ]

    for k,t in enumerate(dtc.scores.keys()):
        fitness[k] = dtc.scores[str(t)]#.sort_key
    #fitness[7] = np.sum(list(fitness[0:6]))
    print(fitness)
    return fitness[0],fitness[1],\
           fitness[2],fitness[3],\
           fitness[4],fitness[5],\
           fitness[6],#fitness[7]



def get_trans_list(param_dict):
    trans_list = []
    for i,k in enumerate(list(param_dict.keys())):
        trans_list.append(k)
    return trans_list

def pre_format(dtc):
    '''
    pre format the current injection dictionary based on pre computed
    rheobase values of current injection.
    This is much like the hooked method from the old get neab file.
    '''
    import quantities as pq
    import copy
    dtc.vtest = None
    dtc.vtest = {}
    from neuronunit.optimization import get_neab
    tests = get_neab.tests
    for k,v in enumerate(tests):
        dtc.vtest[k] = {}
        dtc.vtest[k]['injected_square_current'] = {}
    for k,v in enumerate(tests):
        if k == 1 or k == 2 or k == 3:
            # Negative square pulse current.
            dtc.vtest[k]['injected_square_current']['duration'] = 100 * pq.ms
            dtc.vtest[k]['injected_square_current']['amplitude'] = -10 *pq.pA
            dtc.vtest[k]['injected_square_current']['delay'] = 30 * pq.ms

        if k == 0 or k == 4 or k == 5 or k == 6 or k == 7:
            # Threshold current.
            dtc.vtest[k]['injected_square_current']['duration'] = 1000 * pq.ms
            dtc.vtest[k]['injected_square_current']['amplitude'] = dtc.rheobase['value']
            dtc.vtest[k]['injected_square_current']['delay'] = 250 * pq.ms # + 150
    return dtc



def update_dtc_pop(pop, td):

    '''
    inputs a population of genes/alleles, the population size MU, and an optional argument of a rheobase value guess
    outputs a population of genes/alleles, a population of individual object shells, ie a pickleable container for gene attributes.
    Rationale, not every gene value will result in a model for which rheobase is found, in which case that gene is discarded, however to
    compensate for losses in gene population size, more gene samples must be tested for a successful return from a rheobase search.
    If the tests return are successful these new sampled individuals are appended to the population, and then their attributes are mapped onto
    corresponding virtual model objects.
    '''

    import copy
    import numpy as np
    from deap import base
    toolbox = base.Toolbox()
    Individual = ipp.Reference('Individual')
    pop = [toolbox.clone(i) for i in pop ]
    
    #sl = [ (i, val) for i, val in enumerate(t_analysis.searchList) ]
    #df = pd.DataFrame(data=obj_arrs)
    #df



    def transform(ind):
        import dask.bag as db
        from neuronunit.optimization.data_transport_container import DataTC
        dtc = DataTC()
        dtc.attrs = {}
        for i,j in enumerate(ind):
            dtc.attrs[str(td[i])] = j
        dtc.evaluated = False
        return dtc
    if len(pop) > 1:
        b = db.from_sequence(pop, npartitions=8)
        dtcpop = list(db.map(transform,b).compute())
    
        #dtcpop = list(dview.map_sync(transform, pop))
    else:
        # In this case pop is not really a population but an individual
        # but parsimony of naming variables
        # suggests not to change the variable name to reflect this.
        dtcpop = list(transform(pop))
    return dtcpop


def update_pop(pop,td):
    import dask.bag as db

    # change name to update deap pop
    '''
    Inputs a population of genes (pop).
    Returned neuronunit scored DTCs (dtcpop).
    this method converts a population of genes to a population of Data Transport Containers,
    Which act as communicatable data types for storing model attributes.
    Rheobase values are found on the DTCs
    DTCs for which a rheobase value of x (pA)<=0 are filtered out
    DTCs are then scored by neuronunit, using neuronunit models that act in place.
    '''

    from neuronunit.optimization import model_parameters as modelp
    # given the wrong attributes, and they don't have rheobase values.
    dtcpop = list(update_dtc_pop(pop, td))
    dtcpop = list(map(dtc_to_rheo,dtcpop))
    dtcpop = list(filter(lambda dtc: dtc.rheobase['value'] > 0.0 , dtcpop))
    while len(dtcpop) < len(pop):
        dtcpop.append(dtcpop[0])
    dtcpop = list(map(pre_format,dtcpop))

    b = db.from_sequence(dtcpop, npartitions=8)
    dtcpop = list(db.map(nunit_evaluation,b).compute())


    dtcpop = list(filter(lambda dtc: type(dtc.scores['RheobaseTestP']) is not type(None), dtcpop))
    dtcpop = list(filter(lambda dtc: not type(None) in (list(dtc.scores.values())), dtcpop))
    ##
    # get rid of transport containers for genes that are responsible for returning None scores
    ## 
    dtc_temp = []
    for dtc in dtcpop:
        temp = list(dtc.scores.values())
        if None not in temp:
            dtc_temp.append(dtc)
    dtcpop = dtc_temp
    ##
    while len(dtcpop) < len(pop):
        dtcpop.append(dtcpop[0])

    for i,d in enumerate(dtcpop):
        pop[i].rheobase = d.rheobase
    return_package = zip(dtcpop, pop)
    return return_package


def create_subset(nparams=10, provided_keys=None):
    from neuronunit.optimization import model_parameters as modelp
    import numpy as np
    mp = modelp.model_params

    key_list = list(mp.keys())

    if type(provided_keys) is type(None):
        key_list = list(mp.keys())
        reduced_key_list = key_list[0:nparams]
    else:
        reduced_key_list = provided_keys

    subset = { k:mp[k] for k in reduced_key_list }
    return subset
