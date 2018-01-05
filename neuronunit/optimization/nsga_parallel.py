##
# Assumption that this file was executed after first executing the bash: ipcluster start -n 8 --profile=default &
##
import matplotlib # Its not that this file is responsible for doing plotting, but it calls many modules that are, such that it needs to pre-empt
# setting of an appropriate backend.
matplotlib.use('agg')
import os
from numpy import random
import numpy as np

#import sys
#os.system('ipcluster start -n 8 --profile=default & sleep 15 ; python stdout_worker.py &')

import ipyparallel as ipp
rc = ipp.Client(profile='default')
#rc.Client.become_dask()

rc[:].use_cloudpickle()
dview = rc[:]




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

    from neuronunit.optimization import evaluate_as_module
    model = ReducedModel(dtc.LEMS_MODEL_PATH,name=str('vanilla'),backend='NEURON')

    model.set_attrs(dtc.attrs)
    model.rheobase = None
    dtc.scores = None
    dtc.scores = {}
    dtc.differences = None
    dtc.differences = {}
    get_neab.tests[0].dview = dview
    score = get_neab.tests[0].judge(model,stop_on_error = False, deep_error = True)
    print(score)
    print(type(score))
    print(model.attrs,dtc.attrs)
    #print('sometime')
    observation = score.observation
    dtc.rheobase = score.prediction
    delta = evaluate_as_module.difference(observation,dtc.rheobase)
    dtc.differences[str(get_neab.tests[0])] = delta
    dtc.scores[str(get_neab.tests[0])] = score.sort_key
    print(dtc.scores)
    print(dtc.rheobase, 'recalculation')
    #import pdb; pdb.set_trace()
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


def evaluate(dtc):

    from neuronunit.optimization import get_neab
    import numpy as np
    fitness = [ 1.0 for i in range(0,7) ]

    for k,t in enumerate(dtc.scores.keys()):
        fitness[k] = dtc.scores[str(t)]
    #fitness[7] = np.sum(list(fitness[0:6]))
    print(fitness)
    return fitness[0],fitness[1],\
           fitness[2],fitness[3],\
           fitness[4],fitness[5],\
           fitness[6],#fitness[7]

def update_pop(pop,td):
    '''
    Inputs a population of genes (pop).
    Returned neuronunit scored DTCs (dtcpop).
    '''
    # this method converts a population of genes to a population of Data Transport Containers,
    # Which act as communicatable data types for storing model attributes.
    # Rheobase values are found on the DTCs
    # DTCs for which a rheobase value of x (pA)<=0 are filtered out
    # DTCs are then scored by neuronunit, using neuronunit models that act in place.

    from neuronunit.optimization import model_parameters as modelp
    from neuronunit.optimization import evaluate_as_module
    from neuronunit.optimization.exhaustive_search import parallel_method

    update_dtc_pop = evaluate_as_module.update_dtc_pop
    pre_format = evaluate_as_module.pre_format
    # given the wrong attributes, and they don't have rheobase values.
    dtcpop = list(update_dtc_pop(pop, td))
    #import pdb; pdb.set_trace()
    dtcpop = list(map(dtc_to_rheo,dtcpop))
    print(len(dtcpop), dtcpop)
    for dtc in dtcpop:
        print(dtc.rheobase['value'])
    #import pdb; pdb.set_trace()

    dtcpop = list(filter(lambda dtc: dtc.rheobase['value'] > 0.0 , dtcpop))
    print(len(dtcpop),len(pop),dtcpop, 'hangs here?')
    print(len(dtcpop), dtcpop)

    while len(dtcpop) < len(pop):
        print('stuck here c.1.1?')
        dtcpop.append(dtcpop[0])

    dtcpop = list(map(pre_format,dtcpop))
    print('stuck here d.1.2?')

    dtcpop = dview.map(parallel_method,dtcpop).get()
    print('stuck here d.1.3?')

    rc.wait(dtcpop)
    print(len(dtcpop),'stuck broken 0.a')

    dtcpop = list(filter(lambda dtc: type(dtc.scores['RheobaseTestP']) is not type(None), dtcpop))
    print(len(dtcpop),'stuck broken a')

    dtcpop = list(filter(lambda dtc: not type(None) in (list(dtc.scores.values())), dtcpop))
    print(len(dtcpop),'stuck broken b')
    dtc_temp = []
    for dtc in dtcpop:
        temp = list(dtc.scores.values())
        if None not in temp:
            dtc_temp.append(dtc)
        print('stuck broken c?')
    dtcpop = dtc_temp
    #dtcpop = [ dtc for dtc in dtcpop for v in list(dtc.scores.values()) if type(v) is not type(None) ]
    #dtcpop = list(filter(lambda dtc: for v in list(dtc.scores.values()) is not type(None), dtcpop))

    while len(dtcpop) < len(pop):
        dtcpop.append(dtcpop[0])

    #pop = [pop[i]
    for i,d in enumerate(dtcpop):
        pop[i].rheobase = d.rheobase
    #return
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
