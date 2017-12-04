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
dview = rc[:]




from ipyparallel import depend, require, dependent
# Import get_neab has to happen exactly here. It has to be called only on
# controller (rank0, it has)
from neuronunit import tests
from neuronunit.optimization import get_neab
tests = get_neab.tests



def file_write(tests):
    import pickle
    with open('ne_pickle', 'wb') as f:
        pickle.dump(tests, f)

#dview.block = True
test_container = { 'tests':tests }
dview.push(test_container, targets=0)
dview.execute('print(tests)')
dview.execute("import pickle")
#with dview.sync_imports():
#    import pickle

# serial file write.
rc[0].apply(file_write, tests)
#Broadcast the tests to workers
test_dic = {}
for t in tests:
    test_dic[str(t.name)] = t
dview.push(test_dic)

def get_tests():
    '''
    Pull tests
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
    #from neuronunit.optimization import get_neab
    from neuronunit.optimization import evaluate_as_module
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
    tests = get_tests()
    get_neab.tests[0].dview = dview
    score = get_neab.tests[0].judge(model,stop_on_error = False, deep_error = True)
    observation = score.observation
    prediction = score.prediction
    delta = evaluate_as_module.difference(observation,prediction)
    dtc.differences[str(get_neab.tests[0])] = delta
    dtc.scores[str(get_neab.tests[0])] = score.sort_key
    dtc.rheobase = score.prediction
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

def make_files(dtc):
    return ref

def evaluate(dtc):

    from neuronunit.optimization import get_neab
    import numpy as np
    fitness = [ -100.0 for i in range(0,8)]
    for k,t in enumerate(dtc.scores.keys()):
        if dtc.rheobase['value'] > 0.0:
            fitness[k] = dtc.scores[str(t)]
        else:
            fitness[k] = -100.0

    return fitness[0],fitness[1],\
           fitness[2],fitness[3],\
           fitness[4],fitness[5],\
           fitness[6],fitness[7],

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
    dtcpop = list(update_dtc_pop(pop, td))
    dtcpop = list(map(dtc_to_rheo,dtcpop))

    print('\n\n\n\n rheobase complete \n\n\n\n')
    dtcpop = list(map(pre_format,dtcpop))
    print('\n\n\n\n preformat complete \n\n\n\n')
    dtcpop = dview.map(parallel_method,dtcpop).get()
    rc.wait(dtcpop)

    print('\n\n\n\n score calculation complete \n\n\n\n')
    return list(dtcpop)



def create_subset(nparams=10,provided_keys=None):
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
