import unittest
import os
import quantities as pq
import numpy as np


os.system('ipcluster start -n 8 --profile=default & sleep 5;')
import ipyparallel as ipp
rc = ipp.Client(profile='default')
rc[:].use_cloudpickle()
dview = rc[:]

def create_list():
    from neuronunit.optimization import model_parameters as modelp

    mp = modelp.model_params
    all_keys = [ key for key in mp.keys() ]
    smaller = {}
    # First create a smaller subet of the larger parameter dictionary.
    #
    for k in all_keys:
        subset = {}
        subset[k] = (mp[k][0] , mp[k][int(len(mp[k])/2.0)], mp[k][-1] )
        smaller.update(subset)


    iter_list=[ {'a':i,'b':j,'vr':k,'vpeak':l,'k':m,'c':n,'C':o,'d':p,'v0':q,'vt':r} for i in smaller['a'] for j in smaller['b'] \
    for k in smaller['vr'] for l in smaller['vpeak'] \
    for m in smaller['k'] for n in smaller['c'] \
    for o in smaller['C'] for p in smaller['d'] \
    for q in smaller['v0'] for r in smaller['vt'] ]
    # the size of this list is 59,049 approx 60,000 calls after rheobase is found.
    # assert 3**10 == 59049
    return iter_list

def parallel_method(item_of_iter_list):

    from neuronunit.optimization import get_neab
    get_neab.LEMS_MODEL_PATH = '/home/jovyan/neuronunit/neuronunit/optimization/NeuroML2/LEMS_2007One.xml'
    #from neuronunit.models import backends
    from neuronunit.models.reduced import ReducedModel
    model = ReducedModel(get_neab.LEMS_MODEL_PATH,name=str('vanilla'),backend='NEURON')
    model.set_attrs(item_of_iter_list)
    get_neab.tests[0].prediction = dtc.rheobase
    model.rheobase = dtc.rheobase['value']
    scores = []
    for k,t in enumerate(get_neab.tests):
        if k>1:
            t.params = dtc.vtest[k]
            score = t.judge(model,stop_on_error = False, deep_error = True)
            scores.append(score.sort_key,score)
    return scores

def dtc_to_rheo(dtc):
    from neuronunit.models.reduced import ReducedModel
    from neuronunit.tests import get_neab
    import evaluate_as_module
    model = ReducedModel(get_neab.LEMS_MODEL_PATH,name=str('vanilla'),backend='NEURON')
    model.set_attrs(dtc.attrs)
    score = get_neab.tests[0].judge(model,stop_on_error = False, deep_error = True)
    observation = score.observation
    dtc.score.append(score)
    #prediction = score.prediction
    dtc.rheobase =  score.prediction
    return dtc

def update_dtc_pop(item_of_iter_list):
    '''
    inputs a population of genes/alleles, the population size MU, and an optional argument of a rheobase value guess
    outputs a population of genes/alleles, a population of individual object shells, ie a pickleable container for gene attributes.
    Rationale, not every gene value will result in a model for which rheobase is found, in which case that gene is discarded, however to
    compensate for losses in gene population size, more gene samples must be tested for a successful return from a rheobase search.
    If the tests return are successful these new sampled individuals are appended to the population, and then their attributes are mapped onto
    corresponding virtual model objects.
    '''

    from data_transport_container import DataTC
    dtc = DataTC()
    dtc.attrs = item_of_iter_list
    dtc.evaluated = False
    return dtc

def exhaustive_search(self):
    iter_list = create_list()
    dtcpop = list(dview.map(exhaustive_search,iter_list).get())
    dtcpop = list(dview.map(dtc_to_rheo,dtcpop).get())
    scores = list(dview.map(parallel_method,dtcpop).get())
    return scores
scores = exhaustive_search()
    #score_parameter_pairs = zip(scores,iter_list)


#test_0_run_exhaust()
