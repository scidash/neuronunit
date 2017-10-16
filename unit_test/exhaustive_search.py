import unittest
import os
import quantities as pq
import numpy as np


os.system('ipcluster start -n 8 --profile=default & sleep 25;')
import ipyparallel as ipp
rc = ipp.Client(profile='default')
rc[:].use_cloudpickle()
dview = rc[:]


def sample_points(iter_dict, npoints=3):
    import numpy as np
    replacement={}
    for k,v in iter_dict.items():
        sample_points = list(np.linspace(v.max(),v.min(),npoints))
        replacement[k] = sample_points
    return replacement



def create_list(npoints=3):
    from neuronunit.optimization import model_parameters as modelp

    mp = modelp.model_params
    all_keys = [ key for key in mp.keys() ]
    smaller = {}
    smaller = sample_points(mp, npoints=npoints)



    iter_list=[ {'a':i,'b':j,'vr':k,'vpeak':l,'k':m,'c':n,'C':o,'d':p,'v0':q,'vt':r} for i in smaller['a'] for j in smaller['b'] \
    for k in smaller['vr'] for l in smaller['vpeak'] \
    for m in smaller['k'] for n in smaller['c'] \
    for o in smaller['C'] for p in smaller['d'] \
    for q in smaller['v0'] for r in smaller['vt'] ]
    # the size of this list is 59,049 approx 60,000 calls after rheobase is found.
    # assert 3**10 == 59049
    return iter_list

def parallel_method(dtc):
    from neuronunit.optimization import get_neab
    get_neab.LEMS_MODEL_PATH = '/home/jovyan/neuronunit/neuronunit/optimization/NeuroML2/LEMS_2007One.xml'
    #from neuronunit.models import backends
    from neuronunit.models.reduced import ReducedModel
    model = ReducedModel(get_neab.LEMS_MODEL_PATH,name=str('vanilla'),backend='NEURON')
    model.set_attrs(dtc.attrs)
    get_neab.tests[0].prediction = dtc.rheobase
    model.rheobase = dtc.rheobase['value']
    scores = []

    #scores.append(score.sort_key,score)
    from neuronunit.optimization import evaluate_as_module
    dtc = evaluate_as_module.pre_format(dtc)
    #for k,t in dtc.
    #get_neab.tests.parameters = dtc.vtests
    for k,t in enumerate(get_neab.tests):
        if k>1:
            t.params=dtc.vtest[k]
            score = t.judge(model,stop_on_error = False, deep_error = True)
            scores.append(score.sort_key)
    return scores

def dtc_to_rheo(dtc):
    print(dtc)
    from neuronunit.models.reduced import ReducedModel
    from neuronunit.optimization import get_neab
    model = ReducedModel(get_neab.LEMS_MODEL_PATH,name=str('vanilla'),backend='NEURON')
    model.set_attrs(dtc.attrs)
    score = get_neab.tests[0].judge(model,stop_on_error = False, deep_error = True)
    observation = score.observation
    dtc.rheobase =  score.prediction
    return dtc

def update_dtc_pop(item_of_iter_list):

    from neuronunit.optimization import data_transport_container
    dtc = data_transport_container.DataTC()
    dtc.attrs = item_of_iter_list
    dtc.scores = []
    dtc.rheobase = None
    dtc.evaluated = False
    return dtc

npoints = 3
returned_list = create_list(npoints = npoints)
assert len(returned_list) == (npoints ** 10)
dtcpop = list(dview.map_sync(update_dtc_pop,returned_list[0:15]))
print(dtcpop)
scores = []
for dtc in dtcpop:
    dtc = dtc_to_rheo(dtc)
    if float(dtc.rheobase['value']) > 0.0:
        scores.append(parallel_method(dtc))
        print(scores)
    else:
        print('excluded: ',dtc.attrs)

dtcpop = list(map(dtc_to_rheo,dtcpop))
print([i.rheobase for i in dtcpop])

dtcpop = [i for i in dtcpop if i.rheobase > 0 ]

scores = list(dview.map(parallel_method,dtcpop).get())
