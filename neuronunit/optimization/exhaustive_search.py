import unittest
import os
import quantities as pq
import numpy as np
import importlib
#importlib.machinery
#import os
#importlib.machinery.SourceFileLoader('neuronunit', os.getcwd()+str('../'))
#os.system('ipcluster start -n 8 --profile=default & sleep 25; python stdout_worker.py &')
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

def create_grid(npoints=3,nparams=7):
    from neuronunit.optimization import model_parameters as modelp
    from sklearn.grid_search import ParameterGrid
    mp = modelp.model_params
    # smaller is a dictionary thats not necessarily as big
    # as the grid defined in the model_params file. Its not necessarily
    # a smaller dictionary, if it is smaller it is reduced by reducing sampling
    # points.
    smaller = {}
    smaller = sample_points(mp, npoints=npoints)
    key_list = list(smaller.keys())
    reduced_key_list = key_list[0:nparams]
    # subset is reduced, by reducing parameter keys.
    subset = { k:smaller[k] for k in reduced_key_list }
    grid = list(ParameterGrid(subset))
    return grid

def parallel_method(dtc):
    from neuronunit.optimization import get_neab

    from neuronunit.models.reduced import ReducedModel
    model = ReducedModel(get_neab.LEMS_MODEL_PATH,name=str('vanilla'),backend='NEURON')
    model.set_attrs(dtc.attrs)
    get_neab.tests[0].prediction = dtc.rheobase
    model.rheobase = dtc.rheobase['value']
    scores = []
    from neuronunit.optimization import evaluate_as_module
    dtc = evaluate_as_module.pre_format(dtc)
    for k,t in enumerate(get_neab.tests):
        if k!=0 and float(dtc.rheobase['value']) > 0:
            t.params = dtc.vtest[k]
            score = t.judge(model,stop_on_error = False, deep_error = True)
            dtc.scores[str(t)] = score.sort_key
            #import pdb; pdb.set_trace()
            #scores.append(score.sort_key)
    return dtc

def dtc_to_rheo(dtc):
    from neuronunit.models.reduced import ReducedModel
    from neuronunit.optimization import get_neab
    model = ReducedModel(get_neab.LEMS_MODEL_PATH,name=str('vanilla'),backend='NEURON')
    model.set_attrs(dtc.attrs)
    dtc.scores = {}

    score = get_neab.tests[0].judge(model,stop_on_error = False, deep_error = True)
    #dtc.scores[str(get_neab.tests[0])] = score.sort_key

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

#npoints = 2
#nparams = 2

def run_grid(npoints,nparams):
    # not all models will produce scores, since models with rheobase <0 are filtered out.

    grid_points = create_grid(npoints = npoints,nparams=nparams)
    dtcpop = list(dview.map_sync(update_dtc_pop,grid_points))
    print(dtcpop)
    # The mapping of rheobase search needs to be serial mapping for now, since embedded in it's functionality is a
    # a call to dview map.
    # probably this can be bypassed in the future by using zeromq's Client (by using ipyparallel's core module/code base more directly)
    dtcpop = list(map(dtc_to_rheo,dtcpop))
    print(dtcpop)

    filtered_dtcpop = list(filter(lambda dtc: dtc.rheobase['value'] > 0.0 , dtcpop))
    print(filtered_dtcpop)
    #dtcpop = list(map(parallel_method,filtered_dtcpop))


    dtcpop = list(dview.map_sync(parallel_method,filtered_dtcpop))
    return dtcpop
