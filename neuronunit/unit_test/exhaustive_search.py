
from neuronunit.optimization import get_neab
tests = get_neab.tests


def sample_points(iter_dict, npoints=3):
    import numpy as np
    replacement={}
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
    from sklearn.grid_search import ParameterGrid
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
    from neuronunit.optimization import data_transport_container
    import copy
    dtc = data_transport_container.DataTC()
    from copy import deepcopy
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

    from neuronunit.optimization import model_parameters as modelp
    from sklearn.grid_search import ParameterGrid
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

'''
def dtc_to_rheo(dtc):
    from neuronunit.optimization import get_neab
    import copy
    dtc = copy.copy(dtc)
    dtc.scores = {}
    from neuronunit.models.reduced import ReducedModel
    model = ReducedModel(get_neab.LEMS_MODEL_PATH,name=str('vanilla'),backend=('NEURON',{'DTC':dtc}))
    model.set_attrs(dtc.attrs)
    model.backend = dtc.backend
    model.rheobase = None
    rbt = get_neab.tests[0]
    score = rbt.judge(model,stop_on_error = False, deep_error = True)
    dtc.scores[str(rbt)] = score.sort_key
    dtc.rheobase =  score.prediction
    return dtc

def update_dtc_pop(item_of_iter_list):
    from neuronunit.optimization import data_transport_container
    import copy
    dtc = data_transport_container.DataTC()
    from copy import deepcopy
    dtc.attrs = deepcopy(item_of_iter_list)
    dtc.scores = {}
    dtc.rheobase = None
    dtc.evaluated = False
    dtc.backend = 'NEURON'
    return dtc
'''
def run_grid(npoints,nparams,provided_keys=None):
    # not all models will produce scores, since models with rheobase <0 are filtered out.
    from neuronunit.optimization.optimization_management import nunit_evaluation
    from neuronunit.optimization.optimization_management import update_dtc_pop

    grid_points = create_grid(npoints = npoints,nparams = nparams,vprovided_keys = provided_keys )
    import dask.bag as db
    b = db.bag(grid_points)
    dtcpop = list(db.map(update_dtc_pop,b).compute())
    print(dtcpop)
    # The mapping of rheobase search needs to be serial mapping for now, since embedded in it's functionality is a
    # probably this can be bypassed in the future by using zeromq's Client (by using ipyparallel's core module/code base more directly)
    dtcpop = list(map(dtc_to_rheo,dtcpop))
    print(dtcpop)

    filtered_dtcpop = list(filter(lambda dtc: dtc.rheobase['value'] > 0.0 , dtcpop))
    dtcpop = list(db.map(nunit_evaluation,filtered_dtcpop).compute())
    dtcpop = list(dtcpop)
    dtcpop = list(filter(lambda dtc: type(dtc.scores['RheobaseTestP']) is not type(None), dtcpop))

    return dtcpop
