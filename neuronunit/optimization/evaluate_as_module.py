##
# Assumption that this file was executed after first executing the bash: ipcluster start -n 8 --profile=default &
##



import os
from neuronunit.models import backends
import neuronunit
print(neuronunit.models.__file__)
from neuronunit.models.reduced import ReducedModel
from neuronunit.tests import get_neab
from ipyparallel import depend, require, dependent
import ipyparallel as ipp
rc = ipp.Client(profile='default')
rc[:].use_cloudpickle()
dview = rc[:]
model = ReducedModel(get_neab.LEMS_MODEL_PATH,name='vanilla',backend='NEURON')
#model.load_model()


class Individual(object):
    '''
    When instanced the object from this class is used as one unit of chromosome or allele by DEAP.
    Extends list via polymorphism.
    '''
    def __init__(self, *args):
        list.__init__(self, *args)
        self.error=None
        self.results=None
        self.name=''
        self.attrs = {}
        self.params=None
        self.score=None
        self.fitness=None
        self.lookup={}
        self.rheobase=None
        self.fitness = creator.FitnessMin
@require('numpy, model_parameters, deap','random')
def import_list(ipp):
    Individual = ipp.Reference('Individual')
    from deap import base, creator, tools
    import deap
    import random
    history = deap.tools.History()
    toolbox = base.Toolbox()
    import model_parameters as modelp
    import numpy as np
    sub_set = []
    whole_BOUND_LOW = [ np.min(i) for i in modelp.model_params.values() ]
    whole_BOUND_UP = [ np.max(i) for i in modelp.model_params.values() ]
    BOUND_LOW = whole_BOUND_LOW
    BOUND_UP = whole_BOUND_UP
    NDIM = len(BOUND_UP)#+1
    def uniform(low, up, size=None):
        try:
            return [random.uniform(a, b) for a, b in zip(low, up)]
        except TypeError:
            return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]
    # weights vector should compliment a numpy matrix of eigenvalues and other values
    creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=30.0)
    toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM)
    return toolbox, tools, history, creator, base
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


    def transform(ind):
        from data_transport_container import DataTC
        dtc = DataTC()
        print(dtc)
        param_dict = {}
        for i,j in enumerate(ind):
            param_dict[td[i]] = str(j)
        dtc.attrs = param_dict
        dtc.evaluated = False
        return dtc


    if len(pop) > 0:
        dtcpop = list(dview.map_sync(transform, pop))
    else:
        # In this case pop is not really a population but an individual
        # but parsimony of naming variables
        # suggests not to change the variable name to reflect this.
        dtcpop = transform(pop)
    return dtcpop

def get_trans_dict(param_dict):
    trans_dict = {}
    for i,k in enumerate(list(param_dict.keys())):
        trans_dict[i]=k
    return trans_dict

def dt_to_ind(dtc,td):
    '''
    Re instanting data transport container at every update dtcpop
    is Noneifying its score attribute, and possibly causing a
    performance bottle neck.
    '''
    ind =[]
    for k in td.keys():
        ind.append(dtc.attrs[td[k]])
    ind.append(dtc.rheobase)
    return ind



def difference(observation,prediction): # v is a tesst
    '''
    This method does not do what you would think
    from reading it.

    rescaling is the culprit. I suspect I do not
    understand how to rescale one unit with another
    compatible unit.
    '''
    import numpy as np

    # The trick is.
    # prediction always has value. but observation 7 out of 8 times has mean.

    if 'value' in prediction.keys():
        unit_predictions = prediction['value']
        if 'mean' in observation.keys():
            unit_observations = observation['mean']
        elif 'value' in observation.keys():
            unit_observations = observation['value']

    if 'mean' in prediction.keys():
        unit_predictions = prediction['mean']
        if 'mean' in observation.keys():
            unit_observations = observation['mean']
        elif 'value' in observation.keys():
            unit_observations = observation['value']

    ##
    # Repurposed from from sciunit/sciunit/scores.py
    # line 156
    ##
    assert type(observation) in [dict,float,int,pq.Quantity]
    assert type(prediction) in [dict,float,int,pq.Quantity]

    def extract_mean_or_value(observation, prediction):
        values = {}
        for name,data in [('observation',observation),
                          ('prediction',prediction)]:
            if type(data) is not dict:
                values[name] = data
            elif key is not None:
                values[name] = data[key]
            else:
                try:
                    values[name] = data['mean'] # Use the mean.
                except KeyError: # If there isn't a mean...
                    try:
                        values[name] = data['value'] # Use the value.
                    except KeyError:
                        raise KeyError(("%s has neither a mean nor a single "
                                        "value" % name))
        return values['observation'], values['prediction']

    obs, pred = extract_mean_or_value(observation, prediction)
    ratio = pred / obs
    ratio = utils.assert_dimensionless(value)


    #to_r_s = unit_observations.units
    #unit_predictions = unit_predictions.rescale(to_r_s)
    #unit_observations = unit_observations.rescale(to_r_s)
    unit_delta = np.abs( np.abs(unit_observations)-np.abs(unit_predictions) )


    return float(unit_delta), ratio


def pre_format(dtc):
    import quantities as pq
    import copy
    dtc.vtest = None
    dtc.vtest = {}
    from neuronunit.tests import get_neab
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
            dtc.vtest[k]['injected_square_current']['delay'] = 100 * pq.ms
    return dtc


def cache_sim_runs(dtc):
    '''
    This could be used to stop neuronunit tests
    from rerunning the same current injection set on the same
    set of parameters
    '''
    from neuronunit.models import backends
    from neuronunit.models.reduced import ReducedModel
    import quantities as pq
    import numpy as np
    from neuronunit.tests import get_neab


    import copy
    # copying here is critical for get_neab
    tests = copy.copy(get_neab.tests)
    vtests = pre_format(dtc)
    if float(dtc.rheobase) > 0.0:
        for k,t in enumerate(tests):
            if k > 0:
                model = ReducedModel(get_neab.LEMS_MODEL_PATH,name=str('vanilla'),backend='NEURON')
                model.set_attrs(attrs = dtc.attrs)
                # check if these attributes have been evaluated before.
                if str(dtc.attrs) in model.lookup.keys:
                    return dtc
                else:
                    score = t.judge(model,stop_on_error = False, deep_error = True)
                    v_m = model.get_membrane_potential()
                    print(type(v_m),'within pre evaluate, eam')
                    if 't' not in dtc.results:
                        dtc.results[t] = {}
                        dtc.results[t]['v_m'] = v_m
                    elif 't' in dtc.results:
                        dtc.results[t]['v_m'] = v_m
                    dtc.cached[str(dtc.attrs)] = dtc.results
    return dtc

'''
def evaluate(dtc,weight_matrix = None):#This method must be pickle-able for ipyparallel to work.

    # Inputs: An individual gene from the population that has compound parameters, and a tuple iterator that
    # is a virtual model object containing an appropriate parameter set, zipped togethor with an appropriate rheobase
    # value, that was found in a previous rheobase search.

    # outputs: a tuple that is a compound error function that NSGA can act on.

    # Assumes rheobase for each individual virtual model object (dtc) has already been found
    # there should be a check for dtc.rheobase, and if not then error.
    # Inputs a gene and a virtual model object.
    # outputs are error components.


    from neuronunit.models import backends
    from neuronunit.models.reduced import ReducedModel
    import quantities as pq
    import numpy as np
    from neuronunit.tests import get_neab

    model = ReducedModel(get_neab.LEMS_MODEL_PATH,name=str('vanilla'),backend='NEURON')
    model.load_model()
    assert type(dtc.rheobase) is not type(None)
    model.set_attrs(attrs = dtc.attrs)
    model.rheobase = dtc.rheobase['value']

    import copy
    tests = copy.copy(get_neab.tests)
    pre_fitness = []
    fitness = []
    differences = []
    fitness1 = []

    if float(dtc.rheobase) <= 0.0:
        fitness1 = [ 125.0 for i in tests ]



    elif float(dtc.rheobase) > 0.0:
        for k,v in enumerate(tests):


            # Spike width tests and amplitude tests assume a rheobase current injection which does not seem
            # to be happening.

            if k == 1 or k == 2 or k == 3:
                # Negative square pulse current.
                v.params['injected_square_current']['duration'] = 100 * pq.ms
                v.params['injected_square_current']['amplitude'] = -10 *pq.pA
                v.params['injected_square_current']['delay'] = 30 * pq.ms

            if k == 0 or k == 4 or k == 5 or k == 6 or k == 7:
                # Threshold current.
                v.params['injected_square_current']['duration'] = 1000 * pq.ms
                v.params['injected_square_current']['amplitude'] = dtc.rheobase['value']
                v.params['injected_square_current']['delay'] = 100 * pq.ms

            vtests = pre_format(copy.copy(dtc))
            for key, value in vtests[k].items():
                print('broken')
                print(key,value,v.params)
                #v.params['injected_square_current'][key] = value
            #vtest[k] = {}
            if k == 0:
                v.prediction = None
                v.prediction = {}
                v.prediction['value'] = dtc.rheobase['value']

            assert type(model) is not type(None)
            score = v.judge(model,stop_on_error = False, deep_error = True)

            #if type(v.prediction) is type(None):
            #    import pdb; pdb.set_trace()
            if type(v.prediction) is not type(None):
                differences.append(difference(v))
                pre_fitness.append(float(score.sort_key))
            else:
                differences.append(None)
                pre_fitness.append(125.0)

            model.run_number += 1
            #dtc.results[t]
    # outside of the test iteration block.
        for k,f in enumerate(copy.copy(pre_fitness)):

            fitness1.append(difference(v))

            if k == 5:
                from neuronunit import capabilities
                ans = model.get_membrane_potential()
                sw = capabilities.spikes2widths(ans)
                unit_observations = tests[5].observation['mean']

                #unit_observations = v.observation['value']
                to_r_s = unit_observations.units
                unit_predictions  = float(sw.rescale(to_r_s))
                fitness1[5] = float(np.abs( np.abs(float(unit_observations))-np.abs(float(unit_predictions))))
                #fitness1[5] = unit_delta
            if k == 0:
                fitness1.append(differences[0])
            if differences[0] > 10.0:
                if k != 0:
                    #fitness1.append(pre_fitness[k])
                    fitness1.append(pre_fitness[k] + 1.5 * differences[0] ) # add the rheobase error to all the errors.
                    assert fitness1[k] != pre_fitness[k]
            else:
                fitness1.append(pre_fitness[k])
            if k == 1:
                fitness1.append(differences[1])
            if differences[1] > 10.0 :
                if k != 1 and len(fitness1)>1 :
                    #fitness1.append(pre_fitness[k])
                    fitness1.append(pre_fitness[k] + 1.25 * differences[1] ) # add the rheobase error to all the errors.
                    assert fitness1[k] != pre_fitness[k]
    pre_fitness = []
    return fitness1[0],fitness1[1],\
           fitness1[2],fitness1[3],\
           fitness1[4],fitness1[5],\
           fitness1[6],fitness1[7],

'''
