#import matplotlib # Its not that this file is responsible for doing plotting, but it calls many modules that are, such that it needs to pre-empt
# setting of an appropriate backend.
#matplotlib.use('agg')

import numpy as np
import dask.bag as db
import pandas as pd
# Import get_neab has to happen exactly here. It has to be called only on
from neuronunit import tests
from neuronunit.optimization import get_neab
from neuronunit.models.reduced import ReducedModel
from neuronunit.optimization.model_parameters import model_params, path_params
import numpy
import dask.bag as db
from neuronunit.optimization import model_parameters as modelp
from itertools import repeat
import copy
from neuronunit.optimization import get_neab
from pyneuroml import pynml
import dask.bag as db

import copy
import numpy as np
from deap import base
import dask.bag as db
from neuronunit.optimization.data_transport_container import DataTC
from neuronunit.models.interfaces import glif
from neuronunit.optimization import model_parameters as modelp

from neuronunit.models.interfaces import glif
from neuronunit.models.reduced import ReducedModel

from itertools import repeat
import neuronunit
import multiprocessing
npartitions = multiprocessing.cpu_count()
from collections import Iterable


class WSListIndividual(list):
    """Individual consisting of list with weighted sum field"""
    def __init__(self, *args, **kwargs):
        """Constructor"""
        self.rheobase = None

        super(WSListIndividual, self).__init__(*args, **kwargs)



class WSFloatIndividual(float):
    """Individual consisting of list with weighted sum field"""
    def __init__(self, *args, **kwargs):
        """Constructor"""
        self.rheobase = None

        super(WSFloatIndividual, self).__init__()

def write_opt_to_nml(path,param_dict):
    '''
    Write optimimal simulation parameters back to NeuroML.
    '''
    orig_lems_file_path = path_params['model_path']
    more_attributes = pynml.read_lems_file(orig_lems_file_path,
                                           include_includes=True,
                                           debug=False)
    for i in more_attributes.components:
        new = {}
        if str('izhikevich2007Cell') in i.type:
            for k,v in i.parameters.items():
                units = v.split()
                if len(units) == 2:
                    units = units[1]
                else:
                    units = 'mV'
                new[k] = str(param_dict[k]) + str(' ') + str(units)
            i.parameters = new
    fopen = open(path+'.nml','w')
    more_attributes.export_to_file(fopen)
    fopen.close()
    return

def dtc_to_rheo(xargs):
    dtc,rtest,backend = xargs
    LEMS_MODEL_PATH = path_params['model_path']

    model = ReducedModel(LEMS_MODEL_PATH,name=str('vanilla'),backend='NEURON')
    model.set_attrs(dtc.attrs)
    dtc.scores = {}
    dtc.score = {}
    score = rtest.judge(model,stop_on_error = False, deep_error = True)
    if score.sort_key is not None:
        dtc.scores.get(str(rtest), 1 - score.sort_key)
        dtc.scores[str(rtest)] = 1 - score.sort_key
    dtc.rheobase = score.prediction
    return dtc

def nunit_evaluation(tuple_object):#,backend=None):
    # Inputs single data transport container modules, and neuroelectro observations that
    # inform test error error_criterion
    # Outputs Neuron Unit evaluation scores over error criterion
    dtc,tests = tuple_object
    dtc = copy.copy(dtc)
    dtc.model_path = path_params['model_path']
    LEMS_MODEL_PATH = path_params['model_path']
    assert dtc.rheobase is not None
    backend = dtc.backend
    if backend == 'glif':

        model = glif.GC()#ReducedModel(LEMS_MODEL_PATH,name=str('vanilla'),backend=('NEURON',{'DTC':dtc}))
        tests[0].prediction = dtc.rheobase
        model.rheobase = dtc.rheobase['value']
    else:
        model = ReducedModel(LEMS_MODEL_PATH,name = str('vanilla'),backend = ('NEURON',{'DTC':dtc}))
        model.set_attrs(dtc.attrs)
        tests[0].prediction = dtc.rheobase
        model.rheobase = dtc.rheobase['value']


    for k,t in enumerate(tests[1:-1]):
        t.params = dtc.vtest[k]
        score = None
        score = t.judge(model,stop_on_error = False, deep_error = False)
        dtc.score[str(t)] = {}
        dtc.score[str(t)]['value'] = copy.copy(score.sort_key)

        if hasattr(score,'prediction'):

            dtc.score[str(t)][str('prediction')] = score.prediction
            dtc.score[str(t)][str('observation')] = score.observation
            if 'mean' in score.observation.keys() and 'mean' in score.prediction.keys():
                print(score.observation.keys())
                dtc.score[str(t)][str('agreement')] = np.abs(score.observation['mean'] - score.prediction['mean'])
        else:
            print(score,type(score))
        if score.sort_key is not None:
    	    ##
    	    # This probably does something different to what I thought.
    	    ##
            #dtc.scores.get(str(t), 1.0 - score.sort_key)

            dtc.scores[str(t)] = 1.0 - score.sort_key
        else:
            dtc.scores[str(t)] = 1.0
    return dtc


def evaluate(dtc):
    fitness = [ 1.0 for i in range(0,len(dtc.scores.keys())) ]
    for k,t in enumerate(dtc.scores.keys()):
        fitness[k] = dtc.scores[str(t)]
    return tuple(fitness,)

def get_trans_list(param_dict):
    trans_list = []
    for i,k in enumerate(list(param_dict.keys())):
        trans_list.append(k)
    return trans_list

def format_test(xargs):
    '''
    pre format the current injection dictionary based on pre computed
    rheobase values of current injection.
    This is much like the hooked method from the old get neab file.
    '''
    dtc,tests = xargs
    #import copy
    import quantities as pq
    #import copy
    dtc.vtest = None
    dtc.vtest = {}
    #from neuronunit.optimization import get_neab
    #tests = get_neab.tests
    for k,v in enumerate(tests):
        dtc.vtest[k] = {}
        #dtc.vtest.get(k,{})
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



def update_dtc_pop(pop, td, backend = None):

    '''
    inputs a population of genes/alleles, the population size MU, and an optional argument of a rheobase value guess
    outputs a population of genes/alleles, a population of individual object shells, ie a pickleable container for gene attributes.
    Rationale, not every gene value will result in a model for which rheobase is found, in which case that gene is discarded, however to
    compensate for losses in gene population size, more gene samples must be tested for a successful return from a rheobase search.
    If the tests return are successful these new sampled individuals are appended to the population, and then their attributes are mapped onto
    corresponding virtual model objects.
    '''

    toolbox = base.Toolbox()
    pop = [toolbox.clone(i) for i in pop ]

    def transform(ind):
        # The merits of defining a function in a function
        # is that it yields a semi global scoped variables.
        #
        dtc = DataTC()
        LEMS_MODEL_PATH = str(neuronunit.__path__[0])+str('/models/NeuroML2/LEMS_2007One.xml')
        if backend is not None:
            dtc.backend = backend
        else:
            dtc.backend = 'NEURON'

        dtc.attrs = {}
        if isinstance(ind, Iterable):
            for i,j in enumerate(ind):
                dtc.attrs[str(td[i])] = j
        else:
            dtc.attrs[str(td[0])] = ind
        dtc.evaluated = False
        return dtc

    if len(pop) > 1:

        npart = np.min([multiprocessing.cpu_count(),len(pop)])
        bag = db.from_sequence(pop, npartitions = npart)
        dtcpop = list(bag.map(transform).compute())

    else:
        # In this case pop is not really a population but an individual
        # but parsimony of naming variables
        # suggests not to change the variable name to reflect this.
        dtcpop = list(transform(pop))
    return dtcpop

def rheobase(pop, td, rt):
    if not hasattr(pop[0],'rheobase'):
        pop = [ WSFloatIndividual(ind) for ind in pop if type(ind) is not type(list) ]
    print(pop)
    dtcpop = update_dtc_pop(pop, td)
    #if isinstance(dtcpop, Iterable):
    dtcpop = iter(dtcpop)
    xargs = iter(zip(dtcpop,repeat(rt),repeat('NEURON')))
    dtcpop = list(map(dtc_to_rheo,xargs))
    for ind,d in zip(pop,dtcpop):
        ind.rheobase = d.rheobase
    # Done changed the score away from Ratio to Z.
    return pop, dtcpop


def sense_non_viable_impute(pop, td, tests):
    orig_MU = len(pop)
    pop, dtcpop = rheobase(pop, td, tests[0])
    delta = orig_MU-len(pop)
    if delta:
        # making new genes here introduces too much code complexity,
        # instead make up differences by extending existing lists with duplicates
        # from itself.
        # This will decrease diversity.
        far_back = -delta-1
        pop.extend(pop[far_back:-1])
        dtcpop.extend(dtcpop[far_back:-1])
    return pop,dtcpop

def update_exhaust_pop(pop, tests, td, backend = None):
    '''
    # This is different to the GA in two ways.
    Inputs a population of genes (pop).
    Returned neuronunit scored DTCs (dtcpop).
    This method converts a population of genes to a population of Data Transport Containers,
    Which act as communicatable data types for storing model attributes.
    Rheobase values are found on the DTCs
    DTCs for which a rheobase value of x (pA)<=0 are filtered out
    DTCs are then scored by neuronunit, using neuronunit models that act in place.

    Assumptions, DTC pop is massiveself. The state of the iterator should be saved, such as to interupt tolerant.
    The sampling should occur at coarse resolution first, then finer resolutions later.

    Both the search, and the iterator should routinely save to disk.
    '''
    pop, dtcpop = rheobase(pop, td, tests[0])
    dtcpop = copy.copy(dtcpop)
    xargs = zip(dtcpop,repeat(tests))
    dtcpop = iter(map(format_test,xargs))
    npart = np.min([multiprocessing.cpu_count(),len(pop)])
    dtcbag = db.from_sequence(list(zip(dtcpop,repeat(tests))), npartitions = npart)

    # NeuronUnit testing
    dtcpop = list(dtcbag.map(nunit_evaluation).compute())
    #import pdb; pdb.set_trace()
    last = 0

    for d in dtcpop:
        print(d.attrs.values(),'values different')

    for d in dtcpop: temp = sum(d.attrs.values()); assert last != temp; print(last,temp) ;last = temp
    for d in dtcpop: temp = d.scores.values();  print(last,temp) ;last = temp
    for d in dtcpop: temp = d.scores.values();  print(last,temp) ;last = temp

    for d in dtcpop:
        temp = sum(d.scores.values());
        if last==temp:
            print(len(d.scores.values()))
            import pdb; pdb.set_trace()
        assert last != temp ;last = temp

    for i,d in enumerate(copy.copy(dtcpop)):
        pop[i].dtc = copy.copy(dtcpop[i])
        assert hasattr(pop[i],'dtc')
    invalid_dtc_not = [ i for i in pop if not hasattr(i,'dtc') ]
    try:
        assert len(invalid_dtc_not) == 0
    except:
        print(len(invalid_dtc_not)>0)
        raise ValueError('value error invalid_dtc_not')
    for d in pop: temp = sum(d.dtc.scores.values()); assert last != temp ;print(temp,last) ; last = temp

    # https://distributed.readthedocs.io/en/latest/memory.html

    return pop


def update_deap_pop(pop, tests, td, backend = None):
    '''
    Inputs a population of genes (pop).
    Returned neuronunit scored DTCs (dtcpop).
    This method converts a population of genes to a population of Data Transport Containers,
    Which act as communicatable data types for storing model attributes.
    Rheobase values are found on the DTCs
    DTCs for which a rheobase value of x (pA)<=0 are filtered out
    DTCs are then scored by neuronunit, using neuronunit models that act in place.
    '''
    # Rheobase value obtainment.

    pop,dtcpop = sense_non_viable_impute(pop, td, tests)
    # NeuronUnit testing
    xargs = zip(dtcpop,repeat(tests))
    dtcpop = list(map(format_test,xargs))
    npart = np.min([multiprocessing.cpu_count(),len(pop)])
    dtcbag = db.from_sequence(list(zip(dtcpop,repeat(tests))), npartitions = npart)
    dtcpop = list(dtcbag.map(nunit_evaluation).compute())

    for i,d in enumerate(dtcpop):
        #assert pop[i][0] in list(d.attrs.values())
        pop[i].dtc = None
        pop[i].dtc = copy.copy(dtcpop[i])
    invalid_dtc_not = [ i for i in pop if not hasattr(i,'dtc') ]
    try:
        assert len(invalid_dtc_not) == 0
    except:
        print(len(invalid_dtc_not)>0)
        raise ValueError('value error invalid_dtc_not')
    # https://distributed.readthedocs.io/en/latest/memory.html
    return pop


def create_subset(nparams = 10, provided_dict = None):
    import numpy as np
    if type(provided_dict) is type(None):
        mp = modelp.model_params
        key_list = list(mp.keys())
        reduced_key_list = key_list[0:nparams]
    else:
        key_list = list(provided_dict.keys())
        reduced_key_list = key_list[0:nparams]

    subset = { k:provided_dict[k] for k in reduced_key_list }
    return subset
