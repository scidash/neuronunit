##
# Assumption that this file was executed after first executing the bash: ipcluster start -n 8 --profile=default &
##



import os
from neuronunit.models import backends
import neuronunit
print(neuronunit.models.__file__)
from neuronunit.models.reduced import ReducedModel
import get_neab
from ipyparallel import depend, require, dependent

model = ReducedModel(get_neab.LEMS_MODEL_PATH,name='vanilla',backend='NEURON')
model.load_model()

class DataTC(object):
    '''
    Data Transport Vessel

    This Object class serves as a data type for storing rheobase search
    attributes and apriori model parameters,
    with the distinction that unlike the NEURON model this class
    can be cheaply transported across HOSTS/CPUs
    '''
    def __init__(self):
        self.lookup = {}
        self.rheobase = None
        self.previous = 0
        self.run_number = 0
        self.attrs = None
        self.steps = None
        self.name = None
        self.results = None
        self.fitness = None
        self.score = None
        self.boolean = False
        self.initiated = False
        self.delta = []
        self.evaluated = False
        self.results = {}
        self.searched = []
        self.searchedd = {}
#dview.apply_sync(p_imports)
#p_imports()

def difference(v): # v is a tesst
    '''
    This method does not do what you would think
    from reading it.

    rescaling is the culprit. I suspect I do not
    understand how to rescale one unit with another
    compatible unit.
    '''
    assert type(v) is not type(None)
    import numpy as np
    print(v.prediction.keys())
    print(v.prediction.values())

    # The trick is.
    # prediction always has value. but observation 7 out of 8 times has mean.

    if 'value' in v.prediction.keys():
        unit_predictions = v.prediction['value']
        if 'mean' in v.observation.keys():
            unit_observations = v.observation['mean']
        elif 'value' in v.observation.keys():
            unit_observations = v.observation['value']



    if 'mean' in v.prediction.keys():
        unit_predictions = v.prediction['mean']
        if 'mean' in v.observation.keys():
            unit_observations = v.observation['mean']
        elif 'value' in v.observation.keys():
            unit_observations = v.observation['value']

    to_r_s = unit_observations.units
    unit_predictions = unit_predictions.rescale(to_r_s)
    unit_observations = unit_observations.rescale(to_r_s)
    unit_delta = np.abs( np.abs(unit_observations)-np.abs(unit_predictions) )
    return float(unit_delta)

def pre_format(dtc):
    import quantities as pq
    import copy
    vtest = {}
    import get_neab
    tests = get_neab.tests
    for k,v in enumerate(tests):
        vtest[k] = {}
    for k,v in enumerate(tests):
        if k == 1 or k == 2 or k == 3:
            # Negative square pulse current.
            vtest[k]['duration'] = 100 * pq.ms
            vtest[k]['amplitude'] = -10 *pq.pA
            vtest[k]['delay'] = 30 * pq.ms

        if k == 0 or k == 4 or k == 5 or k == 6 or k == 7:
            # Threshold current.
            vtest[k]['duration'] = 1000 * pq.ms
            vtest[k]['amplitude'] = dtc.rheobase * pq.pA
            vtest[k]['delay'] = 100 * pq.ms
    return vtest
#@require('quantities','numpy','get_neab','quanitites')
@require('get_neab')
def evaluate(dtc,weight_matrix = None):#This method must be pickle-able for ipyparallel to work.
    '''
    Inputs: An individual gene from the population that has compound parameters, and a tuple iterator that
    is a virtual model object containing an appropriate parameter set, zipped togethor with an appropriate rheobase
    value, that was found in a previous rheobase search.

    outputs: a tuple that is a compound error function that NSGA can act on.

    Assumes rheobase for each individual virtual model object (dtc) has already been found
    there should be a check for dtc.rheobase, and if not then error.
    Inputs a gene and a virtual model object.
    outputs are error components.
    '''

    from neuronunit.models import backends
    from neuronunit.models.reduced import ReducedModel
    import quantities as pq
    import numpy as np
    #import unittest
    #tc = unittest.TestCase('__init__')
    model = ReducedModel(get_neab.LEMS_MODEL_PATH,name=str('vanilla'),backend='NEURON')
    model.load_model()
    assert type(dtc.rheobase) is not type(None)
    #tests = get_neab.suite.tests
    model.set_attrs(attrs = dtc.attrs)
    model.rheobase = dtc.rheobase * pq.pA
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

            vtests = pre_format(copy.copy(dtc))
            for key, value in vtests[k].items():
                v.params['injected_square_current'][key] = value

            #vtest[k] = {}
            if k == 0:
                v.prediction = None
                v.prediction = {}
                v.prediction['value'] = dtc.rheobase * pq.pA

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
    import get_neab

    import copy
    # copying here is critical for get_neab
    tests = copy.copy(get_neab.tests)
    #from itertools import repeat
    vtests = pre_format(copy.copy(dtc))
    tests[0].prediction = {}
    tests[0].prediction['value'] = dtc.rheobase * pq.pA

    if float(dtc.rheobase) > 0.0:
        for k,t in enumerate(tests):
            '''
            can tests be re written such that it is more closure compatible?
            '''
            t.params = {}
            t.params['injected_square_current'] = {}
            t.params['injected_square_current']['duration'] = None
            t.params['injected_square_current']['amplitude'] = None
            t.params['injected_square_current']['delay'] = None

            for key, value in vtests[k].items():
                t.params['injected_square_current'][key] = value


            if k == 0:
                tests[k].prediction = {}
                tests[k].prediction['value'] = dtc.rheobase * pq.pA

            #new_file_path = str(get_neab.LEMS_MODEL_PATH)+str(os.getpid())
            model = ReducedModel(get_neab.LEMS_MODEL_PATH,name=str('vanilla'),backend='NEURON')
            model.load_model()
            model.set_attrs(attrs = dtc.attrs)
            # check if these attributes have been evaluated before.
            if str(dtc.attrs) in model.lookup.keys:
                return dtc
            else:

                print(t,model,'within pre evaluate, eam')

                score = t.judge(model,stop_on_error = False, deep_error = True)
                print(score,'within pre evaluate, eam')

                print(model.get_spike_count())
                print(type(v_m),'within pre evaluate, eam')

                v_m = model.get_membrane_potential()
                print(type(v_m),'within pre evaluate, eam')
                if 't' not in dtc.results.keys():
                    dtc.results[t] = {}
                    dtc.results[t]['v_m'] = v_m
                elif 't' in dtc.results.keys():
                    dtc.results[t]['v_m'] = v_m
                dtc.lookup[str(dtc.attrs)] = dtc.results
    return dtc

#from scoop import futures


def get_trans_dict(param_dict):
    trans_dict = {}
    for i,k in enumerate(list(param_dict.keys())):
        trans_dict[i]=k
    return trans_dict
import model_parameters
param_dict = model_parameters.model_params

def dtc_to_ind(dtc,td):
    '''
    Re instanting Virtual Model at every update dtcpop
    is Noneifying its score attribute, and possibly causing a
    performance bottle neck.
    '''

    ind =[]
    for k in td.keys():
        ind.append(dtc.attrs[td[k]])
    ind.append(dtc.rheobase)
    return ind



def update_dtc_pop(pop, trans_dict):
    '''
    inputs a population of genes/alleles, the population size MU, and an optional argument of a rheobase value guess
    outputs a population of genes/alleles, a population of individual object shells, ie a pickleable container for gene attributes.
    Rationale, not every gene value will result in a model for which rheobase is found, in which case that gene is discarded, however to
    compensate for losses in gene population size, more gene samples must be tested for a successful return from a rheobase search.
    If the tests return are successful these new sampled individuals are appended to the population, and then their attributes are mapped onto
    corresponding virtual model objects.
    '''
    #from itertools import repeat
    import numpy as np
    import copy
    pop = [toolbox.clone(i) for i in pop ]
    #import utilities
    def transform(ind):
        '''
        Re instanting Virtual Model at every update dtcpop
        is Noneifying its score attribute, and possibly causing a
        performance bottle neck.
        '''
        dtc = DataTC()

        param_dict = {}
        for i,j in enumerate(ind):
            param_dict[trans_dict[i]] = str(j)
        dtc.attrs = param_dict
        dtc.name = dtc.attrs
        dtc.evaluated = False
        return dtc


    if len(pop) > 0:
        dtcpop = dview.map_sync(transform, pop)
        dtcpop = list(copy.copy(dtcpop))
    else:
        # In this case pop is not really a population but an individual
        # but parsimony of naming variables
        # suggests not to change the variable name to reflect this.
        dtcpop = transform(pop)
    return dtcpop



def check_rheobase(dtcpop,pop=None):
    '''
    inputs a population of genes/alleles, the population size MU, and an optional argument of a rheobase value guess
    outputs a population of genes/alleles, a population of individual object shells, ie a pickleable container for gene attributes.
    Rationale, not every gene value will result in a model for which rheobase is found, in which case that gene is discarded, however to
    compensate for losses in gene population size, more gene samples must be tested for a successful return from a rheobase search.
    If the tests return are successful these new sampled individuals are appended to the population, and then their attributes are mapped onto
    corresponding virtual model objects.
    '''
    def check_fix_range(dtc):
        '''
        Inputs: lookup, A dictionary of previous current injection values
        used to search rheobase
        Outputs: A boolean to indicate if the correct rheobase current was found
        and a dictionary containing the range of values used.
        If rheobase was actually found then rather returning a boolean and a dictionary,
        instead logical True, and the rheobase current is returned.
        given a dictionary of rheobase search values, use that
        dictionary as input for a subsequent search.
        '''
        import pdb
        import copy
        import numpy as np
        import quantities as pq
        sub=[]
        supra=[]
        steps=[]
        dtc.rheobase = 0.0
        for k,v in dtc.lookup.items():
            dtc.searchedd[v]=float(k)

            if v == 1:
                #A logical flag is returned to indicate that rheobase was found.
                dtc.rheobase=float(k)
                dtc.searched.append(float(k))
                dtc.steps = 0.0
                dtc.boolean = True
                return dtc
            elif v == 0:
                sub.append(k)
            elif v > 0:
                supra.append(k)

        sub=np.array(sub)
        supra=np.array(supra)

        if len(sub)!=0 and len(supra)!=0:
            #this assertion would only be occur if there was a bug
            assert sub.max()<=supra.min()
        if len(sub) and len(supra):
            everything = np.concatenate((sub,supra))

            center = np.linspace(sub.max(),supra.min(),7.0)
            # The following code block probably looks counter intuitive.
            # Its job is to delete duplicated search values.
            # Ie everything is a list of 'everything' already explored.
            # It then inserts a bias corrected center position.
            for i,j in enumerate(centerl):
                if j in list(everything):

                    np.delete(center,i)
                    # delete the duplicated elements element, and replace it with a corrected
                    # center below.
            #delete the index
            #np.delete(center,np.where(everything is in center))
            #make sure that element 4 in a seven element vector
            #is exactly half way between sub.max() and supra.min()
            center[int(len(center)/2)+1]=(sub.max()+supra.min())/2.0
            steps = [ i*pq.pA for i in center ]

        elif len(sub):
            steps = np.linspace(sub.max(),2*sub.max(),7.0)
            np.delete(steps,np.array(sub))
            steps = [ i*pq.pA for i in steps ]

        elif len(supra):
            steps = np.linspace(-2*(supra.min()),supra.min(),7.0)
            np.delete(steps,np.array(supra))
            steps = [ i*pq.pA for i in steps ]

        dtc.steps = steps
        dtc.rheobase = None
        return copy.copy(dtc)


    def check_current(ampl,dtc):
        '''
        Inputs are an amplitude to test and a virtual model
        output is an virtual model with an updated dictionary.
        '''

        global model
        import quantities as pq
        import get_neab
        from neuronunit.models import backends
        from neuronunit.models.reduced import ReducedModel

        #new_file_path = str(get_neab.LEMS_MODEL_PATH)+str(int(os.getpid()))
        model = ReducedModel(get_neab.LEMS_MODEL_PATH,name=str('vanilla'),backend='NEURON')
        model.load_model()
        model.set_attrs(attrs = dtc.attrs)


        DELAY = 100.0*pq.ms
        DURATION = 1000.0*pq.ms
        params = {'injected_square_current':
                  {'amplitude':100.0*pq.pA, 'delay':DELAY, 'duration':DURATION}}


        if float(ampl) not in dtc.lookup or len(dtc.lookup)==0:

            current = params.copy()['injected_square_current']

            uc = {'amplitude':ampl}
            current.update(uc)
            current = {'injected_square_current':current}
            dtc.run_number += 1
            model.set_attrs(attrs = dtc.attrs)

            model.inject_square_current(current)
            dtc.previous = ampl
            n_spikes = model.get_spike_count()
            dtc.lookup[float(ampl)] = n_spikes
            if n_spikes == 1:
                dtc.rheobase = float(ampl)

                dtc.name = str('rheobase {0} parameters {1}'.format(str(current),str(model.params)))
                dtc.boolean = True
                return dtc

            return dtc
        if float(ampl) in dtc.lookup:
            return dtc

    #from itertools import repeat
    import numpy as np
    import copy
    import pdb
    import get_neab

    def init_dtc(dtc):
        if dtc.initiated == True:
            # expand values in the range to accomodate for mutation.
            # but otherwise exploit memory of this range.

            if type(dtc.steps) is type(float):
                dtc.steps = [ 0.75 * dtc.steps, 1.25 * dtc.steps ]
            elif type(dtc.steps) is type(list):
                dtc.steps = [ s * 1.25 for s in dtc.steps ]
            #assert len(dtc.steps) > 1
            dtc.initiated = True # logically unnecessary but included for readibility

        if dtc.initiated == False:
            import quantities as pq
            import numpy as np
            dtc.boolean = False
            steps = np.linspace(0,250,7.0)
            steps_current = [ i*pq.pA for i in steps ]
            dtc.steps = steps_current
            dtc.initiated = True
        return dtc

    def find_rheobase(dtc):
        from neuronunit.models import backends
        from neuronunit.models.reduced import ReducedModel
        import get_neab

        #new_file_path = str(get_neab.LEMS_MODEL_PATH)+str(os.getpid())
        #os.system('cp '+str(get_neab.LEMS_MODEL_PATH)+' '+new_file_path)
        model = ReducedModel(get_neab.LEMS_MODEL_PATHh,name=str('vanilla'),backend='NEURON')
        model.load_model()
        model.set_attrs(attrs = dtc.attrs)

        cnt = 0
        # If this it not the first pass/ first generation
        # then assume the rheobase value found before mutation still holds until proven otherwise.
        if type(dtc.rheobase) is not type(None):
            dtc = check_current(dtc.rheobase,dtc)
        # If its not true enter a search, with ranges informed by memory
        cnt = 0
        while dtc.boolean == False:
            for step in dtc.steps:
                dtc = check_current(step, dtc)
                dtc = check_fix_range(dtc)
                cnt+=1
        return dtc

    ## initialize where necessary.
    #import time


    '''
    want to convert these to synchronous calls to event driven asynchronous calls.
    Turning synchronous (blocking) calls into asynchronous (non-blocking) calls using even-driven programming paradigm.
    Converting nested (vertical) method calls (threads) into flat method calls
    '''

    ##
    #
    ##
    #rc[4:7]
    dtcpop = list(dview.map_sync(init_dtc,dtcpop))
    #production = rc[4:7].map(init_dtc,dtcpop).get()


    # if a population has already been evaluated it may be faster to let it
    # keep its previous rheobase searching range where this
    # memory of a previous range as acts as a guess as the next mutations range.

    dtcpop = list(dview.map_sync(find_rheobase,dtcpop))
    #dtcpop = rc[0:3].map(find_rheobase,consumption).get()


    return dtcpop, pop
