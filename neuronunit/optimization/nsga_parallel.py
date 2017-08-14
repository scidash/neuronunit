##
# Assumption that this file was executed after first executing the bash: ipcluster start -n 8 --profile=default &
##


import matplotlib # Its not that this file is responsible for doing plotting, but it calls many modules that are, such that it needs to pre-empt
# setting of an appropriate backend.
matplotlib.use('Agg')

import sys
import os
import ipyparallel as ipp
from ipyparallel import depend, require, dependent

rc = ipp.Client(profile='default')
THIS_DIR = os.path.dirname(os.path.realpath('nsga_parallel.py'))
this_nu = os.path.join(THIS_DIR,'../../')
sys.path.insert(0,this_nu)
from neuronunit import tests
from deap.benchmarks.tools import diversity, convergence, hypervolume
rc[:].use_cloudpickle()
inv_pid_map = {}
dview = rc[:]
ar = rc[:].apply_async(os.getpid)
pids = ar.get_dict()
inv_pid_map = pids
pid_map = {}

# Map PIDs onto unique numeric global identifiers via a dedicated dictionary
# this is not used here, but it could not neaten up the lack of uniqueness of file names
# in the future.
for k,v in inv_pid_map.items():
    pid_map[v] = k

with dview.sync_imports(): # Causes each of these things to be imported on the workers as well as here.
    import get_neab
    import matplotlib
    import neuronunit
    import model_parameters as modelp

    matplotlib.use('Agg') # Need to do this before importing neuronunit on a Mac, because OSX backend won't work
                          # on the worker threads.
    import pdb
    import array
    import random
    import sys

    import numpy as np
    import matplotlib.pyplot as plt
    import quantities as pq
    from deap import algorithms
    from deap import base
    from deap.benchmarks.tools import diversity, convergence, hypervolume
    from deap import creator
    from deap import tools


    import quantities as qt
    import os, sys
    import os.path

    import deap as deap
    import functools
    import utilities
    vm = utilities.VirtualModel()



    import quantities as pq
    import neuronunit.capabilities as cap
    history = tools.History()
    import numpy as np

    import sciunit
    thisnu = str(os.getcwd())+'/../..'
    sys.path.insert(0,thisnu)
    import sciunit.scores as scores




def p_imports():
    from neuronunit.models import backends
    from neuronunit.models.reduced import ReducedModel
    print(get_neab.LEMS_MODEL_PATH)
    ##
    # Necessary even with temp file.
    # workaround might involve PID map
    ##
    new_file_path = '{0}{1}'.format(str(get_neab.LEMS_MODEL_PATH),int(os.getpid()))
    print(new_file_path)

    os.system('cp ' + str(get_neab.LEMS_MODEL_PATH)+str(' ') + new_file_path)
    model = ReducedModel(new_file_path,name='vanilla',backend='NEURON')
    model.load_model()
    return

dview.apply_sync(p_imports)
p_imports()
from deap import base
from deap import creator
toolbox = base.Toolbox()

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

with dview.sync_imports():

    toolbox = base.Toolbox()
    import model_parameters as modelp
    import numpy as np
    sub_set = []
    sub_set.append(modelp.model_params['a'])
    sub_set.append(modelp.model_params['b'])

    # a subset of the parameter space. Can be interchanged with the whole parameter space
    # via artful commenting and uncommenting.

    whole_BOUND_LOW = [ np.min(i) for i in modelp.model_params.values() ]
    whole_BOUND_UP = [ np.max(i) for i in modelp.model_params.values() ]

    BOUND_LOW = [ np.min(i) for i in sub_set ]
    BOUND_UP = [ np.max(i) for i in sub_set ]

    # BOUND_LOW = whole_BOUND_LOW
    # BOUND_UP = whole_BOUND_UP


    NDIM = len(BOUND_UP)#+1
    def uniform(low, up, size=None):
        try:
            return [random.uniform(a, b) for a, b in zip(low, up)]
        except TypeError:
            return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]


    creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("select", tools.selNSGA2)


def p_imports():
    toolbox = base.Toolbox()
    import model_parameters as modelp
    import numpy as np
    sub_set = []
    sub_set.append(modelp.model_params['a'])
    sub_set.append(modelp.model_params['b'])

    # a subset of the parameter space. Can be interchanged with the whole parameter space
    # via artful commenting and uncommenting.

    whole_BOUND_LOW = [ np.min(i) for i in modelp.model_params.values() ]
    whole_BOUND_UP = [ np.max(i) for i in modelp.model_params.values() ]

    BOUND_LOW = [ np.min(i) for i in sub_set ]
    BOUND_UP = [ np.max(i) for i in sub_set ]

    # BOUND_LOW = whole_BOUND_LOW
    # BOUND_UP = whole_BOUND_UP


    NDIM = len(BOUND_UP)#+1
    def uniform(low, up, size=None):
        try:
            return [random.uniform(a, b) for a, b in zip(low, up)]
        except TypeError:
            return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]

    creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("select", tools.selNSGA2)
    return
dview.apply_sync(p_imports)

toolbox = base.Toolbox()

toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM)
toolbox.register("Individual", tools.initIterate, creator.Individual, toolbox.attr_float)
toolbox.register("population", tools.initRepeat, list, toolbox.Individual)
toolbox.register("select", tools.selNSGA2)




def evaluate(vms):#This method must be pickle-able for ipyparallel to work.
    '''
    Inputs: An individual gene from the population that has compound parameters, and a tuple iterator that
    is a virtual model object containing an appropriate parameter set, zipped togethor with an appropriate rheobase
    value, that was found in a previous rheobase search.

    outputs: a tuple that is a compound error function that NSGA can act on.

    Assumes rheobase for each individual virtual model object (vms) has already been found
    there should be a check for vms.rheobase, and if not then error.
    Inputs a gene and a virtual model object.
    outputs are error components.
    '''

    from neuronunit.models import backends
    from neuronunit.models.reduced import ReducedModel
    import quantities as pq
    import numpy as np
    import get_neab
    from itertools import repeat
    import unittest
    tc = unittest.TestCase('__init__')


    new_file_path = str(get_neab.LEMS_MODEL_PATH)+str(os.getpid())
    model = ReducedModel(new_file_path,name=str('vanilla'),backend='NEURON')
    model.load_model()
    assert type(vms.rheobase) is not type(None)
    #tests = get_neab.suite.tests
    model.update_run_params(vms.attrs)
    import copy
    tests = copy.copy(get_neab.tests)
    pre_fitness = []
    fitness = []
    for k,v in enumerate(tests):
        if k == 0:
            v.prediction = {}
            v.prediction['value'] = vms.rheobase * pq.pA

        if k != 0:
            v.prediction = None

        if k == 1 or k == 2 or k == 3:
            # Negative square pulse current.
            v.params['injected_square_current']['duration'] = 100 * pq.ms
            v.params['injected_square_current']['amplitude'] = -10 *pq.pA
            v.params['injected_square_current']['delay'] = 30 * pq.ms
        if k==0 or k == 4 or k == 5 or k == 6 or k == 7:
            # Threshold current.
            v.params['injected_square_current']['duration'] = 1000 * pq.ms
            v.params['injected_square_current']['amplitude'] = vms.rheobase * pq.pA
            v.params['injected_square_current']['delay'] = 100 * pq.ms



        if float(vms.rheobase) > 0.0:# and type(score) is not scores.InsufficientDataScore(None):
            # score needs rheobase to be at least over 0pA current injection
            # otherwise it will fail on attempt.
            score = v.judge(model,stop_on_error = False, deep_error = True)

            if 'value' in v.observation.keys():
                unit_observations = v.observation['value']

            if 'value' in v.prediction.keys():
                unit_predictions = v.prediction['value']


            if 'mean' in v.observation.keys():
                unit_observations = v.observation['mean']

            if 'mean' in v.prediction.keys():
                unit_predictions = v.prediction['mean']

            to_r_s = unit_observations.units
            unit_predictions = unit_predictions.rescale(to_r_s)

            unit_delta = np.abs( np.abs(float(unit_observations))-np.abs(float(unit_predictions)) )


            pre_fitness.append(float(unit_delta))
        if float(vms.rheobase) <=0 :
            pre_fitness.append(100.0)

    model.run_number += 1
    model.rheobase = vms.rheobase * pq.pA

    # Hybrid scheme the Genetic Algorithm
    # Make it a sum of objectives as well as
    # a Non dominated sort by summing
    # prioritized weights onto all the other weights.

    # To undo this step and substitute in normal NSGA function.
    # Substitute the block below with the one line:
    # fitness = pre_fitness
    if float(vms.rheobase) > 0:
        if unit_delta > 10.0:
            for k,f in enumerate(copy.copy(pre_fitness)):
                if k == 0:
                    fitness.append(unit_delta)
                if k != 0:
                    fitness.append(pre_fitness[k] + 1.5 * unit_delta ) # add the rheobase error to all the errors.
                    assert fitness[k] != pre_fitness[k]

            pre_fitness = []
            pre_fitness = copy.copy(fitness)
            fitness = []
        else:
            fitness = pre_fitness

        if pre_fitness[1] > 10.0 :
            for k,f in enumerate(copy.copy(pre_fitness)):
                if k == 1:
                    fitness.append(unit_delta)
                if k != 1:
                    fitness.append(pre_fitness[k] + 1.25 * f ) # add the rheobase error to all the errors.
                    assert fitness[k] != pre_fitness[k]

            pre_fitness = []
        else:
            fitness = pre_fitness


    return fitness[0],fitness[1],\
           fitness[2],fitness[3],\
           fitness[4],fitness[5],\
           fitness[6],fitness[7],

toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=30.0)
toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=30.0, indpb=1.0/NDIM)
toolbox.decorate("mate", history.decorator)
toolbox.decorate("mutate", history.decorator)
toolbox.register("select", tools.selNSGA2)
toolbox.register("map",dview.map_sync)

toolbox.register("evaluate", evaluate)




def get_trans_dict(param_dict):
    trans_dict = {}
    for i,k in enumerate(list(param_dict.keys())):
        trans_dict[i]=k
    return trans_dict
import model_parameters
param_dict = model_parameters.model_params

def vm_to_ind(vm,td):
    '''
    Re instanting Virtual Model at every update vmpop
    is Noneifying its score attribute, and possibly causing a
    performance bottle neck.
    '''

    ind =[]
    for k in td.keys():
        ind.append(vm.attrs[td[k]])
    ind.append(vm.rheobase)
    return ind



def update_vm_pop(pop, trans_dict):
    '''
    inputs a population of genes/alleles, the population size MU, and an optional argument of a rheobase value guess
    outputs a population of genes/alleles, a population of individual object shells, ie a pickleable container for gene attributes.
    Rationale, not every gene value will result in a model for which rheobase is found, in which case that gene is discarded, however to
    compensate for losses in gene population size, more gene samples must be tested for a successful return from a rheobase search.
    If the tests return are successful these new sampled individuals are appended to the population, and then their attributes are mapped onto
    corresponding virtual model objects.
    '''
    from itertools import repeat
    import numpy as np
    import copy
    pop = [toolbox.clone(i) for i in pop ]
    #import utilities
    def transform(ind):
        '''
        Re instanting Virtual Model at every update vmpop
        is Noneifying its score attribute, and possibly causing a
        performance bottle neck.
        '''
        vm = utilities.VirtualModel()

        param_dict = {}
        for i,j in enumerate(ind):
            param_dict[trans_dict[i]] = str(j)
        vm.attrs = param_dict
        vm.name = vm.attrs
        vm.evaluated = False
        return vm


    if len(pop) > 0:
        vmpop = dview.map_sync(transform, pop)
        vmpop = list(copy.copy(vmpop))
    else:
        # In this case pop is not really a population but an individual
        # but parsimony of naming variables
        # suggests not to change the variable name to reflect this.
        vmpop = transform(pop)
    return vmpop



def check_rheobase(vmpop,pop=None):
    '''
    inputs a population of genes/alleles, the population size MU, and an optional argument of a rheobase value guess
    outputs a population of genes/alleles, a population of individual object shells, ie a pickleable container for gene attributes.
    Rationale, not every gene value will result in a model for which rheobase is found, in which case that gene is discarded, however to
    compensate for losses in gene population size, more gene samples must be tested for a successful return from a rheobase search.
    If the tests return are successful these new sampled individuals are appended to the population, and then their attributes are mapped onto
    corresponding virtual model objects.
    '''
    def check_fix_range(vms):
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
        vms.rheobase = 0.0
        for k,v in vms.lookup.items():
            vm.searchedd[v]=float(k)

            if v == 1:
                #A logical flag is returned to indicate that rheobase was found.
                vms.rheobase=float(k)
                vm.searched.append(float(k))
                vms.steps = 0.0
                vms.boolean = True
                return vms
            elif v == 0:
                sub.append(k)
            elif v > 0:
                supra.append(k)

        sub=np.array(sub)
        supra=np.array(supra)

        if len(sub)!=0 and len(supra)!=0:
            #this assertion would only be wrong if there was a bug
            print(str(bool(sub.max()>supra.min())))
            assert not sub.max()>supra.min()
        if len(sub) and len(supra):
            everything = np.concatenate((sub,supra))

            center = np.linspace(sub.max(),supra.min(),7.0)
            centerl = list(center)
            # The following code block probably looks counter intuitive.
            # Its job is to delete duplicated search values.
            # Ie everything is a list of 'everything' already explored.
            # It then inserts a bias corrected center position.
            for i,j in enumerate(centerl):
                if i in list(everything):

                    np.delete(center,i)
                    del centerl[i]
                    # delete the duplicated element(s), and replace element(s)
                    # with a bias corrected
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

        vms.steps = steps
        vms.rheobase = None
        return copy.copy(vms)


    def check_current(ampl,vm):
        '''
        Inputs are an amplitude to test and a virtual model
        output is an virtual model with an updated dictionary.
        '''

        global model
        import quantities as pq
        import get_neab
        from neuronunit.models import backends
        from neuronunit.models.reduced import ReducedModel

        new_file_path = str(get_neab.LEMS_MODEL_PATH)+str(int(os.getpid()))
        model = ReducedModel(new_file_path,name=str('vanilla'),backend='NEURON')
        model.load_model()
        model.update_run_params(vm.attrs)

        DELAY = 100.0*pq.ms
        DURATION = 1000.0*pq.ms
        params = {'injected_square_current':
                  {'amplitude':100.0*pq.pA, 'delay':DELAY, 'duration':DURATION}}


        if float(ampl) not in vm.lookup or len(vm.lookup)==0:

            current = params.copy()['injected_square_current']

            uc = {'amplitude':ampl}
            current.update(uc)
            current = {'injected_square_current':current}
            vm.run_number += 1
            model.update_run_params(vm.attrs)
            model.inject_square_current(current)
            vm.previous = ampl
            n_spikes = model.get_spike_count()
            vm.lookup[float(ampl)] = n_spikes
            if n_spikes == 1:
                vm.rheobase = float(ampl)

                vm.name = str('rheobase {0} parameters {1}'.format(str(current),str(model.params)))
                vm.boolean = True
                return vm

            return vm
        if float(ampl) in vm.lookup:
            return vm

    from itertools import repeat
    import numpy as np
    import copy
    import pdb
    import get_neab

    def init_vm(vm):
        if vm.initiated == True:
            # expand values in the range to accomodate for mutation.
            # but otherwise exploit memory of this range.

            if type(vm.steps) is type(float):
                vm.steps = [ 0.75 * vm.steps, 1.25 * vm.steps ]
            elif type(vm.steps) is type(list):
                vm.steps = [ s * 1.25 for s in vm.steps ]
            #assert len(vm.steps) > 1
            vm.initiated = True # logically unnecessary but included for readibility

        if vm.initiated == False:
            import quantities as pq
            import numpy as np
            vm.boolean = False
            steps = np.linspace(0,250,7.0)
            steps_current = [ i*pq.pA for i in steps ]
            vm.steps = steps_current
            vm.initiated = True
        return vm

    def find_rheobase(vm):
        from neuronunit.models import backends
        from neuronunit.models.reduced import ReducedModel
        import get_neab
        #print(pid_map[int(os.getpid())])

        new_file_path = str(get_neab.LEMS_MODEL_PATH)+str(os.getpid())
        model = ReducedModel(new_file_path,name=str('vanilla'),backend='NEURON')
        model.load_model()
        model.update_run_params(vm.attrs)
        cnt = 0
        # If this it not the first pass/ first generation
        # then assume the rheobase value found before mutation still holds until proven otherwise.
        if type(vm.rheobase) is not type(None):
            vm = check_current(vm.rheobase,vm)
        # If its not true enter a search, with ranges informed by memory
        cnt = 0
        while vm.boolean == False:
            vm.searched.append(vm.steps)

            for step in vm.steps:
                vm = check_current(step, vm)
                vm = check_fix_range(vm)
                cnt+=1
                print(cnt)
        return vm

    ## initialize where necessary.
    #import time
    vmpop = list(dview.map_sync(init_vm,vmpop))

    # if a population has already been evaluated it may be faster to let it
    # keep its previous rheobase searching range where this
    # memory of a previous range as acts as a guess as the next mutations range.

    vmpop = list(dview.map_sync(find_rheobase,vmpop))

    return vmpop, pop




##
# Start of the Genetic Algorithm
# For good results, NGEN * MU  the number of generations
# time the size of the gene pool
# should at least be as big as number of dimensions/model parameters
# explored.
##

MU = 20
NGEN = 20
CXPB = 0.9

import numpy as np
pf = tools.ParetoFront()

stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("min", np.min, axis=0)
stats.register("max", np.max, axis=0)
stats.register("avg", np.mean)
stats.register("std", np.std)

logbook = tools.Logbook()
logbook.header = "gen", "evals", "min", "max", "avg", "std"

dview.push({'pf':pf})
trans_dict = get_trans_dict(param_dict)
td = trans_dict
dview.push({'trans_dict':trans_dict,'td':td})

pop = toolbox.population(n = MU)
pop = [ toolbox.clone(i) for i in pop ]

vmpop = update_vm_pop(pop, td)
vmpop , _ = check_rheobase(vmpop)


rh_values_unevolved = [v.rheobase for v in vmpop ]
new_checkpoint_path = str('un_evolved')+str('.p')
import pickle
with open(new_checkpoint_path,'wb') as handle:#
    pickle.dump([vmpop,rh_values_unevolved], handle)


# sometimes done in serial in order to get access to opaque stdout/stderr

fitnesses = []
#for v in vmpop:
#   fitnesses.append(evaluate_as_module.evaluate(v))
   #pdb.set_trace()
import copy
import evaluate_as_module
fitnesses = dview.map_sync(evaluate_as_module.evaluate, copy.copy(vmpop))

fitnesses = dview.map_sync(evaluate, copy.copy(vmpop))
print(fitnesses)
for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = fit


pop = tools.selNSGA2(pop, MU)
# only update the history after crowding distance has been assigned
history.update(pop)


### After an evaluation of error its appropriate to display error statistics
#pf = tools.ParetoFront()
pf.update([toolbox.clone(i) for i in pop])
hvolumes = []
hvolumes.append(hypervolume(pf))

record = stats.compile(pop)
logbook.record(gen=0, evals=len(pop), **record)
print(logbook.stream)

def rh_difference(unit_predictions):
    unit_observations = get_neab.tests[0].observation['value']
    to_r_s = unit_observations.units
    unit_predictions = unit_predictions.rescale(to_r_s)
    unit_delta = np.abs( np.abs(unit_observations)-np.abs(unit_predictions) )
    print(unit_delta)
    return float(unit_delta)

verbose = True
means = np.array(logbook.select('avg'))
gen = 1
rh_mean_status = np.mean([ v.rheobase for v in vmpop ])
rhdiff = rh_difference(rh_mean_status * pq.pA)
print(rhdiff)
verbose = True
while (gen < NGEN and means[-1] > 0.05):
    # Although the hypervolume is not actually used here, it can be used
    # As a terminating condition.
    hvolumes.append(hypervolume(pf))
    gen += 1
    print(gen)
    offspring = tools.selNSGA2(pop, len(pop))
    if verbose:
        for ind in offspring:
            print('what do the weights without values look like? {0}'.format(ind.fitness.weights[:]))
            print('what do the weighted values look like? {0}'.format(ind.fitness.wvalues[:]))
            #print('has this individual been evaluated yet? {0}'.format(ind.fitness.valid[0]))
            print(rhdiff)
    offspring = [toolbox.clone(ind) for ind in offspring]

    for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
        if random.random() <= CXPB:
            toolbox.mate(ind1, ind2)
        toolbox.mutate(ind1)
        toolbox.mutate(ind2)
        # deleting the fitness values is what renders them invalid.
        # The invalidness is used as a flag for recalculating them.
        # Their fitneess needs deleting since the attributes which generated these values have been mutated
        # and hence they need recalculating
        # Mutation also implies breeding, if a gene is mutated it was also recently recombined.
        del ind1.fitness.values, ind2.fitness.values

    invalid_ind = []
    for ind in offspring:
        if ind.fitness.valid == False:
            invalid_ind.append(ind)
    # Need to make sure that update_vm_pop does not replace instances of the same model
    # Thus waisting computation.
    vmoffspring = update_vm_pop(copy.copy(invalid_ind), trans_dict) #(copy.copy(invalid_ind), td)
    vmoffspring , _ = check_rheobase(copy.copy(vmoffspring))
    rh_mean_status = np.mean([ v.rheobase for v in vmoffspring ])
    rhdiff = rh_difference(rh_mean_status * pq.pA)
    print('the difference: {0}'.format(rh_difference(rh_mean_status * pq.pA)))
    # sometimes fitness is assigned in serial, although slow gives access to otherwise hidden
    # stderr/stdout
    # fitnesses = []
    # for v in vmoffspring:
    #    fitness.append(evaluate(v))
    #import evaluate_as_module
    fitnesses = list(dview.map_sync(evaluate_as_module.evaluate, copy.copy(vmoffspring)))
    #fitnesses = list(dview.map_sync(evaluate, copy.copy(vmoffspring)))

    mf = np.mean(fitnesses)

    for ind, fit in zip(copy.copy(invalid_ind), fitnesses):
        ind.fitness.values = fit
        if verbose:
            print('what do the weights without values look like? {0}'.format(ind.fitness.weights))
            print('what do the weighted values look like? {0}'.format(ind.fitness.wvalues))
            print('has this individual been evaluated yet? {0}'.format(ind.fitness.valid))

    # Its possible that the offspring are worse than the parents of the penultimate generation
    # Its very likely for an offspring population to be less fit than their parents when the pop size
    # is less than the number of parameters explored. However this effect should stabelize after a
    # few generations, after which the population will have explored and learned significant error gradients.
    # Selecting from a gene pool of offspring and parents accomodates for that possibility.
    # There are two selection stages as per the NSGA example.
    # https://github.com/DEAP/deap/blob/master/examples/ga/nsga2.py
    # pop = toolbox.select(pop + offspring, MU)

    # keys = history.genealogy_tree.keys()
    # Optionally
    # Grab evaluated history items and chuck them into the mixture.
    # This may cause stagnation of evolution however.
    # We want to select among the best from the whole history of the GA, not just penultimate and present generations.
    # all_hist = [ history.genealogy_history[i] for i in keys if history.genealogy_history[i].fitness.valid == True ]
    # pop = tools.selNSGA2(offspring + all_hist, MU)

    pop = tools.selNSGA2(offspring + pop, MU)

    record = stats.compile(pop)
    history.update(pop)

    logbook.record(gen=gen, evals=len(pop), **record)
    pf.update([toolbox.clone(i) for i in pop])
    means = np.array(logbook.select('avg'))
    pf_mean = np.mean([ i.fitness.values for i in pf ])


    # if the means are not decreasing at least as an overall trend something is wrong.
    print('means from logbook: {0} from manual meaning the fitness: {1}'.format(means,mf))
    print('means: {0} pareto_front first: {1} pf_mean {2}'.format(logbook.select('avg'), \
                                                        np.sum(np.mean(pf[0].fitness.values)),\
                                                        pf_mean))

import net_graph
net_graph.surfaces(history,td)
best, worst = net_graph.best_worst(history)
listss = [best , worst]
best_worst = update_vm_pop(listss,td)
best_worst , _ = check_rheobase(best_worst)
rheobase_values = [v.rheobase for v in vmoffspring ]
vmhistory = update_vm_pop(history.genealogy_history.values(),td)

import pickle
with open('complete_dump.p','wb') as handle:
   pickle.dump([vmoffspring,history,logbook,rheobase_values,best_worst,vmhistory,hvolumes],handle)

lists = pickle.load(open('complete_dump.p','rb'))
#vmoffspring2,history2,logbook2 = lists[0],lists[1],lists[2]
net_graph.surfaces(history,td)
import net_graph
#reload(net_graph)
#vmhistory = update_vm_pop(history.genealogy_history.values(),td)
#best, worst = net_graph.best_worst(history)
#listss = [best , worst]
#best_worst = update_vm_pop(listss,td)
#best_worst , _ = check_rheobase(best_worst)
best = vm
unev = pickle.load(open('un_evolved.p','rb'))
unev, rh_values_unevolved = unev[0], unev[1]
for x,y in enumerate(unev):
    y.rheobase = rh_values_unevolved[x]
vmoffpsring.append(unev)
net_graph.shadow(vmoffspring,best_worst[0])
net_graph.plotly_graph(history,vmhistory)
#net_graph.graph_s(history)
net_graph.plot_log(logbook)
net_graph.not_just_mean(hvolumes,logbook)
net_graph.plot_objectives_history(logbook)

#Although the pareto front surely contains the best candidate it cannot contain the worst, only history can.
#best_ind_dict_vm = update_vm_pop(pf[0:2],td)
#best_ind_dict_vm , _ = check_rheobase(best_ind_dict_vm)



print(best_worst[0].attrs,' == ', best_ind_dict_vm[0].attrs, ' ? should be the same (eyeball)')
print(best_worst[0].fitness.values,' == ', best_ind_dict_vm[0].fitness.values, ' ? should be the same (eyeball)')

# This operation converts the population of virtual models back to DEAP individuals
# Except that there is now an added 11th dimension for rheobase.
# This is not done in the general GA algorithm, since adding an extra dimensionality that the GA
# doesn't utilize causes a DEAP error, which is reasonable.

net_graph.bar_chart(best_worst[0])
net_graph.pca(final_population,vmpop,fitnesses,td)
#test_dic = bar_chart(best_worst[0])
net_graph.plot_evaluate( best_worst[0],best_worst[1])
#net_graph.plot_db(best_worst[0],name='best')
#net_graph.plot_db(best_worst[1],name='worst')

net_graph.plot_evaluate( best_worst[0],best_worst[1])
net_graph.plot_db(best_worst[0],name='best')
net_graph.plot_db(best_worst[1],name='worst')
