
#Assumption that this file was executed after first executing the bash: ipcluster start -n 8 --profile=default &
import sys
import os
import ipyparallel as ipp
from ipyparallel import depend, require, dependent
#import networkx
#import graphviz

import cProfile
import atexit
import os,sys
def ProfExit(p):
   '''
   http://seiferteric.com/?p=277
   So what does this do? It imports the profiler and the atexit module.
   It creates an instance of the profiler, registers with atexit to stop the profiler and dump the stats
   to a file named with the process
   ID of that python process, and finally starts the profiler.
   So every python process run on the system will now be profiled! FYI,
   the stats won’t get dumped until the process exits,
   so make sure you stop all of them.
   '''
   p.disable()
   prof_f_name = '{0}'.format(os.getpid())
   #Open and close the file, as a quick and dirty way to confirm that exists.
   f = open('NeuroML2/%s'%prof_f_name,'wb')
   f.close()

   p.dump_stats('NeuroML2/%s'%prof_f_name)
profile_hook = cProfile.Profile()
atexit.register(ProfExit, profile_hook)
profile_hook.enable()
#from networkx.drawing.nx_agraph import graphviz_layout

#from ipyparallel.apps import iploggerapp
rc = ipp.Client(profile='default')
THIS_DIR = os.path.dirname(os.path.realpath('nsga_parallel.py'))
this_nu = os.path.join(THIS_DIR,'../../')
sys.path.insert(0,this_nu)
from neuronunit import tests
rc[:].use_cloudpickle()
inv_pid_map = {}
dview = rc[:]
lview = rc.load_balanced_view()
ar = rc[:].apply_async(os.getpid)
pids = ar.get_dict()
inv_pid_map = pids
pid_map = {}

#Map PIDs onto unique numeric global identifiers via a dedicated dictionary
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
    print(utilities.__file__)
    vm = utilities.VirtualModel()
    #print(vm.__init__())
    print(vm.delta)


    import quantities as pq
    import neuronunit.capabilities as cap
    history = tools.History()
    import numpy as np

    import sciunit
    thisnu = str(os.getcwd())+'/../..'
    sys.path.insert(0,thisnu)
    import sciunit.scores as scores

    #from itertools import repeat



def p_imports():
    from neuronunit.models import backends
    from neuronunit.models.reduced import ReducedModel
    print(get_neab.LEMS_MODEL_PATH)
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
    creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMin)


def p_imports():
    toolbox = base.Toolbox()
    import model_parameters as modelp
    import numpy as np
    BOUND_LOW = [ np.min(i) for i in modelp.model_params.values() ]
    BOUND_UP = [ np.max(i) for i in modelp.model_params.values() ]
    NDIM = len(BOUND_UP)+1
    def uniform(low, up, size=None):
        try:
            return [random.uniform(a, b) for a, b in zip(low, up)]
        except TypeError:
            return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]
    creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM)
    #toolbox.register("Individual", tools.initIterate, creator.Individual, toolbox.attr_float)
    #toolbox.register("population", tools.initRepeat, list, toolbox.Individual)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("select", tools.selNSGA2)
    return
dview.apply_sync(p_imports)

BOUND_LOW = [ np.min(i) for i in modelp.model_params.values() ]
BOUND_UP = [ np.max(i) for i in modelp.model_params.values() ]
NDIM = len(BOUND_UP)

def uniform(low, up, size=None):
    try:
        return [random.uniform(a, b) for a, b in zip(low, up)]
    except TypeError:
        return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]
toolbox = base.Toolbox()

toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM)
toolbox.register("Individual", tools.initIterate, creator.Individual, toolbox.attr_float)
toolbox.register("population", tools.initRepeat, list, toolbox.Individual)
toolbox.register("select", tools.selNSGA2)

def prepare_tests(tests,vms):
    for k,v in enumerate(tests):
        if k == 0:
            v.prediction = {}
            v.prediction['value'] = vms.rheobase * pq.pA
            print(v.prediction)

        if k == 1 or k == 2 or k == 3:
            # Negative square pulse current.
            v.params['injected_square_current']['duration'] = 100 * pq.ms
            v.params['injected_square_current']['amplitude'] = -10 *pq.pA
            v.params['injected_square_current']['delay'] = 30 * pq.ms
        if k == 0 or k == 5 or k == 6 or k == 7:
            # Threshold current.
            v.params['injected_square_current']['duration'] = 1000 * pq.ms
            v.params['injected_square_current']['amplitude'] = vms.rheobase * pq.pA
            v.params['injected_square_current']['delay'] = 100 * pq.ms
            v.prediction = {}
            v.prediction['value'] = vms.rheobase * pq.pA
            print(v.prediction,k)
        print(v.params)


            #plt.plot(obs,k)
            #plt.plot(pre,k)
        #plt.xlabel('observations vs predictions')
        #plt.ylabel('test number')
    #plt.savefig('observation_vs_prediction.eps'.format(os.pid,j,vms.run_number))

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
    def plot_db(tests,pre,vms):
        '''
        A method to plot raw predictions
        versus observations
        '''

        import matplotlib.pyplot as plt
        import numpy as np
        plt.clf()
        matplotlib.use('Agg') # Need to do this before importing neuronunit on a Mac, because OSX backend won't work
        matplotlib.style.use('ggplot')
        vms.delta[vms.run_number] = []
        print(pre)
        for iterator,returns in enumerate(pre):
            for predictions in returns.values():
                unit_predictions = predictions
            print('observations: {0} predictions: {1}'.format(tests[iterator].observation, returns.values()))
            if 'mean' in tests[iterator].observation.keys():
                obs = tests[iterator].observation['mean']
                unit_observations = obs
            else:
                for obs in tests[iterator].observation.values():
                    unit_observations = obs
            # make sure unit order of magnitude matches, by rescaling one to the other:
            to_r_s = unit_observations.units
            print(to_r_s)
            unit_delta = np.abs(unit_observations-unit_predictions.rescale(to_r_s))
            print('unit delta', unit_delta)
            vms.delta[vms.run_number].append(unit_delta)
            #if iterator!=1:
            plt.scatter(iterator,unit_delta)
        labels = tuple([str(t.describe()) for t in tests ])
        tick_locations = tuple(range(0,len(tests)))
        plt.xticks(tick_locations , labels)
        plt.xticks(rotation=70)
        plt.xlabel('test type')
        plt.ylabel('observation versus prediction')
        plt.savefig('obsevation_versus_prediction_{0}_.eps'.format(vms.run_number))
        return vms
    #vms = plot_db(get_neab.suite.tests,new_preds,vms)
    #plot_db(get_neab.suite.tests,pre,vms)
    from neuronunit.models import backends
    from neuronunit.models.reduced import ReducedModel
    import quantities as pq
    import numpy as np
    import get_neab
    from itertools import repeat

    new_file_path = str(get_neab.LEMS_MODEL_PATH)+str(os.getpid())
    model = ReducedModel(new_file_path,name=str(vms.attrs),backend='NEURON')
    model.load_model()
    assert type(vms.rheobase) is not type(None)
    model.update_run_params(vms.attrs)
    tests = get_neab.suite.tests
    prepare_tests(get_neab.suite.tests,vms)
    # a performance hit but one that will change.
    #pre = dview.map(repeat(model))

    model.update_run_params(vms.attrs)
    score = get_neab.suite.judge(model, stop_on_error = False, deep_error = True)
    pre = [ t.generate_prediction(model) for t in tests ]
    #new_preds = [ t.prediction for t in tests ]
    #print(new_preds)
    #assert.equals(pre,new_preds)
    #delta = dview.map_sync(plot_db,get_neab.suite.tests,new_preds,repeat(vms))
    vms = plot_db(get_neab.suite.tests,pre,vms)
    print(score)
    model.run_number += 1
    # Run the model, then:
    error = []
    vms.score = []
    #other_mean = np.mean([i for i in score.sort_key.values.tolist()[0] if type(i) is not type(None)])

    for my_score in score.sort_key.values.tolist()[0]:
        assert type(my_score) is not type(dict)
        error.append(my_score)
    vms.evaluated = True
    error = [ -1.0 * e for e in error ]
    vms.error = error
    return error[0],error[1],error[2],error[3],error[4],error[5],error[6],error[7],


toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM)
toolbox.decorate("mate", history.decorator)
toolbox.decorate("mutate", history.decorator)
toolbox.register("select", tools.selNSGA2)
toolbox.register("map",dview.map_sync)

toolbox.register("evaluate", evaluate)

def sum_over(individual):
    return sum(individual),

toolbox.register("sum_over", sum_over)


def plot_ss(vmlist,gen):
    '''
    '''
    import matplotlib.pyplot as plt
    plt.clf()
    for ind,j in enumerate(vmlist):
        if hasattr(ind,'results'):
            plt.plot(ind.results['t'],ind.results['vm'],label=str(vmlist[j].attr) )
            #plt.xlabel(str(vmlist[j].attr))
    plt.savefig('snap_shot_at_gen_'+str(gen)+'.png')
    plt.clf()


def get_trans_dict(param_dict):
    trans_dict = {}
    for i,k in enumerate(list(param_dict.keys())):
        trans_dict[i]=k
    return trans_dict
import model_parameters
param_dict = model_parameters.model_params

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
        print(dir(vm))
        print(vm.delta)
        param_dict = {}
        for i,j in enumerate(ind):
            param_dict[trans_dict[i]] = str(j)
        vm.attrs = param_dict
        vm.name = vm.attrs
        vm.evaluated = False
        return vm

    if len(pop) > 1:
        vmpop = dview.map_sync(transform, pop)
        vmpop = list(copy.copy(vmpop))
    else:
        # In this case pop is not really a population but an individual
        # but parsimony of naming variables
        # suggests not to change the variable name to reflect this.
        vmpop = transform(pop)
    return vmpop


def update_vm_existing(pop, vmpop, trans_dict):
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

    def transform(ind,vm):
        '''
        Re instanting Virtual Model at every update vmpop
        is Noneifying its score attribute, and possibly causing a
        performance bottle neck.
        '''

        param_dict = {}
        for i,j in enumerate(ind):
            param_dict[trans_dict[i]] = str(j)
        vm.attrs = param_dict
        vm.name = vm.attrs
        vm.evaluated = False
        return vm

    if len(pop) > 1:
        vmpop = dview.map_sync(transform, pop, vmpop)
        vmpop = list(copy.copy(vmpop))
    else:
        # In this case pop is not really a population but an individual
        # but parsimony of naming variables
        # suggests not to change the variable name to reflect this.
        vmpop = transform(pop,vmpop)
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
        vms.rheobase=0.0
        for k,v in vms.lookup.items():
            if v==1:
                #A logical flag is returned to indicate that rheobase was found.
                vms.rheobase=float(k)
                vms.steps = 0.0
                vms.boolean = True
                return vms
            elif v==0:
                sub.append(k)
            elif v>0:
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
            for i,j in enumerate(centerl):
                if i in list(everything):
                    np.delete(center,i)
                    del centerl[i]
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

        model = ReducedModel(new_file_path,name=str(vm.attrs),backend='NEURON')
        model.load_model()
        model.update_run_params(vm.attrs)

        DELAY = 100.0*pq.ms
        DURATION = 1000.0*pq.ms
        params = {'injected_square_current':
                  {'amplitude':100.0*pq.pA, 'delay':DELAY, 'duration':DURATION}}


        print('print model name {0}'.format(model.name))
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
                print(type(vm.rheobase))
                print('current {0} spikes {1}'.format(vm.rheobase,n_spikes))
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
            vm.steps = [ s * 1.25 for s in vm.steps ]
            vm.initiated = True # logically unnecessary but included for readibility

        if vm.initiated == False:
            import quantities as pq
            import numpy as np
            vm.boolean = False
            steps = list(np.linspace(-50,200,7.0))
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
        model = ReducedModel(new_file_path,name=str(vm.attrs),backend='NEURON')
        model.load_model()
        model.update_run_params(vm.attrs)
        cnt = 0
        # If this it not the first pass/ first generation
        # then assume the rheobase value found before mutation still holds until proven otherwise.
        if type(vm.rheobase) is not type(None):
            vm = check_current(vm.rheobase,vm)
        # If its not true enter a search, with ranges informed by memory
        while vm.boolean == False:
            for step in vm.steps:
                vm = check_current(step, vm)
                vm = check_fix_range(vm)
                cnt+=1
        return vm

    ## initialize where necessary.
    import time
    vmpop = list(dview.map_sync(init_vm,vmpop))

    # if a population has already been evaluated it may be faster to let it
    # keep its previous rheobase searching range where this
    # memory of a previous range as acts as a guess as the next mutations range.

    vmpop = list(dview.map_sync(find_rheobase,vmpop))
    #vmpop = list(vmpop)
    if type(pop) is not type(None):
        vmpop, pop = final_check(vmpop,pop)
    return vmpop, pop




##
# Start of the Genetic Algorithm
##

MU = 5
NGEN = 5
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
history.update(pop)
pop = [ toolbox.clone(i) for i in pop ]
vmpop = update_vm_pop(pop, td)
print(type(vmpop[0]))
print(dir(vmpop[0]))
#print(vmpop[0].__init__())
print(vmpop[0].delta)
tests = get_neab.suite.tests
for t in tests:
    print(t.observation, t.describe())

vmpop , _ = check_rheobase(vmpop)
for i in vmpop:
    print('the rheobase value is {0}'.format(i.rheobase))


new_checkpoint_path = str('rh_checkpoint')+str('.p')
import pickle
with open(new_checkpoint_path,'wb') as handle:
    pickle.dump({'vmpop':vmpop,'pop':pop}, handle)
#cp = pickle.load(open(new_checkpoint_path,'rb'))




import copy
fitnesses = toolbox.map(toolbox.evaluate, copy.copy(vmpop))
for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = fit

history.update(pop)
pop = tools.selNSGA2(pop, MU)

### After an evaluation of error its appropriate to display error statistics
#pf = tools.ParetoFront()
pf.update([toolbox.clone(i) for i in pop])
record = stats.compile(pop)
logbook.record(gen=0, evals=len(pop), **record)
print(logbook.stream)

score_storage = [ v.score for v in vmpop ]

time_out = 0

for gen in range(1, NGEN):
    offspring = tools.selTournamentDCD(pop, len(pop))
    #assert len(offspring)!=0
    offspring = [toolbox.clone(ind) for ind in offspring]
    #assert len(offspring)!=0
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
        if len(ind.fitness.values) == 0:
            invalid_ind.append(ind)
    # Need to make sure that update_vm_pop does not replace instances of the same model
    # Thus waisting computation.
    vmoffspring = update_vm_existing(copy.copy(invalid_ind), vmpop, trans_dict) #(copy.copy(invalid_ind), td)
    vmoffspring , _ = check_rheobase(copy.copy(vmoffspring))

    fitnesses = list(toolbox.map(toolbox.evaluate, copy.copy(vmoffspring)))


    for ind, fit in zip(copy.copy(offspring), fitnesses):
        ind.fitness.values = fit

    # Its possible that the offspring are worse than the parents of the penultimate generation
    # Selecting from a gene pool of offspring and parents accomodates for that possibility.
    # There are two selection stages as per the NSGA example.
    # https://github.com/DEAP/deap/blob/master/examples/ga/nsga2.py
    #pop = toolbox.select(pop + offspring, MU)
    pop = tools.selNSGA2(pop + offspring, MU)
    history.update(pop)
    pf.update([toolbox.clone(i) for i in pop])
    record = stats.compile([toolbox.clone(i) for i in pop])
    logbook.record(gen=gen, evals=len(pop), **record)
    means = np.array(logbook.select('avg'))
    # if the means are not decreasing at least as an overall trend something is wrong.
    print('means: {0}'.format(means))
    if means[-1] <= means[-2] or means[-1] < means[0]:
        time_out = 0

    # if the lasest mean error is greater than second latest mean error.
    # or if the latest mean error is greater than the initial mean error.
    # Then the error might be growing not shrinking.
    if means[-1] > means[-2] or means[-1] > means[1]:
        time_out += 1
        print('time out: {0}'.format(time_out))
        assert time_out < 3
    print('means: {0} std: {1} max: {2} min: {3}'.format(logbook.select('avg'), \
                                                        logbook.select('std'), \
                                                        logbook.select('max'), \
                                                        logbook.select('min')))
    assert len(vmoffspring) != 0


with open('complete_dump.p','wb') as handle:
   pickle.dump([vmpop,pop,pf,history,logbook],handle)



rhstorage = [ v.rheobase for v in copy.copy(vmoffspring) ]
score_storage = [ np.sum(v.score) for v in copy.copy(vmoffspring) ]
score_storage_sum = [ np.sum(v.score) for v in copy.copy(vmoffspring) ]


import net_graph
net_graph.plot_log(logbook)
net_graph.just_mean(logbook)
plt.clf()
import networkx
graph = networkx.DiGraph(history.genealogy_tree)
graph = graph.reverse()
labels ={}
for i in graph.nodes():
    labels[i] = i
node_colors = [ np.sum(history.genealogy_history[i].fitness.values) for i in graph ]
positions = graphviz_layout(graph, prog="dot")
networkx.draw(graph, positions, node_color=node_colors, labels = labels)
nodes=networkx.draw_networkx_nodes(graph,positions,node_color=node_colors)
edges=networkx.draw_networkx_edges(graph,positions,width=0.5,edge_cmap=plt.cm.Blues)
plt.sci(nodes)
plt.colorbar()
plt.sci(edges)
plt.savefig('genealogy_history_{0}_.eps'.format('4343'))
print(rhstorage)
print(score_storage)
print('unclear why vmoffspring: {0} and pareto front:  {1} of different lengths'.format(len(vmoffspring),len(pf)))
print(pf)
import pandas as pd
scores = []
for vm in vmoffspring:
    scores.append((vm.attrs,vm.score))
    print(vm.score)

sc = pd.DataFrame(scores[0])
sc
data = [ pf[0] ]
model_values0 = pd.DataFrame(data)
model_values0
rhstorage[0]

data = [ pf[1] ]
model_values0 = pd.DataFrame(data)
model_values0



sc1 = pd.DataFrame(scores[1])
sc1

rhstorage[1]

data = [ pf[1].attrs ]
model_values1 = pd.DataFrame(data)
model_values1


import pickle
import pandas as pd

try:
    ground_error = pickle.load(open('big_model_evaulated.pickle','rb'))
except:
    # The exception code is only skeletal, it would not actually work, but its the right principles.
    print('{0} it seems the error truth data does not yet exist, lets create it now '.format(str(False)))
    ut = utilities.Utilities

    ground_error = list(dview.map_sync(ut.func2map, ground_truth))
    pickle.dump(ground_error,open('big_model_evaulated.pickle','wb'))

# ground_error_nsga=list(zip(vmpop,pop,invalid_ind))
# pickle.dump(ground_error_nsga,open('nsga_evaulated.pickle','wb'))

sum_errors = [ i[0] for i in ground_error ]
composite_errors = [ i[1] for i in ground_error ]
attrs = [ i[2] for i in ground_error ]
rheobase = [ i[3] for i in ground_error ]

indexs = [i for i,j in enumerate(sum_errors) if j==np.min(sum_errors) ][0]
indexc = [i for i,j in enumerate(composite_errors) if j==np.min(composite_errors) ][0]

df_0 = pd.DataFrame([ (k,v,vmpop[0].attrs[k],float(v)-float(vmpop[0].attrs[k])) for k,v in ground_error[indexc][2].items() ])
df_1 = pd.DataFrame([ (k,v,vmpop[1].attrs[k],float(v)-float(vmpop[1].attrs[k])) for k,v in ground_error[indexc][2].items() ])


df_0

df_1
