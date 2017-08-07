##
# Assumption that this file was executed after first executing the bash: ipcluster start -n 8 --profile=default &
##


import matplotlib # Its not that this file is responsible for doing plotting, but it calls many modules that are, such that it needs to pre-empt
# setting of an appropriate backend.
try:
    matplotlib.use('Qt5Agg')
except:
    matplotlib.use('Agg')

import sys
import os
import ipyparallel as ipp
from ipyparallel import depend, require, dependent


import cProfile
import atexit
import os,sys




#if ipython:
#%load_ext autoreload
#%autoreload 2

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
rc = ipp.Client(profile='default')
THIS_DIR = os.path.dirname(os.path.realpath('nsga_parallel.py'))
this_nu = os.path.join(THIS_DIR,'../../')
sys.path.insert(0,this_nu)
from neuronunit import tests
from deap.benchmarks.tools import diversity, convergence, hypervolume
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
    try:
        matplotlib.use('Qt5Agg') # Need to do this before importing neuronunit on a Mac, because OSX backend won't work
    except:
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
    #print(get_neab.LEMS_MODEL_PATH)
    new_file_path = '{0}{1}'.format(str(get_neab.LEMS_MODEL_PATH),int(os.getpid()))
    #print(new_file_path)

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
        self.fitness = creator.FitnessMax

with dview.sync_imports():

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
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("select", tools.selNSGA2)


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
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("select", tools.selNSGA2)
    return
dview.apply_sync(p_imports)

BOUND_LOW = [ np.min(i) for i in modelp.model_params.values() ]
BOUND_UP = [ np.max(i) for i in modelp.model_params.values() ]
NDIM = len(BOUND_UP)+1 #One extra to store rheobase values in.

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
            v.params['injected_square_current']['duration'] = 1000 * pq.ms
            v.params['injected_square_current']['amplitude'] = vms.rheobase * pq.pA
            v.params['injected_square_current']['delay'] = 100 * pq.ms
        if k != 0:
            v.prediction = None

        if k == 1 or k == 2 or k == 3:
            # Negative square pulse current.
            v.params['injected_square_current']['duration'] = 100 * pq.ms
            v.params['injected_square_current']['amplitude'] = -10 *pq.pA
            v.params['injected_square_current']['delay'] = 30 * pq.ms
        if k == 5 or k == 6 or k == 7:
            # Threshold current.
            v.params['injected_square_current']['duration'] = 1000 * pq.ms
            v.params['injected_square_current']['amplitude'] = vms.rheobase * pq.pA
            v.params['injected_square_current']['delay'] = 100 * pq.ms


        score = v.judge(model,stop_on_error = False, deep_error = True)
        #v.descripition()
        #import pdb; pdb.set_trace()
        if k == 0 and float(vms.rheobase) > 0:# and type(score) is not scores.InsufficientDataScore(None):
            if 'value' in v.observation.keys():
                unit_observations = v.observation['value']

            if 'value' in v.prediction.keys():
                unit_predictions = v.prediction['value']

            to_r_s = unit_observations.units
            unit_predictions = unit_predictions.rescale(to_r_s)

            unit_delta = np.abs( np.abs(float(unit_observations))-np.abs(float(unit_predictions)) )
            print(float(vms.rheobase),float(unit_predictions))
            assert float(vms.rheobase) == float(unit_predictions)
            diff = np.abs(np.abs(float(unit_observations)) - np.abs(float(vms.rheobase)))
            print(unit_delta, diff, float(unit_observations))
            assert unit_delta == diff

            pre_fitness.append(float(unit_delta))
        elif k == 0 and float(vms.rheobase) <=0 :
            pre_fitness = 100.0
        else:
            pre_fitness.append(float(score.sort_key))

    model.run_number += 1
    model.rheobase = vms.rheobase * pq.pA


    for k,f in enumerate(pre_fitness):
        if k == 0:
            fitness.append(unit_delta)
        if k != 0:
            fitness.append(pre_fitness[k] + 1.5 *unit_delta ) # add the rheobase error to all the errors.
            assert fitness[k] != pre_fitness[k]

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
            # The following code block probably looks counter intuitive.
            # Its job is to delete duplicated search values.
            # Ie everything is a list of everything already explored.
            # It then makes a corrected center position.
            for i,j in enumerate(centerl):
                if i in list(everything):

                    np.delete(center,i)
                    del centerl[i]
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
            steps = np.linspace(-50,200,7.0)
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
# Exhaustive search.
##
# Pick out just two parameters.
# Do all of this in a big loop

for k in range(1,9):
    quads = []
    for i,j in enumerate(td):
        print(i,k)
        if i+k < 10:
            quads.append((td[i],td[i+k],i,i+k))

    def model2map(iter_value):#This method must be pickle-able for scoop to work.

        vm = utilities.VirtualModel()
        (x,y,w,z) = iter_value
        vm.attrs = {}
        vm.attrs[str(x)] = w
        vm.attrs[str(y)] = z
        return vm
    vmpop1 = list(dview.map_sync(model2map,quads))
    vmpop1 , _ = check_rheobase(vmpop1)
'''
for q in quads:
    print(k)
    (x,y,w,z) = q
    print(x,y,w,z,i)

    import model_parameters as modelp

    iter_list = [ (i,j) for i in modelp.model_params['a'] for j in modelp.model_params['b'] ]
'''



    new_checkpoint_path = str('rh_checkpoint_exhaustive')+str('.p')
    import pickle
    with open(new_checkpoint_path,'wb') as handle:#
        pickle.dump(vmpop1, handle)

    print([ (v.rheobase,v.attrs) for v in vmpop1])

    import copy
    efitnesses = dview.map_sync(evaluate, copy.copy(vmpop1))

    import pickle
    with open('complete_exhaust.p','wb') as handle:
       pickle.dump([efitnesses,iter_list,vmpop1],handle)


    lists = pickle.load(open('complete_exhaust.p','rb'))
    efitnesses,iter_value,vmpop1 = lists[2],lists[1],lists[0]
    matrix_fill = [ (i,j) for i in range(0,len(modelp.model_params['a'])) for j in range(0,len(modelp.model_params['b'])) ]
    mf = list(zip(matrix_fill,efitnesses))
    #pixels = list(zip(iter_list,efitnesses))

    #empty = np.array(len(pixels),len(pixels))
    #empty = np.zeros(shape=(len(mf)+1,len(mf)+1))
    empty = np.zeros(shape=(int(np.sqrt(len(mf))),int(np.sqrt(len(mf)))))

    def fitness2map(pixels,dfimshow):
        for i in pixels:
        #for i,j,k in pixels:
            dfimshow[i[0][0],i[0][1]]=np.sum(i[1])
        return dfimshow
    dfimshow =fitness2map(mf,empty)


    from matplotlib import pylab
    import numpy
    from matplotlib.colors import LogNorm
    plt.clf()

    pylab.imshow(dfimshow, interpolation="nearest", origin="upper", aspect="equal", norm=LogNorm())
    cbar = pylab.colorbar()
    plt.savefig('2d_error_surface.png')


import net_graph
from IPython.lib.deepreload import reload
reload(net_graph)
'''
vmhistory = update_vm_pop(history.genealogy_history.values(),td)
net_graph.plotly_graph(history,vmhistory)
#net_graph.graph_s(history)
net_graph.plot_log(logbook)
net_graph.just_mean(logbook)
net_graph.plot_objectives_history(logbook)

#Although the pareto front surely contains the best candidate it cannot contain the worst, only history can.
best_ind_dict_vm = update_vm_pop(pf[0:2],td)
best_ind_dict_vm , _ = check_rheobase(best_ind_dict_vm)

best, worst = net_graph.best_worst(history)
listss = [best , worst]
best_worst = update_vm_pop(listss,td)
best_worst , _ = check_rheobase(best_worst)

print(best_worst[0].attrs,' == ', best_ind_dict_vm[0].attrs, ' ? should be the same (eyeball)')
print(best_worst[0].fitness.values,' == ', best_ind_dict_vm[0].fitness.values, ' ? should be the same (eyeball)')

# This operation converts the population of virtual models back to DEAP individuals
# Except that there is now an added 11th dimension for rheobase.
# This is not done in the general GA algorithm, since adding an extra dimensionality that the GA
# doesn't utilize causes a DEAP error, which is reasonable.

net_graph.plot_evaluate( best_worst[0],best_worst[1])
net_graph.plot_db(best_worst[0],name='best')
net_graph.plot_db(best_worst[1],name='worst')
'''
