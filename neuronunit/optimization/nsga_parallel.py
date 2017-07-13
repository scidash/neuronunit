
#Assumption that this command was called via bash: ipcluster start -n 8 --profile=default &
import sys
import os
import ipyparallel as ipp
rc = ipp.Client(profile='default');
THIS_DIR = os.path.dirname(os.path.realpath('nsga_parallel.py'))
this_nu = os.path.join(THIS_DIR,'../../')
print(os.getcwd())
sys.path.insert(0,this_nu)
print(this_nu)

#rc[:].use_cloudpickle()
#dview = rc[0:-2]
dview = rc[:]

with dview.sync_imports(): # Causes each of these things to be imported on the workers as well as here.
#def p_imports():
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
    import os
    import os.path

    import deap as deap
    import functools
    import utilities


    import quantities as pq
    import neuronunit.capabilities as cap
    history = tools.History()
    #import neuronunit.optimization.model_parameters as modelp
    import numpy as np

    import quantities as pq
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    import sciunit
    import os, sys
    thisnu = str(os.getcwd())+'/../..'
    sys.path.insert(0,thisnu)
    import sciunit.scores as scores
    import neuronunit.capabilities as cap



    from scipy.optimize import curve_fit
    required_capabilities = (cap.ReceivesSquareCurrent,
                             cap.ProducesSpikes)
    #params = {'injected_square_current':
    #            {'amplitude':100.0*pq.pA, 'delay':DELAY, 'duration':DURATION}}
    name = "Rheobase test"
    description = ("A test of the rheobase, i.e. the minimum injected current "
                   "needed to evoke at least one spike.")
    score_type = scores.RatioScore
    lookup = {} # A lookup table global to the function below.
    import quantities as pq
    units = pq.pA
    import copy
    from itertools import repeat
    #from utilities import Utilities


def p_imports():
    from neuronunit.models import backends
    from neuronunit.models.reduced import ReducedModel
    model = ReducedModel(get_neab.LEMS_MODEL_PATH,name='vanilla',backend='NEURON')
    model.load_model()
    utilities.get_neab = get_neab
    utilities.model = model

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
    #from neuronunit.tests import utilities as outils
    BOUND_LOW = [ np.min(i) for i in modelp.model_params.values() ]
    BOUND_UP = [ np.max(i) for i in modelp.model_params.values() ]
    NDIM = len(BOUND_UP)
    def uniform(low, up, size=None):
        try:
            return [random.uniform(a, b) for a, b in zip(low, up)]
        except TypeError:
            return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]
    creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM)
    toolbox.register("Individual", tools.initIterate, creator.Individual, toolbox.attr_float)
    toolbox.register("population", tools.initRepeat, list, toolbox.Individual)
    toolbox.register("select", tools.selNSGA2)
    #model = outils.model
dview.apply_sync(p_imports)

#list_checkpoints(notebook_id)
#create_checkpoint(notebook_id)
#restore_checkpoint(notebook_id, checkpoint_id)
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


def evaluate_e(individual,vms):#This method must be pickle-able for scoop to work.
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

    model = ReducedModel(get_neab.LEMS_MODEL_PATH,name=str(vms.attrs),backend='NEURON')
    model.load_model()

    assert type(vms.rheobase) is not type(None)
    uc = {'amplitude':vms.rheobase}


    DELAY = 100.0*pq.ms
    DURATION = 1000.0*pq.ms
    params = {'injected_square_current':
              {'amplitude':100.0*pq.pA, 'delay':DELAY, 'duration':DURATION}}

    current = params.copy()['injected_square_current']
    current.update(uc)
    current = {'injected_square_current':current}
    #Its very important to reset the model here. Such that its vm is new, and does not carry charge from the last simulation
    model.update_run_params(vms.attrs)
    model.inject_square_current(current)
    #reset model, clear away charge from previous model
    model.update_run_params(vms.attrs)
    n_spikes = model.get_spike_count()
    sane = get_neab.suite.tests[0].sanity_check(vms.rheobase*pq.pA, model)
    print(len(get_neab.tests))
    if (n_spikes == 1 and sane == True): #or n_spikes == 0  # Its possible that no rheobase was found
        for i in [4,5,6]:
            get_neab.suite.tests[i].params['injected_square_current']['amplitude'] = vms.rheobase*pq.pA
        get_neab.suite.tests[0].prediction={}
        assert type(vms.rheobase) != type(None)
        get_neab.suite.tests[0].prediction['value']=vms.rheobase * pq.pA
        #try:
        assert get_neab.suite.judge(model)#passing in model, changes the model
        score = get_neab.suite.judge(model)
        import pdb; pdb.set_trace()
        vms.score = score
        print(vms.score)
        model.run_number+=1
        error = score.sort_key.values.tolist()[0]

        other_mean = np.mean([i for i in error if type(i) is not type(None)])
        for i in error:
            if type(i) is type(None):
                i = other_mean
        #individual.error = error
        individual.rheobase = vms.rheobase
    else:
        inderr = getattr(individual, "error", None)
        if type(inderr) is not (None):
            if len(individual.error)!=0:
                #the average of 10 and the previous score is chosen as a nominally high distance from zero
                error = [ (abs(-10.0+i)/2.0) for i in individual.error ]
        else:
            error = [ 100.0 for i in range(0,8) ]

    print('This is the error {0}'.format((error)))
    return error[0],error[1],error[2],error[3],error[4],error[5],error[6],error[7],

#dview.push({'evaluate_e':evaluate_e})
#toolbox.register("evaluate", evaluate_e)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM)
toolbox.decorate("mate", history.decorator)
toolbox.decorate("mutate", history.decorator)
toolbox.register("select", tools.selNSGA2)




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

def replace_rh(pop,vmpop):
    '''
    #discard models that cause None rheobase results,
    # and create new models by mutating away from the corresponding  parameters.
    #make sure that the old individual, and virtual model object are
    #over written so do not use list append pattern, as this will not
    #over write the objects in place, but instead just grow the lists inappropriately
    #also some of the lines below may be superflous in terms of machine instructions, but
    #they function to make the code more explicit and human readable.
    '''

    from itertools import repeat
    import copy
    for i,ind in enumerate(pop):
        j=0
        while type(vmpop[i].rheobase) is type(None):
            print(j)
            j+=1
            #print('this loop appropriately exits none mutate away from ')
            toolbox.mutate(ind)
            toolbox.mutate(ind)
            toolbox.mutate(ind)
            print('trying mutations: {0}'.format(ind))
            #temp = individual_to_vm(ind,param_dict)
            trans_dict=vmpop[i].trans_dict
            local_tuple = (ind,trans_dict)
            vm_temp = individual_to_vm(local_tuple)
            init_value = (False, 0)
            vmpop[i] = searhcer(vm_temp,init_value)
            print('trying value {0}'.format(vmpop[i].rheobase))
            ind.rheobase = vmpop[i].rheobase
            pop[i] = ind
            print('rheobase value is updating {0}'.format(vmpop[i].rheobase))
            if type(vmpop[i].rheobase) is not type(None):
                break
        assert type(vmpop[i].rheobase) is not type(None)
    assert ind.rheobase == vmpop[i].rheobase
    assert len(pop)!=0
    assert len(vmpop)!=0
    return pop, vmpop



def test_to_model(vms,local_test_methods):
    from neuronunit.tests import get_neab
    tests = get_neab.suite.tests
    import matplotlib.pyplot as plt
    import copy
    global model
    model.update_run_params(vms.attrs)
    tests = None
    tests = get_neab.suite.tests
    tests[0].prediction={}
    tests[0].prediction['value']=vms.rheobase*qt.pA
    tests[0].params['injected_square_current']['amplitude']=vms.rheobase*qt.pA
    #TODO all of the external rheobase related things need to be re-encapsulated into the NeuroUnit class.
    if local_test_methods in [4,5,6]:
        tests[local_test_methods].params['injected_square_current']['amplitude']=vms.rheobase*qt.pA
    #model.results['vm'] = [ 0 ]
    model.re_init(vms.attrs)
    tests[local_test_methods].generate_prediction(model)


def update_vm_pop(pop, trans_dict, rh_value=None):
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
    import pdb
    pop = [toolbox.clone(i) for i in pop ]
    vmpop = []
    def transform(ind):
        vm = utilities.VirtualModel()
        param_dict={}
        for i,j in enumerate(ind):
            param_dict[trans_dict[i]] = str(j)
        vm.attrs = param_dict
        vm.name = vm.attrs
        return vm
    vmpop = dview.map_sync(transform, pop)
    vmpop = list(copy.copy(vmpop))
    return vmpop


def check_rheobase(vmpop):
    '''
    inputs a population of genes/alleles, the population size MU, and an optional argument of a rheobase value guess
    outputs a population of genes/alleles, a population of individual object shells, ie a pickleable container for gene attributes.
    Rationale, not every gene value will result in a model for which rheobase is found, in which case that gene is discarded, however to
    compensate for losses in gene population size, more gene samples must be tested for a successful return from a rheobase search.
    If the tests return are successful these new sampled individuals are appended to the population, and then their attributes are mapped onto
    corresponding virtual model objects.
    '''
    def check_fix_range2(vms):
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
            everything=np.concatenate((sub,supra))

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
            steps2 = np.linspace(sub.max(),2*sub.max(),7.0)
            np.delete(steps2,np.array(sub))
            steps = [ i*pq.pA for i in steps2 ]

        elif len(supra):
            steps2 = np.linspace(-2*(supra.min()),supra.min(),7.0)
            np.delete(steps2,np.array(supra))
            steps = [ i*pq.pA for i in steps2 ]

        vms.steps = steps
        vms.rheobase = None
        return copy.copy(vms)


    def check_current2(ampl,vm):
        '''
        Inputs are an amplitude to test and a virtual model
        output is an virtual model with an updated dictionary.
        '''

        global model
        import quantities as pq
        import get_neab
        from neuronunit.models import backends
        from neuronunit.models.reduced import ReducedModel

        DELAY = 100.0*pq.ms
        DURATION = 1000.0*pq.ms
        params = {'injected_square_current':
                  {'amplitude':100.0*pq.pA, 'delay':DELAY, 'duration':DURATION}}


        model = ReducedModel(get_neab.LEMS_MODEL_PATH, name=str(vm.name), backend='NEURON')
        model.load_model()
        model.update_run_params(vm.attrs)
        #print(type(model),type(self.model))
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
        import quantities as pq
        import numpy as np
        vm.boolean = False
        steps = list(np.linspace(50,150,7.0))
        steps_current = [ i*pq.pA for i in steps ]
        vm.steps = steps_current
        return vm

    def find_rheobase(vms):
        from neuronunit.models import backends
        from neuronunit.models.reduced import ReducedModel
        import get_neab
        model = ReducedModel(get_neab.LEMS_MODEL_PATH,name='place_holder',backend='NEURON')
        model.load_model()
        model.update_run_params(vms.attrs)
        cnt = 0
        while vms.boolean == False:
            for step in vms.steps:
                vms = check_current2(step,vms)#,repeat(vms))
                vms = check_fix_range2(vms)
                cnt+=1
                print(cnt)
        return vms
    vmpop = dview.map_sync(init_vm,vmpop)
    vmpop = list(vmpop)
    vmpop = dview.map_sync(find_rheobase,vmpop)
    vmpop = list(vmpop)
    return vmpop



'''
rh_value = [ i.rheobase for i in vmpop ]
def rbc(rh_value):
    boolean_r_check=False
    for r in rh_value:
        print(r)
        if type(r) is type(None):
            print(type(r))
            boolean_r_check == True
    return boolean_r_check

while rbc(rh_value) is True:
    pop,vmpop = replace_rh(pop,vmpop,rh_value)
    rh_value = [ i.rheobase for i in vmpop ]
rh_value = [ toolbox.clone(i).rheobase for i in copy.copy(vmpop) ]
assert len(pop) == len(vmpop)
assert len(pop)!=0
assert len(vmpop)!=0
assert rbc(rh_value) is False
'''
    #return pop,vmpop,rh_value


##
# Start of the Genetic Algorithm
##
NGEN=3
import numpy as np
MU=4
CXPB = 0.9
pf = tools.ParetoFront()
dview.push({'pf':pf})
trans_dict = get_trans_dict(param_dict)
td = trans_dict
dview.push({'trans_dict':trans_dict,'td':td})
pop = toolbox.population(n = MU)
pop = [ toolbox.clone(i) for i in pop ]
pf.update([toolbox.clone(i) for i in pop])
vmpop = update_vm_pop(pop, td)
vmpop = check_rheobase(vmpop)
fitnesses = []
for key,value in enumerate(pop):
    fitnesses.append(evaluate_e(value,vmpop[key]))
    print(fitnesses[key])

# eventually serial syntax above will be replaced with map, and then dview.map_sync

#fitnesses = list(map(evaluate_e, pop, vmpop ))
#fitnesses = list(dview.map_sync(evaluate_e, pop, vmpop ))
invalid_ind = [ ind for ind in pop if not ind.fitness.valid ]
for gen in range(1, NGEN):


    invalid_ind = [ ind for ind in pop if ind.fitness.valid ]
    offspring = tools.selTournamentDCD(invalid_ind, len(invalid_ind))
    offspring = tools.selNSGA2(pop, len(pop))
    assert len(offspring)!=0
    offspring = [toolbox.clone(ind) for ind in offspring]
    assert len(offspring)!=0
    for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
        if random.random() <= CXPB:
            toolbox.mate(ind1, ind2)
        toolbox.mutate(ind1)
        toolbox.mutate(ind2)
        del ind1.fitness.values, ind2.fitness.values

    invalid_ind = [ ind for ind in offspring if ind.fitness.valid ]

    vmpop = update_vm_pop(offspring, td)
    vmpop = check_rheobase(vmpop)
    fitnesses = list(dview.map_sync(evaluate_e, offspring , vmpop))
    for ind, fit in zip(offspring, fitnesses):
        ind.fitness.values = fit
    size_delta = MU-len(offspring)
    assert size_delta == 0
    pop = toolbox.select(offspring, MU)
    print('the pareto front is: {0}'.format(pf))

pop,vmpop = list(update_vm_pop(pop,td))

import pandas as pd
scores = []
for j,i in enumerate(pf):
    i.name = vmpop[j].attrs
    scores.append(vmpop[j].score)
    print(vmpop[j].score)

sc = pd.DataFrame(scores[0])
sc
data = [ pf[0].name ]
model_values0 = pd.DataFrame(data)
model_values0
rhstorage[0]

data = [ pf[1].name ]
model_values0 = pd.DataFrame(data)
model_values0



sc1 = pd.DataFrame(scores[1])
sc1

rhstorage[1]

data = [ pf[1].name ]
model_values1 = pd.DataFrame(data)
model_values1

pf[1].name

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
