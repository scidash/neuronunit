
import time
from math import sqrt
import pdb
import array
import random
import json


"""
Scoop can only operate on variables classes and methods at top level 0
This means something with no indentation, no nesting,
and no virtual nesting (like function decorators etc)
anything that starts at indentation level 0 is okay.

However the case may be different for functions. Functions may be imported from modules.
I am unsure if it is only the case that functions can be imported from a module, if they are not bound to
any particular class in that module.

Code from the DEAP framework, available at:
https://code.google.com/p/deap/source/browse/examples/ga/onemax_short.py
from scoop import futures
"""

import numpy as np
import matplotlib.pyplot as plt
import quantities as pq
from deap import algorithms
from deap import base
from deap.benchmarks.tools import diversity, convergence, hypervolume
from deap import creator
from deap import tools
from scoop import futures
import scoop

import get_neab

import quantities as qt
import os
import os.path
from scoop import utils

import sciunit.scores as scores


init_start=time.time()
creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0, -1.0,
                                                    -1.0, -1.0, -1.0, -1.0))
# -1.0, -1.0, -1.0, -1.0,))
creator.create("Individual",list, fitness=creator.FitnessMin)

class Individual(object):
    '''
    When instanced the object from this class is used as one unit of chromosome or allele by DEAP.
    Extends list via polymorphism.
    '''
    def __init__(self, *args):
        list.__init__(self, *args)
        self.error=None
        self.error=None
        self.results=None
        self.name=''
        self.attrs={}
        self.params=None
        self.score=None
        self.fitness=None
        self.s_html=None
        self.lookup={}
        self.rheobase=None
toolbox = base.Toolbox()

import model_parameters as params

vr = np.linspace(-75.0,-50.0,1000)
a = np.linspace(0.015,0.045,1000)
b = np.linspace(-3.5*10E-9,-0.5*10E-9,1000)
k = np.linspace(7.0E-4-+7.0E-5,7.0E-4+70E-5,1000)
C = np.linspace(1.00000005E-4-1.00000005E-5,1.00000005E-4+1.00000005E-5,1000)

c = np.linspace(-55,-60,1000)
d = np.linspace(0.050,0.2,1000)
v0 = np.linspace(-75.0,-45.0,1000)
vt =  np.linspace(-50.0,-30.0,1000)
vpeak= np.linspace(20.0,30.0,1000)

#vpeak as currently stated causes problems.
param=['vr','a','b','C','c','d','v0','k','vt','vpeak']
#,'d','v0','k','vt','vpeak']
#param=['a','b','vr']#,'vpeak']#,'k']#,'C']#,'c','d','v0','k','vt','vpeak']#,'d'
rov=[]
#vr = np.linspace(-75.0,-50.0,1000)
#a = np.linspace(0.015,0.045,1000)
#b = np.linspace(-3.5*10E-9,-0.5*10E-9,1000)
rov.append(vr)
rov.append(a)
rov.append(b)
rov.append(C)
rov.append(c)

rov.append(d)
rov.append(v0)
rov.append(k)
rov.append(vt)
rov.append(vpeak)

BOUND_LOW=[ np.min(i) for i in rov ]
BOUND_UP=[ np.max(i) for i in rov ]
#rov.append(vpeak)

NDIM = len(param)

LOCAL_RESULTS_spiking=[]
#LOCAL_RESULTS_no_spiking=[]
RUN_TIMES=''
import functools
#seed_in=1

def uniform(low, up, size=None):
    try:
        return [random.uniform(a, b) for a, b in zip(low, up)]
    except TypeError:
        return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]

toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM)
toolbox.register("Individual", tools.initIterate, creator.Individual, toolbox.attr_float)
import deap as deap

toolbox.register("population", tools.initRepeat, list, toolbox.Individual)

#from neuronunit.models import backends
#from neuronunit.models.reduced import ReducedModel
#model = ReducedModel(get_neab.LEMS_MODEL_PATH,name='vanilla',backend='NEURON')
#model.load_model()
#model = ReducedModel(get_neab.LEMS_MODEL_PATH,name='vanilla',backend='NEURON')
#model.rheobase=None

import grid_search as gs
model=gs.model
#model.cell_name


print(model)
def evaluate(individual,iter_):#This method must be pickle-able for scoop to work.
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

    '''
    DELAY = 100.0*pq.ms
    DURATION = 1000.0*pq.ms
    params = {'injected_square_current':
                {'amplitude':100.0*pq.pA, 'delay':DELAY, 'duration':DURATION}}
    '''
    vms,rheobase=iter_
    print(vms,rheobase)
    print(vms.attrs)
    import quantities as pq

    params=gs.params
    print(type(params))
    model=gs.model
    print(type(model))
    #model=gs.model()
    print(type(model))
    uc = {'amplitude':rheobase}
    current = params.copy()['injected_square_current']
    current.update(uc)
    current = {'injected_square_current':current}

    #Its very important to reset the model here. Such that its vm is new, and does not carry charge from the last simulation
    model.load_model()
    model.update_run_params(vms.attrs)

    #if len(model.attrs) == 0:
    #    model.update_run_params(vms.attrs)
    print(model)
    model.inject_square_current(current)
    n_spikes = model.get_spike_count()
    print(n_spikes)
    assert n_spikes == 1

    sane = False
    sane = get_neab.suite.tests[3].sanity_check(vms.rheobase*pq.pA,model)


    print(sane)
    if sane == True and n_spikes == 1:

        individual.params=[]
        for i in attrs['//izhikevich2007Cell'].values():
            if hasattr(individual,'params'):
                individual.params.append(i)
        get_neab.suite.tests[0].prediction={}
        get_neab.suite.tests[0].prediction['value']=0
        assert vms.rheobase!=None
        get_neab.suite.tests[0].prediction['value']=vms.rheobase*qt.pA
        #Reset the model again.
        model.load_model()
        score = get_neab.suite.judge(model)#passing in model, changes model
        if sane == True and n_spikes == 1:
            for i in [4,5,6]:
                get_neab.suite.tests[i].params['injected_square_current']['amplitude']=value*pq.pA
            get_neab.suite.tests[0].prediction={}
            score = get_neab.suite.tests[0].prediction['value']=value*pq.pA
            score = get_neab.suite.judge(model)#passing in model, changes model
            import neuronunit.capabilities as cap
            spikes_numbers=[]
            plt.clf()
            plt.hold(True)
            for k,v in score.related_data.items():
                spikes_numbers.append(cap.spike_functions.get_spike_train(((v.values[0]['vm']))))
                plt.plot(model.results['t'],v.values[0]['vm'])
            plt.savefig(str(model.name)+'.png')
            plt.clf()

        model.run_number+=1
        individual.results=model.results
        vms.results=model.results
        vms.score=score.sort_key.values.tolist()[0]
        error= score.sort_key.values.tolist()[0]

        import pickle
        for i in get_neab.suite.tests:
            i.last_model=None
            pickle.dump(i, open(str(i)+".p", "wb" ) )
            test=pickle.load(open(str(i)+".p", "rb" ) )
            individual.error=error

    elif sane == False:
        if len(individual.error)!=0:
            error = [ ((10.0+i)/2.0) for i in individual.error ]
        else:
            error = [ 10.0 for i in range(0,8) ]

    #pdb.set_trace()
    return error[0],error[1],error[2],error[3],error[4],error[5],error[6],error[7],




toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM)
toolbox.register("select", tools.selNSGA2)
toolbox.register("map", futures.map)


def plotss(vmlist,gen):
    import matplotlib.pyplot as plt
    plt.clf()
    for ind,j in enumerate(vmlist):
        if hasattr(ind,'results'):
            plt.plot(ind.results['t'],ind.results['vm'])
            plt.xlabel(str(vmlist[j].attr))
    plt.savefig('snap_shot_at_gen_'+str(gen)+'.png')
    plt.clf()




class VirtualModel:
    '''
    This is a pickable dummy clone
    version of the NEURON simulation model
    It does not contain an actual model, but it can be used to
    wrap the real model.
    This Object class serves as a data type for storing rheobase search
    attributes and other useful parameters,
    with the distinction that unlike the NEURON model this class
    can be transported across HOSTS/CPUs
    '''
    def __init__(self):
        self.lookup={}
        self.rheobase=None
        self.previous=0
        self.run_number=0
        self.attrs=None
        self.name=None
        self.s_html=None
        self.results=None
        self.score=None




def test_current(ampl,vm):
    '''
    Inputs are an amplitude to test and a virtual model
    output is an virtual model with an updated dictionary.
    '''
    import copy
    if float(ampl) not in vm.lookup or len(vm.lookup)==0:
        current = params.copy()['injected_square_current']
        uc={'amplitude':ampl}
        current.update(uc)

        current={'injected_square_current':current}
        vm.run_number+=1
        model.update_run_params(vm.attrs)

        model.load_model()
        #print('got here 1')
        #print(type(model.h.v_v_of0))
        model.inject_square_current(current)
        vm.previous=ampl
        n_spikes = model.get_spike_count()
        if n_spikes==1:
            vm.rheobase=ampl
            print(vm.attrs)
            print(model.attrs)
            print('hit')
        verbose=False
        if verbose:
            print("Injected %s current and got %d spikes" % \
                    (ampl,n_spikes))
        vm.lookup[float(ampl)] = n_spikes
        return vm.lookup
        #return copy.copy(vm.lookup)
    if float(ampl) in vm.lookup:
        return vm.lookup
small=None
from scoop import futures
#from neuronunit.models import backends
AMPL = 0.0*pq.pA
DELAY = 100.0*pq.ms
DURATION = 1000.0*pq.ms
from scipy.optimize import curve_fit
import sciunit
import sciunit.scores as scores
import neuronunit.capabilities as cap

required_capabilities = (cap.ReceivesSquareCurrent,
                         cap.ProducesSpikes)
params = {'injected_square_current':
            {'amplitude':100.0*pq.pA, 'delay':DELAY, 'duration':DURATION}}

name = "Rheobase test"
description = ("A test of the rheobase, i.e. the minimum injected current "
               "needed to evoke at least one spike.")

score_type = scores.RatioScore
guess=None

lookup = {} # A lookup table global to the function below.
verbose=True
import quantities as pq
units = pq.pA
verbose=True


def check_fix_range(lookup):
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
    sub=[]
    supra=[]
    print(lookup)
    for k,v in lookup.items():
        if v==1:
            #A logical flag is returned to indicate that rheobase was found.
            return (True,k)
        elif v==0:
            sub.append(k)
        elif v>0:
            supra.append(k)

    sub=np.array(sub)
    supra=np.array(supra)
                 # concatenate
    if len(sub) and len(supra):

        everything=np.concatenate((sub,supra))

        center = np.linspace(sub.max(),supra.min(),7.0)
        np.delete(center,np.array(everything))
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


    return (False,steps)



def main():

    #random.seed(seed)

    NGEN=4
    MU=12

    CXPB = 0.9
    import numpy as numpy
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean, axis=0)
    stats.register("std", numpy.std, axis=0)
    stats.register("min", numpy.min, axis=0)
    stats.register("max", numpy.max, axis=0)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    pop = toolbox.population(n=MU)
    #create the first population of individuals
    guess_attrs=[]

    #find rheobase on a model constructed out of the mean parameter values.
    for x,y in enumerate(param):
        guess_attrs.append(np.mean( [ i[x] for i in pop ]))

    from itertools import repeat
    mean_vm=VirtualModel()

    for i, p in enumerate(param):
        value=str(guess_attrs[i])
        model.name=str(model.name)+' '+str(p)+str(value)
        if i==0:
            attrs={'//izhikevich2007Cell':{p:value }}
        else:
            attrs['//izhikevich2007Cell'][p]=value
    mean_vm.attrs=attrs
    import copy

    #The above code between 492-544
    # was a lot of code, but all it was really doing was establishing a rheobase value in a fast way,
    #a parallel way, and a reliable way.
    #soon this code will be restated in much neater function definitions.

    def individual_to_vm(ind):
        for i, p in enumerate(param):
            value = str(ind[i])
            if i == 0:
                attrs={'//izhikevich2007Cell':{p:value }}
            else:
                attrs['//izhikevich2007Cell'][p] = value
        vm = VirtualModel()
        vm.attrs = attrs
        return vm


    steps = np.linspace(50,150,7.0)
    steps_current = [ i*pq.pA for i in steps ]
    rh_param=(False,steps_current)
    searcher=gs.searcher
    check_current=gs.check_current
    pre_rh_value=searcher(check_current,rh_param,mean_vm)
    rh_value=pre_rh_value.rheobase
    vmpop=list(map(individual_to_vm,pop))

    #Now attempt to get the rheobase values by first trying the mean rheobase value.
    #This is not an exhaustive search that results in found all rheobase values
    #It is just a trying out an educated guess on each individual in the whole population as a first pass.
    #invalid_ind = [ ind for ind in pop if not ind.fitness.valid ]
    rhstorage=list(futures.map(gs.evaluate,vmpop,repeat(rh_value)))
    rhstorage2 = [i.rheobase for i in rhstorage]
    rhstorage=rhstorage2
    iter_ = zip(vmpop,rhstorage)

    fitnesses = list(toolbox.map(toolbox.evaluate, pop, iter_))
    assert len(fitnesses)==len(invalid_ind)

    invalid_ind = [ ind for ind in pop if not ind.fitness.valid ]
    vmlist=list(futures.map(individual_to_vm,invalid_ind))

    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit



    # This is just to assign the crowding distance to the individuals
    # no actual selection is done
    pop = toolbox.select(pop, len(pop))

    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind), **record)

    # Begin the generational process
    for gen in range(1, NGEN):
        # Vary the population
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]


        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= CXPB:
                toolbox.mate(ind1, ind2)

            toolbox.mutate(ind1)
            toolbox.mutate(ind2)
            del ind1.fitness.values, ind2.fitness.values

        # Evaluate the individuals with an invalid fitness
        #invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        invalid_ind = [ ind for ind in pop if not ind.fitness.valid ]
        vmlist=[]
        vmlist=list(map(individual_to_vm,invalid_ind))


        #genes have changed so check/search rheobase again.
        for i,j in enumerate(invalid_ind):
            if vmlist[i].rheobase!=None:
                d=test_current(vmlist[i].rheobase,vmlist[i])
            else:
                d=test_current(rh_value,vmlist[i])
            if 1 not in d.values():
                unpack = check_fix_range(d)
                unpack = check_repeat(test_current,unpack[1],vmlist[i])
                if unpack[0] == True:
                    rh_value = unpack[1]
                else:
                    rh_value = searcher(test_current,unpack[1],vmlist[i])
        for i in vmlist:
            assert i.rheobase!=None

        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind, vmlist)

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        plotss(vmlist,gen)
        # Select the next generation population
        #This way the initial genes keep getting added to each generation.
        #pop = toolbox.select(pop + offspring, MU)
        #This way each generations genes are completely replaced by the result of mating.
        pop = toolbox.select(offspring, MU)
        if gen==NGEN:
            vmlist=[]
            error=evaluate(invalid_ind[0], vmlist[0])
            vmlist=list(map(individual_to_vm,pop))
            print(vmlist[0])
            f=open('html_score_matrix.html','w')
            f.write(vmlist[0].s_html)
            f.close()


        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        print(logbook.stream)
        pop.sort(key=lambda x: x.fitness.values)
        import pickle
        with open('minumum_and_maximum_values.pickle', 'rb') as handle:
            opt_values=pickle.load(handle)
            print('minumum and maximum values from exhaustive search routine')
            print(opt_values)


    return pop, list(logbook)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import pyneuroml as pynml
    import os
    import time
    start_time=time.time()
    whole_initialisation=start_time-init_start
    model=gs.model
    pop, stats = main()

    finish_time=time.time()
    ga_time=finish_time-start_time
    plt.clf()
    print(stats)
    f=open('finish_time.txt','w')
    init_time='{}{}{}'.format("init time: ",whole_initialisation,"\n")
    ft='{}{}{}'.format("ga_time: ",ga_time,"\n")
    f.write(init_time)
    f.write(ft)

    f=open('other_nrn_count_invokations_run_time_metric.txt','w')
    f.write(RUN_TIMES)
    f.write(ft)



    bfl=time.time()
    results = pynml.pynml.run_lems_with_jneuroml(os.path.split(get_neab.LEMS_MODEL_PATH)[1],
                             verbose=False, load_saved_data=True, nogui=True,
                             exec_in_dir=os.path.split(get_neab.LEMS_MODEL_PATH)[0],
                             plot=True)
    allr=time.time()
    lemscalltime=allr-bfl
    flt='{}{}{}'.format("lemscalltime: ",float(lemscalltime),"\n")
    f=open('jneuroml_call_time.txt','w')
    #vanilla model via neuron: 1.1804585456848145

    f.write(flt)
    f.close()

    plt.clf()
    plt.hold(True)

    for i in stats:

        plt.plot(np.sum(i['avg']),i['gen'])
        '{}{}{}'.format(np.sum(i['avg']),i['gen'],'results')
    plt.savefig('avg_error_versus_gen.png')
    plt.hold(False)


    plt.clf()
