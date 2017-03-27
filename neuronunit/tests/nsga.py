
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
anything that starts at indentation level 0.
Code from the deap framework, available at:
https://code.google.com/p/deap/source/browse/examples/ga/onemax_short.py
Conversion to its parallel form took two lines:
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

from neuronunit.models import backends
from neuronunit.models.reduced import ReducedModel
model = ReducedModel(get_neab.LEMS_MODEL_PATH,name='vanilla',backend='NEURON')
model=model.load_model()


def evaluate(individual,vms):#This method must be pickle-able for scoop to work.
    '''
    Assumes rheobase for each individual virtual model object (vms) has already been found
    there should be a check for vms.rheobase, and if not then error.
    Inputs a gene and a virtual model object.
    outputs are error components.
    '''


    model.name=''
    for i, p in enumerate(param):
        name_value=str(individual[i])
        #reformate values.
        model.name=str(model.name)+' '+str(p)+str(name_value)
        if i==0:
            attrs={'//izhikevich2007Cell':{p:name_value }}
        else:
            attrs['//izhikevich2007Cell'][p]=name_value


    individual.attrs=attrs
    #make sure that the virtual model, and the real model have the same attributes.
    assert individual.attrs==vms.attrs


    #Its very important to reset the model here. Such that its vm is new, and does not carry charge from the last simulation
    model.load_model()

    individual.model=model.update_run_params(attrs)
    sane = False
    #model.update_run_params(vm.attrs)

    sane = get_neab.suite.tests[3].sanity_check(value*1.01*pq.pA,model)

    if sane == True:

        individual.params=[]
        for i in attrs['//izhikevich2007Cell'].values():
            if hasattr(individual,'params'):
                individual.params.append(i)
        get_neab.suite.tests[0].prediction={}
        get_neab.suite.tests[0].prediction['value']=0
        assert vms.rheobase!=None
        get_neab.suite.tests[0].prediction['value']=vms.rheobase*qt.pA
        model.load_model()
        score = get_neab.suite.judge(model)#passing in model, changes model
        model.run_number+=1
        individual.results=model.results
        error= score.sort_key.values
    elif sane == False:
        error = [ 10.0 for i in range(0,7) ]
    return error[0],error[1],error[2],error[3],error[4],error[5],error[6],error[7],




toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM)
toolbox.register("select", tools.selNSGA2)
toolbox.register("map", futures.map)


def plotss(pop,gen):
    import matplotlib.pyplot as plt
    plt.clf()

    for ind in pop:
        if hasattr(ind,'results'):
            plt.plot(ind.results['t'],ind.results['vm'])
            plt.xlabel(str(ind.attrs))
    plt.savefig('snap_shot_at_'+str(gen)+'.png')
    plt.clf()




class VirtuaModel:
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




def ff(ampl,vm):
    '''
    Inputs are an amplitude to test and a virtual model
    output is an virtual model with an updated dictionary.
    '''
    if float(ampl) not in vm.lookup:
        current = params.copy()['injected_square_current']
        uc={'amplitude':ampl}
        current.update(uc)
        current={'injected_square_current':current}
        vm.run_number+=1
        model.load_model()

        model.inject_square_current(current)
        vm.previous=ampl
        n_spikes = model.get_spike_count()
        if n_spikes==1:
            vm.rheobase=ampl
        verbose=False
        if verbose:
            print("Injected %s current and got %d spikes" % \
                    (ampl,n_spikes))
        vm.lookup[float(ampl)] = n_spikes
        return vm

    if float(ampl) in vm.lookup:
        return vm

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


def check_fix_range(lookup2):
    '''
    Inputs: lookup2, A dictionary of previous current injection values
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
    for v,k in lookup2:
        if v==1:
            #A logical flag is returned to indicate that rheobase was found.
            return (True,k)
        elif v==0:
            sub.append(k)
        elif v>0:
            supra.append(k)

    sub=np.array(sub)
    supra=np.array(supra)

    if len(sub) and len(supra):
        #center=(sub.max()+supra.min())/2.0
        #steps2=np.linspace(center-sub.max(),center+supra.min(),7.0)
        steps2 = np.linspace(sub.max(),supra.min(),7.0)
        np.delete(steps2,np.array(lookup2))
        #make sure that element 4 in a seven element vector
        #is exactly half way between sub.max() and supra.min()
        steps2[int(len(steps2)/2)+1]=(sub.max()+supra.min())/2.0
        steps = [ i*pq.pA for i in steps2 ]

    elif len(sub):
        steps2 = np.linspace(sub.max(),2*sub.max(),7.0)
        np.delete(steps2,np.array(lookup2))
        steps = [ i*pq.pA for i in steps2 ]

    elif len(supra):
        steps2 = np.linspace(-2*(supra.min()),supra.min(),7.0)
        np.delete(steps2,np.array(lookup2))
        steps = [ i*pq.pA for i in steps2 ]

    if len(steps)<7:
        steps2 = np.linspace(steps.min(),steps.max(),7.0)
        steps = [ i*pq.pA for i in steps2 ]

    import copy
    return (False,copy.copy(steps))


def main(seed=None):

    random.seed(seed)

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
    mean_vm=VirtuaModel()

    for i, p in enumerate(param):
        value=str(guess_attrs[i])
        model.name=str(model.name)+' '+str(p)+str(value)
        if i==0:
            attrs={'//izhikevich2007Cell':{p:value }}
        else:
            attrs['//izhikevich2007Cell'][p]=value
    mean_vm.attrs=attrs
    import copy


    def check_repeat(ff,unpack,vm):
        '''
        inputs:
        ff, a function,
        unpack, a dictionary of previous search values:
        vm a virtual model object.
        outputs: a tuple consisting of a boolean flag and
        the eecond element is a dictionary containing the
        last range searched such that the latest range searched
        can be used to seed future searches.
        '''
        from itertools import repeat
        lookup2=list(futures.map(ff,unpack,repeat(vm)))
        l3=[]
        for l in lookup2:
            for k,v in l.lookup.items():
                l3.append((v, k))

        unpack=check_fix_range(l3)
        l3=None
        boolean=False
        new_ranges=[]
        boolean=unpack[0]
        guess_value=unpack[1]
        if True == boolean:
            vm.rheobase=guess_value
            return (True,guess_value)
        else:
            return (False,guess_value)

    from itertools import repeat


    def searcher2(ff,unpack,vms):
        '''
        ultimately an attempt to capture the essence a lot of repeatative code below.
        This is not yet used, but it is intended for future use.
        Its intended to replace the less general searcher function
        '''
        steps2 = np.linspace(40,80,7.0)
        steps = [ i*pq.pA for i in steps2 ]
        model.attrs=mean_vm.attrs
        lookup2=list(futures.map(ff,steps,repeat(mean_vm)))

        while unpack[0]==False:
            l3=[]# convert a dictionary to a list.
            for l in unpack[1]:
                for k,v in vms.lookup.items():
                    l3.append((v, k))
            unpack=check_fix_range(l3)
            unpack=check_repeat(ff,unpack[1],vms)
            if unpack[0]==True:
                guess_value=unpack[1]
                return guess_value


    steps2 = np.linspace(40,80,7.0)
    steps = [ i*pq.pA for i in steps2 ]



    #this might look like a big list iteration, but its not.
    #the statement below just finds rheobase on one value, that is the value
    #constituted by mean_vm. This will be used to speed up the rheobase search later.
    model.attrs=mean_vm.attrs
    #def bulk_process(ff,steps,mean_vm):
    lookup2=list(futures.map(ff,steps,repeat(mean_vm)))

    l3=[]
    for l in lookup2:
        for k,v in l.lookup.items():
            l3.append((v, k))
    unpack=check_fix_range(l3)
    unpack=check_repeat(ff,unpack[1],mean_vm)
    if unpack[0]==True:
        guess_value=unpack[1]


    def searcher(ff,unpack,vms):
        while unpack[0]==False:
            l3=[]# convert a dictionary to a list.
            for l in unpack[1]:
                for k,v in vms.lookup.items():
                    l3.append((v, k))
            unpack=check_fix_range(l3)
            unpack=check_repeat(ff,unpack[1],vms)
            if unpack[0]==True:
                guess_value=unpack[1]
                return guess_value


    #The above code between 492-544
    # was a lot of code, but all it was really doing was establishing a rheobase value in a fast way,
    #a parallel way, and a reliable way.
    #soon this code will be restated in much neater function definitions.

    def individual_to_vm(ind):
        for i, p in enumerate(param):
            value=str(ind[i])
            if i==0:
                attrs={'//izhikevich2007Cell':{p:value }}
            else:
                attrs['//izhikevich2007Cell'][p]=value
        vm=VirtuaModel()
        vm.attrs=attrs
        #assert ind.attrs==vm.attrs
        return vm


    #Now attempt to get the rheobase values by first trying the mean rheobase value.
    #This is not an exhaustive search that results in found all rheobase values
    #It is just a trying out an educated guess on each individual in the whole population as a first pass.
    invalid_ind = [ ind for ind in pop if not ind.fitness.valid ]
    vmlist=list(map(individual_to_vm,invalid_ind))

    list_of_hits_misses=list(futures.map(ff,repeat(guess_value),vmlist))

    #For each individual in the new GA population.
    #Create Virtual Models that are readily pickle-able.
    #for the list of invalid indexs
    assert len(invalid_ind)==len(vmlist)

    for i,j in enumerate(invalid_ind):
        if vmlist[i].rheobase==None:
            lookup2=ff(guess_value,vmlist[i])
            l3=[]
            #d={}
            d=lookup2.lookup
            for k,v in d.items():
                l3.append((v, k))
                #d[k]=v
            if 1 not in d.values():
                unpack=check_fix_range(l3)
                unpack=check_repeat(ff,unpack[1],vmlist[i])
                if unpack[0]==True:
                    guess_value=unpack[1]
                else:
                    guess_value=searcher(ff,unpack,vmlist[i])

    for i in vmlist:
        assert i.rheobase!=None

    assert len(invalid_ind)==len(vmlist)
    #fitnesses = toolbox.map(toolbox.evaluate, invalid_ind, vmlist)

    fitnesses = []
    fitnesses = list(toolbox.map(toolbox.evaluate, invalid_ind, vmlist))
    assert len(fitnesses)==len(invalid_ind)
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
                lookup2=ff(vmlist[i].rheobase,vmlist[i])
            else:
                lookup2=ff(guess_value,vmlist[i])
            l3=[]
            d=lookup2.lookup
            for k,v in d.items():
                d[k]=v
                l3.append((v, k))
            if 1 not in d.values():
                unpack=check_fix_range(l3)
                unpack=check_repeat(ff,unpack[1],vmlist[i])
                if unpack[0]==True:
                    guess_value=unpack[1]
                else:
                    guess_value=searcher(ff,unpack,vmlist[i])
        for i in vmlist:
            assert i.rheobase!=None

        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind, vmlist)

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        plotss(invalid_ind,gen)
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




    f=open('stats_summart.txt','w')
    for i in list(logbook):
        f.write(str(i))
    f=open('mean_call_length_spiking.txt','w')
    mean_spike_call_time='{}{}{}'.format('mean spike call time',str(np.mean(LOCAL_RESULTS_spiking)), str(': \n') )
    f.write(mean_spike_call_time)
    f.write('the number of calls to NEURON on one CPU only : \n')
    f.write(str(len(LOCAL_RESULTS_spiking))+str(' \n'))

    plt.clf()
    plt.hold(True)
    for i in logbook:
        plt.plot(np.sum(i['avg']),i['gen'])
        '{}{}{}'.format(np.sum(i['avg']),i['gen'],'results')
    plt.savefig('avg_error_versus_gen.png')
    plt.hold(False)
    return pop, list(logbook)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import pyneuroml as pynml
    import os

    start_time=time.time()
    whole_initialisation=start_time-init_start

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
