
import time
init_start=time.time()
import get_neab

#Scoop can only operate on variables classes and methods at top level 0
#This means something with no indentation, no nesting,
#and no virtual nesting (like function decorators etc)

"""
anything that starts at indentation level 0.
Code from the deap framework, available at:
https://code.google.com/p/deap/source/browse/examples/ga/onemax_short.py
Conversion to its parallel form took two lines:
from scoop import futures
"""
import array
import random
import json

import numpy as np
import pdb
import matplotlib.pyplot as plt
import quantities as pq

from math import sqrt

from deap import algorithms
from deap import base
from deap.benchmarks.tools import diversity, convergence, hypervolume
from deap import creator
from deap import tools
from scoop import futures

import scoop
creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0))
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
        #self.sciunitscore=[]
        #self.model=model
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

#vpeak as currently stated causes problems.
#param=['vr','a','b','C','c','d','v0','k','vt','vpeak']
param=['a','b','vr']#,'vpeak']#,'k']#,'C']#,'c','d','v0','k','vt','vpeak']#,'d'
rov=[]
vr = np.linspace(-75.0,-50.0,1000)
a = np.linspace(0.015,0.045,1000)
b = np.linspace(-3.5*10E-9,-0.5*10E-9,1000)
#vpeak= np.linspace(30.0,30.0,1000)
rov.append(a)
rov.append(b)
rov.append(vr)
#rov.append(vpeak)
BOUND_LOW=[ np.min(i) for i in rov ]
BOUND_UP=[ np.max(i) for i in rov ]
#rov.append(vpeak)

NDIM = len(param)
LOCAL_RESULTS_spiking=[]
LOCAL_RESULTS_no_spiking=[]
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

vanila_start=time.time()
#model.local_run()
vanila_stop=time.time()
vanila_nrn_time=float(vanila_stop-vanila_start)
f=open('vanila_nrn_time.txt','w')
vt='{}{}{}'.format("vanila_nrn_time : ",float(vanila_nrn_time),"\n")
f.write(str(vanila_nrn_time))
f.close()

from neuronunit.models import backends
from neuronunit.models.reduced import ReducedModel
model = ReducedModel(get_neab.LEMS_MODEL_PATH,name='vanilla',backend='NEURON')
model=model.load_model()


def evaluate(individual,vms):#This method must be pickle-able for scoop to work.
    print(vms.rheobase)
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
    individual.model=model.update_run_params(attrs)

    individual.params=[]
    for i in attrs['//izhikevich2007Cell'].values():
        if hasattr(individual,'params'):
            individual.params.append(i)
    import quantities as qt
    get_neab.suite.tests[0].prediction={}
    print(vms.rheobase)
    get_neab.suite.tests[0].prediction['value']=vms.rheobase*qt.pA
    import os
    import os.path
    from scoop import utils

    f=open('scoop_log_'+str(utils.getHosts()),'w')
    f.write(str(attrs))
    f.close()
    score = get_neab.suite.judge(model)#passing in model, changes model
    model.run_number+=1
    RUN_TIMES='{}{}{}'.format('counting simulation run times on models',model.results['run_number'],model.run_number)

    individual.results=model.results
    LOCAL_RESULTS_spiking.append(model.results['sim_time'])
    '{}{}'.format('sim time stored: ',model.results['sim_time'])

    try:
        individual.error = []
        individual.error = [ abs(i.score) for i in score.unstack() ]
        individual.s_html=score.to_html()
    except Exception as e:
        '{}'.format('Insufficient Data')
        #individual.error = []
        if np.sum(individual.error)!=0:
            individual.error = [ (10.0+i)/2.0 for i in individual.error ]
        else:
            individual.error = [ 10.0 for i in range(0,8) ]

        individual.s_html=None


    error=individual.error
    assert individual.results
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
    def __init__(self):
        self.lookup={}
        self.rheobase=None
        self.previous=0
        self.run_number=0
        self.attrs=None
        self.name=None




def ff(ampl,vm):
    print(vm, ampl)
    if float(ampl) not in vm.lookup:
        current = params.copy()['injected_square_current']
        print('current, previous = ',ampl,vm.previous)
        uc={'amplitude':ampl}
        current.update(uc)
        current={'injected_square_current':current}
        vm.run_number+=1
        print('model run number',vm.run_number)

        model.inject_square_current(current)
        vm.previous=ampl
        n_spikes = model.get_spike_count()
        if n_spikes==1:
            vm.rheobase=ampl
        verbose=True
        if verbose:
            print("Injected %s current and got %d spikes" % \
                    (ampl,n_spikes))
        vm.lookup[float(ampl)] = n_spikes
        return vm

    if float(ampl) in vm.lookup:
        print('model_in lookup')
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



def evaluate2(individual, guess_value=None):#This method must be pickle-able for scoop to work.
    #import rheobase_old2 as rh
    model=VirtuaModel()
    model.name=''

    for i, p in enumerate(param):
        value=str(individual[i])
        #reformate values.
        model.name=str(model.name)+' '+str(p)+str(value)
        if i==0:
            attrs={'//izhikevich2007Cell':{p:value }}
        else:
            attrs['//izhikevich2007Cell'][p]=value

    individual.attrs=attrs
    if guess_value == None:

        individual.lookup={}
        vm=VirtuaModel()
        import copy
        vm.attrs=copy.copy(individual.attrs)
        unpack=check_repeat2(ff,individual.lookup,individual)

        #def check_repeat(ff,unpack,vm):

        vm=ff(guess_value,vm)
        for k,v in vm.lookup.items():
            if v==1:
                individual.rheobase=k
                vm.rheobase=k
                return individual
            if v!=1:
                guess_value = None#more trial and error.
    #if guess_value == None:
    #    (run_number,k,attrs)=main2(individual)
    #individual.rheobase=0
    individual.rheobase=k
    return individual

#some how I destroy the function f.

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
    print(lookup2)
    for v,k in lookup2:
        if v==1:
            #A logical flag is returned to indicate that rheobase was found.
            return (True,k)
        elif v==0:
            sub.append(k)
        elif v>0:
            supra.append(k)
        print(k,v)
    print(sub)
    print(supra)

    sub=np.array(sub)
    supra=np.array(supra)

    if len(sub) and len(supra):
        center=(sub.max()+supra.min())/2.0
        steps2=np.linspace(center-sub.max(),center+supra.min(),7.0)
        steps2 = np.linspace(sub.max(),supra.min(),7.0)
        np.delete(steps2,np.array(lookup2))
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
    guess_attrs.append(np.mean( [ i[0] for i in pop ]))
    guess_attrs.append(np.mean( [ i[1] for i in pop ]))
    guess_attrs.append(np.mean( [ i[2] for i in pop ]))

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



    def check_repeat2(ff,unpack,vm):
        '''
        inputs:
        ff, a function,
        unpack, a dictionary of previous search values:
        vm a virtual model object.

        '''
        print(vm)
        print(type(vm))
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
            print(guess_value)
            vm.rheobase=guess_value
            return (True,guess_value)
        else:
            return (False,guess_value)

    def check_repeat(ff,unpack,vm):
        '''
        inputs:
        ff, a function,
        unpack, a dictionary of previous search values:
        vm a virtual model object.

        '''
        print(vm)
        print(type(vm))
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
            print(guess_value)
            vm.rheobase=guess_value
            return (True,guess_value)
        else:
            return (False,guess_value)

    from itertools import repeat

    steps2 = np.linspace(40,80,7.0)
    steps = [ i*pq.pA for i in steps2 ]



    #this might look like a big list iteration, but its not.
    #the statement below just finds rheobase on one value, that is the value
    #constituted by mean_vm. This will be used to speed up the rheobase search later.
    model.attrs=mean_vm.attrs
    #def bulk_process(ff,steps,mean_vm):
    lookup2=list(futures.map(ff,steps,repeat(mean_vm)))

    l3=[]
    d={}
    for l in lookup2:
        for k,v in l.lookup.items():
            l3.append((v, k))
            d[k]=v


    #Both ckeck repeat, and check fix range seem to be invoked here :)
    unpack=check_fix_range(l3)
    unpack=check_repeat(ff,unpack[1],mean_vm)
    if unpack[0]==True:
        guess_value=unpack[1]


    def searcher(ff,unpack,vms):
        while unpack[0]==False:
            l3=[]
            d={}
            for l in unpack[1]:
                for k,v in l.lookup.items():
                    l3.append((v, k))
                    d[k]=v
            unpack=check_fix_range(l3)
            unpack=check_repeat(ff,unpack[1],vms)
            print('stuck in a loop?')

            if unpack[0]==True:
                guess_value=unpack[1]
                print(guess_value)

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
        vm.attr=attrs
        return vm

    vmlist=list(map(individual_to_vm,pop))

    #Now attempt to get the rheobase values by first trying the mean rheobase value.
    #This is not an exhaustive search that results in found all rheobase values
    #It is just a trying out an educated guess on each individual in the whole population as first pass.
    list_of_hits_misses=list(futures.map(ff,repeat(guess_value),vmlist))
    j=0
    for i in list_of_hits_misses:
        print(i.rheobase)
        j+=1
        print(j)

    invalid_ind = [ ind for ind in pop if not ind.fitness.valid ]

    #For each individual in the new GA population.
    #Create Virtual Models that are readily pickle-able.
    #for the list of invalid indexs

    b=time.time()


    e=time.time()

    for i,j in enumerate(invalid_ind):
        print(i)
        print('vmlist[i].rheobase')
        print(vmlist[i].rheobase)
        print(type(j))

        if vmlist[i].rheobase==None:

            print(vmlist[i])
            print(vmlist[i].lookup)
            lookup2=ff(guess_value,vmlist[i])
            l3=[]
            d={}
            for k,v in lookup2.lookup.items():
                l3.append((v, k))
                d[k]=v
            print(l3)
            print(d)
            if 1 not in d.values():
                unpack=check_fix_range(l3)
                unpack=check_repeat(ff,unpack[1],vmlist[i])
                if unpack[0]==True:
                    guess_value=unpack[1]
                else:
                    guess_value=searcher(ff,unpack,vmlist[i])

    fitnesses = toolbox.map(evaluate, invalid_ind, vmlist)

    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit


    # This is just to assign the crowding distance to the individuals
    # no actual selection is done
    pop = toolbox.select(pop, len(pop))

    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind), **record)
    print(logbook.stream)

    # Begin the generational process
    for gen in range(1, NGEN):
        print(gen)
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
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

        iterator=(futures.map(evaluate,invalid_ind,repeat(invalid_ind.rheobase)))
        invalid_indvm=[]
        for i in iterator:
            if hasattr(i,'rheobase'):
                print('rheobase can update too \n\n\n\n')
                print(i.rheobase)
                vm=VirtuaModel()
                vm.rheobase=i.rheobase
                guess_value=i.rheobase
            invalid_ind.append(copy.copy(i))
            invalid_indvm.append(copy.copy(vm))

        invalid_ind = list(futures.map(evaluate2,invalid_ind))
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind, invalid_indvm)


        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        plotss(invalid_ind,gen)

        # Select the next generation population
        #This way the initial genes keep getting added to each generation.
        #pop = toolbox.select(pop + offspring, MU)
        #This way each generations genes are completely replaced by the result of mating.
        pop = toolbox.select(offspring, MU)
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        print(logbook.stream)
        pop.sort(key=lambda x: x.fitness.values)
        import numpy
        front = numpy.array([ind.fitness.values for ind in pop])
        plt.scatter(front[:,0], front[:,1], front[:,2], front[:,3])
        plt.axis("tight")
        plt.savefig('front.png')
        plt.clf()

    #when scoop is run in parallel only the fitnesses from the individual object
    #are retained after distributing individuals and reducing them back to rank0
    #there is no way to garuntee that the best candidate solution will
    #retain its object attributes, except via re evaluating it, in a scope outside
    #of futures.map as is done below.
    (a,b,c,d,e,f,g,h) = evaluate(invalid_ind[0],invalid_indvm[0])

    f=open('html_score_matrix.html','w')
    f.write(invalid_ind[0].s_html)
    f.close()
    plotss(invalid_ind,gen)
    #os.system('rm *.txt')

    f=open('stats_summart.txt','w')
    for i in list(logbook):
        f.write(str(i))
    f=open('mean_call_length_spiking.txt','w')
    mean_spike_call_time='{}{}{}'.format('mean spike call time',str(np.mean(LOCAL_RESULTS_spiking)), str(': \n') )
    f.write(mean_spike_call_time)
    f.write('the number of calls to NEURON on one CPU only : \n')
    #def padd(LOCAL_RESULTS_spiking):
    #    global_sum+=len(LOCAL_RESULTS_spiking)
    #    return global_sum
    #global_sum = futures.map(padd,LOCAL_RESULTS_spiking)

    f.write(str(len(LOCAL_RESULTS_spiking))+str(' \n'))
    #f.write(str(len(global_sum))+str(' \n'))

    plt.clf()
    plt.hold(True)
    for i in logbook:
        plt.plot(np.sum(i['avg']),i['gen'])
        '{}{}{}'.format(np.sum(i['avg']),i['gen'],'results')
    plt.savefig('avg_error_versus_gen.png')
    plt.hold(False)
    #'{}{}'.format("finish_time: ",finish_time)
    return pop, list(logbook)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import pyneuroml as pynml
    import os


    # with open("pareto_front/zdt1_front.json") as optimal_front_data:
    #     optimal_front = json.load(optimal_front_data)
    # Use 500 of the 1000 points in the json file
    # optimal_front = sorted(optimal_front[i] for i in range(0, len(optimal_front), 2))
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

    #print(LOCAL_RESULTS)
    plt.clf()
    plt.hold(True)

    import os
    #display all the results as travis standard out.
    os.system('cat *.txt')
    for i in stats:

        plt.plot(np.sum(i['avg']),i['gen'])
        '{}{}{}'.format(np.sum(i['avg']),i['gen'],'results')
    plt.savefig('avg_error_versus_gen.png')
    plt.hold(False)


    plt.clf()



    '''
    model = ReducedModel(get_neab.LEMS_MODEL_PATH,name='vanilla',backend='jNeuroMLBackend')
    model = ReducedModel(get_neab.LEMS_MODEL_PATH,name='vanilla')
    for i, p in enumerate(param):
        #name_value=str(individual[i])
        #reformate values.
        #model.name=name_value
        name_value='izhikevich2007Cell'
        if i==0:
            attrs={'//izhikevich2007Cell':{p:name_value }}
        else:
            attrs['//izhikevich2007Cell'][p]=name_value
        model.update_run_params(attrs)
   '''
