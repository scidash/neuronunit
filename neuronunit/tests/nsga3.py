'''
import os
os.system('ipcluster start --profile=jovyan --debug &')
os.system('sleep 5')
import ipyparallel as ipp
rc = ipp.Client(profile='jovyan')
print('hello from before cpu ')
print(rc.ids)
#quit()
v = rc.load_balanced_view()
'''
import time
init_start=time.time()
import get_neab

"""
Code from the deap framework, available at:
https://code.google.com/p/deap/source/browse/examples/ga/onemax_short.py
Conversion to its parallel form took two lines:
from scoop import futures
"""
import array
import random
import json

import numpy as np

from math import sqrt

from deap import algorithms
from deap import base
from deap import benchmarks
from deap.benchmarks.tools import diversity, convergence, hypervolume
from deap import creator
from deap import tools
from scoop import futures

creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0))
# -1.0, -1.0, -1.0, -1.0,))
creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)

class Individual(object):
    '''
    When instanced the object from this class is used as one unit of chromosome or allele by DEAP.
    Extends list via polymorphism.
    '''
    def __init__(self, *args):
        list.__init__(self, *args)
        self.error=None
        self.sciunitscore=[]
        self.model=None
        self.error=None
        self.results=None
        self.name=''
        self.attrs={}
        self.params=None
        self.score=None
        self.fitness=None
        self.s_html=None
toolbox = base.Toolbox()
rov=[]

vr = np.linspace(-75.0,-65.0,1000)
a = np.linspace(0.015,0.045,1000)
b = np.linspace(-0.0010,-0.0035,1000)

k = np.linspace(7.0E-4-+7.0E-5,7.0E-4+70E-5,1000)
C = np.linspace(1.00000005E-4-1.00000005E-5,1.00000005E-4+1.00000005E-5,1000)

c = np.linspace(-55,-60,1000)
d = np.linspace(0.050,0.2,1000)
v0 = np.linspace(-75.0,-45.0,1000)
vt =  np.linspace(-50.0,-30.0,1000)
#vpeak = np.linspace(30.0,40.0,1000)
#vpeak as currently stated causes problems.
#param=['vr','a','b','C','c','d','v0','k','vt','vpeak']
param=['a','b','vr','k','C']#,'c','d','v0','k','vt','vpeak']#,'d'

rov.append(a)
rov.append(b)
rov.append(vr)
rov.append(k)
rov.append(C)
#rov.append(vpeak)

'''
rov.append(c)
rov.append(d)
rov.append(k)
rov.append(v0)
rov.append(vt)
'''

seed_in=1

BOUND_LOW=[ np.min(i) for i in rov ]
BOUND_UP=[ np.max(i) for i in rov ]
NDIM = len(param)
LOCAL_RESULTS_spiking=[]
LOCAL_RESULTS_no_spiking=[]

import functools

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
model.local_run()

def evaluate(individual):#This method must be pickle-able for scoop to work.
    for i, p in enumerate(param):
        name_value=str(individual[i])
        #reformate values.
        model.name=name_value
        if i==0:
            attrs={'//izhikevich2007Cell':{p:name_value }}
        else:
            attrs['//izhikevich2007Cell'][p]=name_value

    individual.attrs=attrs
    b4nrncall=time.time()
    model.update_run_params(attrs)
    afternrncall=time.time()
    #if afternrncall-b4nrncall>25:
    LOCAL_RESULTS_spiking.append(afternrncall-b4nrncall)

    individual.params=[]
    for i in attrs['//izhikevich2007Cell'].values():
        if hasattr(individual,'params'):
            individual.params.append(i)

    individual.results=model.results
    score = get_neab.suite.judge(model)
    import numpy as np
    import pdb
    #import spykeutils as spykeutils
    #import matplotlib as plt
    import matplotlib.pyplot as plt
    import quantities as pq
    #for i in score.unstack():
    #    if i=='Insufficient Data':
    #        pdb.set_trace()
    if 'Insufficient Data' in score.unstack():
        individual.error = [ 100 for i in range(0,8) ]
        if len(LOCAL_RESULTS_spiking)>0:
            del LOCAL_RESULTS_spiking[-1]
        LOCAL_RESULTS_no_spiking.append(afternrncall-b4nrncall)
        #pdb.set_trace()
    else:
        for i in score.unstack():
            if i.score is None:
                i.score=100.0
                if len(LOCAL_RESULTS_spiking)>0:
                    del LOCAL_RESULTS_spiking[-1]
                LOCAL_RESULTS_no_spiking.append(afternrncall-b4nrncall)

        individual.error = [ abs(i.score) for i in score.unstack() ]

    individual.s_html=score.to_html()
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

def main(seed=None):

    random.seed(seed)

    NGEN=2
    MU=8

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

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(evaluate, invalid_ind)

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
        #assert ind.results

        offspring = [toolbox.clone(ind) for ind in offspring]
        #print('cloning not true clone')


        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= CXPB:
                toolbox.mate(ind1, ind2)

            toolbox.mutate(ind1)
            toolbox.mutate(ind2)
            del ind1.fitness.values, ind2.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        #import pdb
        #pdb.set_trace()

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

    #offspring = tools.selTournamentDCD(pop, len(pop))
    (a,b,c,d,e,f,g,h) = evaluate(invalid_ind[0])

    f=open('html_score_matrix.html','w')
    f.write(invalid_ind[0].s_html)
    f.close()
    plotss(invalid_ind,gen)
    os.system('rm *.txt')

    f=open('stats_summart.txt','w')
    for i in list(logbook):
        f.write(str(i))
    f=open('mean_call_length_spiking.txt','w')
    f.write(str(np.mean(LOCAL_RESULTS_spiking))+': \n'))
    f.write('the number of calls to NEURON on one CPU only : \n')
    f.write(str(len(LOCAL_RESULTS_spiking))+str(' \n'))
    #f.write('tseconds ')

    #f=open('mean_call_length_no_spiking.txt','w')
    #f.write(str(np.mean(LOCAL_RESULTS_no_spiking)))
    #f.write('the number of calls to NEURON on one CPU only : \n')
    #f.write(str(len(LOCAL_RESULTS_no_spiking))+str(' \n'))
    #f.write('tseconds ')

    #pop=list(pop)
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
    bfl=time.time()
    #import pdb
    #pdb.set_trace()
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
    init_time='{}{}'.format("init time: ",whole_initialisation)
    ft='{}{}'.format("ga_time: ",ga_time)
    f.write(init_time)
    f.write(ft)

    #print(LOCAL_RESULTS)
    plt.clf()
    plt.hold(True)

    for i in stats:

        plt.plot(np.sum(i['avg']),i['gen'])
        '{}{}{}'.format(np.sum(i['avg']),i['gen'],'results')
    plt.savefig('avg_error_versus_gen.png')
    plt.hold(False)


    plt.clf()
    results = pynml.pynml.run_lems_with_jneuroml(os.path.split(get_neab.LEMS_MODEL_PATH)[1],
                             verbose=False, load_saved_data=True, nogui=True,
                             exec_in_dir=os.path.split(get_neab.LEMS_MODEL_PATH)[0],
                             plot=True)
    allr=time.time()
    lemscalltime=allr-bfl
    flt='{}{}'.format("lemscalltime: ",lemscalltime)
    f.write(flt)
    f.close()
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
