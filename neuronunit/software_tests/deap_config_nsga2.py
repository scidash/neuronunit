import pdb
import numpy as np

import random
import array
import random
import scoop as scoop
import numpy as np, numpy
import scoop
from math import sqrt
from scoop import futures
from deap import algorithms
from deap import base
from deap import benchmarks
#from deap.benchmarks.tools import diversity, convergence, hypervolume
from deap import creator
from deap import tools



#OBJ_SIZE = 1 #single weighted objective function.
#NDIM = 1 #single dimensional optimization problem

class deap_capsule:
    '''
    Just a container for hiding implementation, not a very sophisticated one at that.
    print this file not importing.
    '''
    def __init__(self,*args):
        self.tb = base.Toolbox()
        self.ngen=None
        self.pop_size=None
        self.model=None
        self.param=None
        self.rov=None
        #self.range_of_values=np.linspace(-65.0,-55.0,1000)

        #self.range_of_values=None
    def sciunit_optimize(self,test_or_suite,param,pop_size,ngen,range_of_values,NDIM=3,OBJ_SIZE=2,seed_in=1):
        self.ngen = ngen#250
        self.pop_size=pop_size#population size
        self.param=param
        self.rov=range_of_values

        toolbox = base.Toolbox()
        creator.create("FitnessMax", base.Fitness, weights=(-1.0,-1.0,-1.0,-1.0,-1.0))#Final comma here, important, not a typo, must be a tuple type.

        creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMax)

        class Individual(list):
            '''
            This object is used as one unit of chromosome or allele by DEAP.
            '''
            def __init__(self, *args):
                list.__init__(self, *args)
                model=None
                self.time_trace=None
                self.voltage_trace=None
                self.attrs=None



        def uniform(low, up, size=None):
            #assert size==2
            '''
            This is the PRNG distribution that defines the initial
            allele population. Inputs are the maximum and minimal numbers that the PRNG can generate.
            '''
            try:
                return [random.uniform(a, b) for a, b in zip(low, up)]
            except TypeError:
                return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]


        BOUND_LOW=[ np.min(i) for i in self.rov ]
        BOUND_UP=[ np.max(i) for i in self.rov ]

        NDIM = len(self.rov )




        toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)





        def sciunitjudge(individual):#,Previous_best=Previous_best):
            '''
            sciunit_judge is pretending to take the model individual and return the quality of the model f(X).
            '''


            self.model.load_model()

            for i,p in enumerate(self.param):
                name_value=str(individual[i])#i
                attrs={'//izhikevich2007Cell':{p:name_value }}
                self.model.update_run_params(attrs)
            individual.attrs=attrs

            #self.model.h.psection()
            score = test_or_suite.judge(self.model)

            error = score.sort_keys.values[0]
            individual.sciunitscore=error

            individual.time_trace=self.model.results['t']
            individual.voltage_trace=self.model.results['vm']
            #,error[5]
            return (error[0],error[1],error[2],error[3],error[4],)

        toolbox.register("evaluate",sciunitjudge)
        toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
        toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM)
        toolbox.register("select", tools.selNSGA2)

        random.seed(seed_in)

        CXPB = 0.9#cross over probability

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", numpy.mean, axis=0)
        stats.register("std", numpy.std, axis=0)
        stats.register("min", numpy.min, axis=0)
        stats.register("max", numpy.max, axis=0)

        logbook = tools.Logbook()
        logbook.header = "gen", "evals", "std", "min", "avg", "max"

        pop = toolbox.population(n=self.pop_size)



        for ind in pop:
            ind.sciunitscore={}


        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)

        for ind, fit in zip(invalid_ind, fitnesses):
            print(ind,fit)
            ind.fitness.values = fit
            print(ind.fitness.values)
        # This is just to assign the crowding distance to the individuals
        # no actual selection is done
        pop = toolbox.select(pop, len(pop))

        gen=0


        stats_fit = tools.Statistics(key=lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(key=len)
        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        record = mstats.compile(pop)

        # Begin the generational process
        for gen in range(1,self.ngen):
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
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Select the next generation population
            #was this: pop = toolbox.select(pop + offspring, MU)
            pop = toolbox.select(offspring, self.pop_size)


        print(record)
        return (pop,pop[0],pop[0].sciunitscore)
