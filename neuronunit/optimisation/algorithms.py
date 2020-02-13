"""Optimisation class
Copyright (c) 2016, EPFL/Blue Brain Project
 This file is part of BluePyOpt <https://github.com/BlueBrain/BluePyOpt>
"""
from deap.benchmarks.tools import diversity, convergence
import deap.tools as tools

import copy
import logging
import math
import pdb
import pickle
import random

import deap.algorithms
import deap.tools
#from . import tools
import numpy as np
try:
    import asciiplotlib as apl
except:
    pass
import numpy
from deap import algorithms

from neuronunit.optimisation import optimization_management as om
#from neuronunit.optimisation.optimization_management import WSListIndividual
class WSListIndividual(list):
    """Individual consisting of list with weighted sum field"""
    def set_fitness(self,obj_size):
        self.fitness = WeightedSumFitness(obj_size=obj_size)
    def __init__(self, *args, **kwargs):
        """Constructor"""
        self.rheobase = None
        self.dtc = None
        super(WSListIndividual, self).__init__(args)#args)#, **kwargs)
        #super(WSListIndividual, self).__init__()
        #self.extend(args)
        self.obj_size = len(args)
        #self.set_fitness()
        #self.fitness = tuple(1.0 for i in range(0,self.obj_size))
        self.set_fitness(obj_size=self.obj_size)


logger = logging.getLogger('__main__')
def _evaluate_invalid_fitness(toolbox, population):
    '''Evaluate the individuals with an invalid fitness
    Returns the count of individuals with invalid fitness
    '''
    invalid_ind = [ind for ind in population if not ind.fitness.valid]

    invalid_pop,fitnesses = toolbox.evaluate(invalid_ind)

    for j, ind in enumerate(invalid_pop):
        ind.fitness.values = fitnesses[j]
        #ind.dtc = None
    return invalid_pop
'''
def strip_object(p):
    state = super(type(p)).state
    p.unpicklable = []

    return p._state(state=state, exclude=['unpicklable','verbose'])

def purify2(population):
    pop2 = WSListIndividual()
    for ind in population:
        for i,j in enumerate(ind):
            try:
                ind[i] = float(j)
            except:
                import pdb
                pdb.set_trace()

        ind.dtc = ind.dtc
        try:
            ind.dtc.tests.DO = None
        except:
            ind.dtc.tests = None
        assert hasattr(ind,'dtc')

        pop2.append(ind)
    return pop2
'''
def _update_history_and_hof(halloffame,pf, history, population,GEN,MU):
    '''Update the hall of fame with the generated individuals

    Note: History and Hall-of-Fame behave like dictionaries
    '''
    temp = copy.copy([p for p in population if hasattr(p,'dtc')])
    dtcpop = copy.copy([p.dtc for p in population if hasattr(p,'dtc')])

    if "ADEXP" in temp[0].dtc.backend or "BHH" in temp[0].dtc.backend:
        for t in temp:
            t.dtc.tests = None
            t.dtc = None
    if halloffame is not None:
        try:
            halloffame.update(temp)
        except:
            print(temp,'temp bad')
    if history is not None:
        try:
            history.update(temp)
        except:
            print(temp,'temp bad')
    if pf is not None:
        if GEN ==0:
            pf = deap.tools.ParetoFront(MU)
        pf.update(temp)
    return (halloffame,pf,history)


def _record_stats(stats, logbook, gen, population, invalid_count):
    '''Update the statistics with the new population'''
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    import numpy
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    record = stats.compile(copy.copy(population)) if stats is not None else {}
    logbook.record(gen=gen, nevals=invalid_count, **record)

def gene_bad(offspring):
    gene_bad = False
    for o in offspring:
        if np.any(np.isnan(o)) or np.any(np.isinf(o)):
            gene_bad = True
    return gene_bad
def prune_constants(parents,num_constants):
    for i in range(0,num_constants):
        for p in parents:
            del p[-1]
    return parents

CXPB = 1.0
MUTPB = 1.0
def eaAlphaMuPlusLambdaCheckpoint(
        pop,
        toolbox,
        MU,
        cxpb,
        mutpb,
        NGEN,
        stats=None,
        hof = None,
        pf = None,
        nelite = 3,
        cp_frequency = 1,
        cp_filename = None,
        continue_cp = False,
        selection = 'selNSGA3',
        td=None):
    gen_vs_pop = []

    if continue_cp:
        # A file name has been given, then load the data from the file
        cp = pickle.load(open(cp_filename, "r"))
        pop = cp["population"]
        parents = cp["parents"]
        start_gen = cp["generation"]
        hof = cp["halloffame"]
        logbook = cp["logbook"]
        history = cp["history"]
        random.setstate(cp["rndstate"])

    else:
        history = deap.tools.History()

        #ref_points = tools.uniform_reference_points(len(pop[0]), len(pop))
        toolbox.register("select", tools.selNSGA2)#, ref_points=ref_points)
        random.seed()

        stats.register("avg", numpy.mean, axis=0)
        stats.register("std", numpy.std, axis=0)
        stats.register("min", numpy.min, axis=0)
        stats.register("max", numpy.max, axis=0)

        logbook = tools.Logbook()
        logbook.header = "gen", "evals", "std", "min", "avg", "max"
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        invalid_ind,fitnesses = toolbox.evaluate(invalid_ind)


        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit


        # Compile statistics about the population
        pop = [p for p in pop if len(p.fitness.values) ]
        #foffspring = [p for p in offspring if len(p.fitness.values) ]

        record = stats.compile(pop)
        logbook.record(gen=0, evals=len(invalid_ind), **record)
        temp_pop = copy.copy(pop)
        hof, pf,history = _update_history_and_hof(hof, pf, history, temp_pop ,0,MU)
        for p in pop:
            assert hasattr(p,'dtc')
        #print(logbook.stream)

        # Begin the generational process
        for gen in range(1, NGEN):
            offspring = algorithms.varAnd(pop, toolbox, CXPB, MUTPB)

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            invalid_ind,fitnesses = toolbox.evaluate(invalid_ind)

            #fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)

            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            hof, pf,history = _update_history_and_hof(hof, pf, history, invalid_ind,gen,MU)

            # Select the next generation population from parents and offspring
            pop = [p for p in pop if len(p.fitness.values) ]
            offspring = [p for p in offspring if len(p.fitness.values) ]
            pool = pop
            pool.extend(offspring)
            if len(pool)>=MU:
                try:
                    pop = toolbox.select(pop + offspring, MU)
                except:
                    pop = toolbox.select(offspring,MU)
            else:
               pop = toolbox.select(pool,MU)
            # Compile statistics about the new population
            record = stats.compile(pop)
            logbook.record(gen=gen, evals=len(invalid_ind), **record)
            pop = [p for p in pop if len(p.fitness.values) ]

            #print(logbook.stream)
    return pop, hof, pf, logbook, history, gen_vs_pop
