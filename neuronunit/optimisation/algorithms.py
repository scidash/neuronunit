"""Optimisation class
Copyright (c) 2016, EPFL/Blue Brain Project
 This file is part of BluePyOpt <https://github.com/BlueBrain/BluePyOpt>
"""

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

from neuronunit.optimisation import optimization_management as om
#from neuronunit.optimisation.optimization_management import WSListIndividual
class WSListIndividual(list):
    """Individual consisting of list with weighted sum field"""
    def __init__(self, *args, **kwargs):
        """Constructor"""
        self.rheobase = None
        super(WSListIndividual, self).__init__(*args, **kwargs)

logger = logging.getLogger('__main__')
try:
    import asciiplotlib as apl
except:
    pass

def _evaluate_invalid_fitness(toolbox, population):
    '''Evaluate the individuals with an invalid fitness

    Returns the count of individuals with invalid fitness
    '''
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    #print(invalid_ind[0].dtc.scores)
    #print(invalid_ind[0].dtc.from_imputation)
    invalid_pop,fitnesses = toolbox.evaluate(invalid_ind)

    for j, ind in enumerate(invalid_pop):
        ind.fitness.values = fitnesses[j]
        ind.dtc = None
    return invalid_pop

def strip_object(p):
    state = super(type(p)).state
    p.unpicklable = []
    #print('i suspect a brian module is in the deap gene, and it should not be')
    #import pdb
    #pdb.set_trace()
    return p._state(state=state, exclude=['unpicklable','verbose'])

def purify2(population):
    pop2 = []
    for ind in population:
        for i,j in enumerate(ind):
            ind[i] = float(j)
        pop2.append(ind)
    return pop2

def _update_history_and_hof(halloffame,pf, history, population,td,mu):
    '''Update the hall of fame with the generated individuals

    Note: History and Hall-of-Fame behave like dictionaries
    '''
    temp = copy.copy(population)

    if halloffame is not None:
        try:

            halloffame.update(temp)
        except:
            temp = purify(temp)
            halloffame.update(temp)

    if history is not None:
        try:
            history.update(temp)
        except:
            temp = purify(temp)
            history.update(temp)
    if pf is not None:
        try:
            pf.update(temp[0:mu])
        except:
            temp = purify2(temp)
            temp = purify(temp)
            try:
                pf.update(temp[0:mu])
            except:
                print(len(temp))
    return (halloffame,pf,history)


def _record_stats(stats, logbook, gen, population, invalid_count):
    '''Update the statistics with the new population'''
    stats_fit = tools.Statistics(key=lambda ind: ind.fitness.sum)
    everything = tools.Statistics(key=lambda ind: ind.fitness.values)
    mstats = tools.MultiStatistics(fitness=stats_fit, every=everything)#,stats_size=stats_size)

    mstats.register("avg", np.mean, axis=0)
    mstats.register("std", np.std, axis=0)
    mstats.register("min", np.min, axis=0)
    mstats.register("max", np.max, axis=0)

    record = mstats.compile(copy.copy(population)) if mstats is not None else {}
    logbook.record(gen=gen, nevals=invalid_count, **record)

def gene_bad(offspring):
    gene_bad = False
    for o in offspring:
        if np.any(np.isnan(o)) or np.any(np.isinf(o)):
            gene_bad = True
    return gene_bad
def purify(parents):
    '''hygeinise
    strip off extra objects that might contain modules
    which are not pickle friendly
    '''
    initial_length = len(parents)
    if type(parents[0].dtc) is not type(None):
        imp = [o for o in parents if o.dtc.from_imputation==True]
        parents = [o for o in parents if o.dtc.from_imputation==False]
        subsequent = len(parents)
        delta = initial_length - subsequent
        for i in range(0,delta):
            ind = copy.copy(parents[0])
            for x,y in enumerate(ind):
                ind[x] = copy.copy(imp[i][x])
                ind[x].fitness = imp[i].fitness
                ind[x].rheobase = imp[i].rheobase
                parents.append(ind)
        parents_ = []
        for i,off_ in enumerate(parents):
            parents_.append(WSListIndividual(off_,obj_size=len(off_)))
            parents_[-1].fitness = off_.fitness
            parents_[-1].rheobase = off_.rheobase
    else:
        parents_ = []
        for i,off_ in enumerate(parents):
            parents_.append(WSListIndividual())
            for j in off_:
                parents_[-1].append(float(j))
                parents_[-1].fitness = off_.fitness
                parents_[-1].rheobae = off_.rheobase
    return parents_

def _get_offspring(parents, toolbox, cxpb, mutpb):
    '''return the offsprint, use toolbox.variate if possible'''
    if hasattr(toolbox, 'variate'):

        try:
            offspring = toolbox.variate(parents, toolbox, cxpb, mutpb)
            offspring = deap.algorithms.varAnd(parents, toolbox, cxpb, mutpb)

        except:
            parents_ = []
            parents = purify(parents)

            for i,off_ in enumerate(parents):
                parents_.append(WSListIndividual())
                for j in off_:
                    parents_[-1].append(float(j))
                    parents_[-1].fitness = off_.fitness

                parents = parents_
                #parents.append(WSListIndividual(off_,obj_size=len(off_)))

            offspring = toolbox.variate(parents, toolbox, cxpb, mutpb)
            offspring = deap.algorithms.varAnd(parents, toolbox, cxpb, mutpb)

        while gene_bad(offspring) == True:
            offspring = deap.algorithms.varAnd(parents, toolbox, cxpb, mutpb)

    return offspring

def _get_worst(halloffame, nworst):

    if nworst > 0 and halloffame is not None:

        normsorted_idx = numpy.argsort([ind.fitness.norm for ind in halloffame])
        #hofst = [(sum(h.dtc.scores.values()),h.dtc) for h in halloffame ]
        #ranked = sorted(hofst, key=lambda w: w[0],reverse = True)
        return [halloffame[idx] for idx in normsorted_idx[-nworst::]]
    else:
        return list()

def _get_elite(halloffame, nelite):

    if nelite > 0 and halloffame is not None:
        normsorted_idx = numpy.argsort([ind.fitness.norm for ind in halloffame])
        #hofst = [(sum(h.dtc.scores.values()),h.dtc) for h in halloffame ]
        #ranked = sorted(hofst, key=lambda w: w[0],reverse = True)
        return [halloffame[idx] for idx in normsorted_idx[:nelite]]
    else:
        return list()

def prune_constants(parents,num_constants):
    for i in range(0,num_constants):
        for p in parents:
            del p[-1]
    return parents

def eaAlphaMuPlusLambdaCheckpoint(
        population,
        toolbox,
        mu,
        cxpb,
        mutpb,
        ngen,
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
        population = cp["population"]
        parents = cp["parents"]
        start_gen = cp["generation"]
        hof = cp["halloffame"]
        logbook = cp["logbook"]
        history = cp["history"]
        random.setstate(cp["rndstate"])

    else:
        start_gen = 1
        gen_vs_pop.append(population)
        logbook = deap.tools.Logbook()
        logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
        history = deap.tools.History()
        toolbox.register("select", tools.selNSGA2)

        parents = _evaluate_invalid_fitness(toolbox, population)
        invalid_count = len(parents)
        gen_vs_hof = []
        hof, pf,history = _update_history_and_hof(hof, pf, history, parents, td,mu)

        gen_vs_hof.append(hof)
        _record_stats(stats, logbook, start_gen, parents, invalid_count)
    stag_check1 = []
    fronts = []

    # Begin the generational    process
    for gen in range(start_gen + 1, ngen + 1):
        delta = len(parents[0]) - len(toolbox.Individual())
        _record_stats(stats, logbook, gen, parents, invalid_count)
        offspring = _get_offspring(parents, toolbox, cxpb, mutpb)
        offspring = [ toolbox.clone(ind) for ind in offspring ]
        assert len(offspring)>0
        gen_vs_pop.append(offspring)
        fitness = [ np.sum(list(i[0].fitness.values)[0]) for i in gen_vs_pop if len(i[0].fitness.values)>0 ]
        rec_lenf = [ i for i in range(0,len(fitness))]

        invalid_ind = _evaluate_invalid_fitness(toolbox, offspring)
        population = parents + invalid_ind
        population = [ p for p in population if len(p.fitness.values)!=0 ]
        invalid_count = len(invalid_ind)
        fronts.append(pf)
        stag_check1.append(np.mean([ np.sum(p.fitness.values) for pf in fronts for p in pf ]))
        if gen%5==0:
            if not np.sum(fronts[-1][0].fitness.values) < stag_check1[gen-4]*0.75:
                print(np.std(population))#<record:
                print('gene poulation stagnant, no appreciable gains in fitness')
        try:
            ref_points = tools.uniform_reference_points(len(population[0]), 12)
            toolbox.register("select", selNSGA3WithMemory(ref_points))
        except:
            toolbox.register("select", tools.selNSGA2)

        old_max = 0
        for ind in population:
            if len(ind.fitness.values) > old_max:
                old_max = len(ind.fitness.values)
        population = [ ind for ind in population if ind if ind.fitness.values is not type(None) ]

        pop = [ i for i in population if len(i.fitness.values)==old_max ]

        #hof, pf,history = _update_history_and_hof(hof, pf, history, pop, td, mu)
        try:
            parents = toolbox.select(pop, mu)
        except:

            parents = toolbox.select(pop, len(population))

        hof, pf,history = _update_history_and_hof(hof, pf, history, parents, td, mu)
        # make new genes that are in the middle of the best and worst.
        # make sure best gene breeds.


        logger.info(logbook.stream)

        if(cp_filename):# and cp_frequency and
           #gen % cp_frequency == 0):
            cp = dict(population=population,
                      generation=gen,
                      parents=parents,
                      halloffame=hof,
                      history=history,
                      logbook=logbook,
                      rndstate=random.getstate())
            pickle.dump(cp, open(cp_filename, "wb"))
            print('Wrote checkpoint to %s', cp_filename)
            logger.debug('Wrote checkpoint to %s', cp_filename)

    return population, hof, pf, logbook, history, gen_vs_pop
