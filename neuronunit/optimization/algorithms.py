"""Optimisation class
Copyright (c) 2016, EPFL/Blue Brain Project

 This file is part of BluePyOpt <https://github.com/BlueBrain/BluePyOpt>

 This library is free software; you can redistribute it and/or modify it under
 the terms of the GNU Lesser General Public License version 3.0 as published
 by the Free Software Foundation.

 This library is distributed in the hope that it will be useful, but WITHOUT
 ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 details.

 You should have received a copy of the GNU Lesser General Public License
 along with this library; if not, write to the Free Software Foundation, Inc.,
 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
"""


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
from deap.tools import selNSGA2

from neuronunit.optimisation import optimisation_management as om

logger = logging.getLogger('__main__')


def _evaluate_invalid_fitness(toolbox, population):
    '''Evaluate the individuals with an invalid fitness

    Returns the count of individuals with invalid fitness
    '''
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    invalid_pop,fitnesses = toolbox.evaluate(invalid_ind)
    for j, ind in enumerate(invalid_pop):
        ind.fitness.values = fitnesses[j]
        ind.dtc.get_ss()


    return invalid_pop


def _update_history_and_hof(halloffame,pf, history, population,td):
    '''Update the hall of fame with the generated individuals

    Note: History and Hall-of-Fame behave like dictionaries
    '''

    if halloffame is not None:
        halloffame.update(population)
    if history is not None:
        history.update(population)

    if pf is not None:
        try:
            pf.update(population)
        except:
            pass

    return (halloffame,pf,history)


def _record_stats(stats, logbook, gen, population, invalid_count):
    '''Update the statistics with the new population'''
    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=gen, nevals=invalid_count, **record)

def gene_bad(offspring):
    gene_bad = False
    for o in offspring:
        if np.any(np.isnan(o)) or np.any(np.isinf(o)):
            gene_bad = True
    return gene_bad

def _get_offspring(parents, toolbox, cxpb, mutpb):
    '''return the offsprint, use toolbox.variate if possible'''
    if hasattr(toolbox, 'variate'):
        offspring = toolbox.variate(parents, toolbox, cxpb, mutpb)
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
        selection = 'selNSGA2',
        td=None):
    gen_vs_pop = []

    if continue_cp:
        # A file name has been given, then load the data from the file
        cp = pickle.load(open(cp_filename, "r"))
        population = cp["population"]
        parents = cp["parents"]
        start_gen = cp["generation"]
        halloffame = cp["halloffame"]
        logbook = cp["logbook"]
        history = cp["history"]
        random.setstate(cp["rndstate"])
    else:
        # Start a new evolution
        start_gen = 1
        parents = population#[:]

        gen_vs_pop.append(population)
        logbook = deap.tools.Logbook()
        logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
        history = deap.tools.History()

        # TODO this first loop should be not be repeated !
        parents = _evaluate_invalid_fitness(toolbox, population)

        invalid_count = len(parents)
        gen_vs_hof = []
        hof, pf,history = _update_history_and_hof(hof, pf, history, parents, td)

        gen_vs_hof.append(hof)
        _record_stats(stats, logbook, start_gen, parents, invalid_count)
    toolbox.register("select",selNSGA2)
    fronts = []

    # Begin the generational    process
    for gen in range(start_gen + 1, ngen + 1):
        delta = len(parents[0]) - len(toolbox.Individual())
        #h = copy.copy(history)
        #temp = [ v for v in h.genealogy_history.values()]

        offspring = _get_offspring(parents, toolbox, cxpb, mutpb)
        offspring = [ toolbox.clone(ind) for ind in offspring ]

        assert len(offspring)>0
        gen_vs_pop.append(offspring)
        invalid_ind = _evaluate_invalid_fitness(toolbox, offspring)
        # something in evaluate fitness has knocked out fitness
        population = parents + invalid_ind
        population = [ p for p in population if len(p.fitness.values)!=0 ]
        invalid_count = len(invalid_ind)
        hof, pf,history = _update_history_and_hof(hof, pf, history, offspring, td)
        fronts.append(pf)
        if gen%14==0:# every tenth count.
            import pdb
	    pdb.set_trace()
            #            stag_check = [ p for p in stag_check if len(p.fitness.values)!=0 ]
            stag_check = np.mean([ p.dtc.get_ss() for p in fronts])
            if not pf[0].dtc.get_ss() < stag_check*0.90:
                print('gene poulation stagnant, no appreciable gains in fitness')
                break
	    else:
		fronts = [] #purge the list and check for stagnation 10 cycles later
        #for k,v in history.items()
        # TODO recreate a stagnation termination criterion.
        # if genes don't get substantially better in 50 generations.
        # Stop.


        if pf[0].dtc is not None:
            print('true minimum',pf[0].dtc.get_ss())
        elif hof[0].dtc is not None:
            print('true minimum',hof[0].dtc.get_ss())



        _record_stats(stats, logbook, gen, offspring, invalid_count)

        if str('selIBEA') == selection:
            if hof[0].fitness.values is None or len(hof[0].fitness.values)==0:
                best,fit = toolbox.evaluate(hof[0:1])
                best.fitness.values = fit
                best.dtc.get_ss()
                if np.sum(best.dtc.get_ss()) != 0:
                    print('true minimum',np.sum(hof[0].fitness.values))
                    population.append(hof[0])
            toolbox.register("select", tools.selIBEA)

        if str('selNSGA') == selection:
            if pf[0].fitness.values is None or len(pf[0].fitness.values)==0:
                best,fit = toolbox.evaluate(pf[0:1])
                best.fitness.values = fit
                best.dtc.get_ss()
                if np.sum(best.dtc.get_ss()) != 0:
                    print('true minimum',np.sum(pf[0].fitness.values))
                    print(best.dtc.get_ss())
                    population.append(best[0])
        population = [ p for p in population if p if len(p.fitness.values)!=0]

        toolbox.register("select",selNSGA2)
        parents = toolbox.select(population, mu)

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
    '''
    two = copy.copy(_get_elite(temp, 2))
    worst = copy.copy(_get_elite(temp, 1))

    # make new genes that are in the middle of the two best.
    mean_best = [ (i+j)/2.0 for i,j in zip(two[0],two[1]) ]
    mean_worst = [ (i+j)/2.0 for i,j in zip(worst[0],two[1]) ]

    one = copy.copy(parents[0])
    for ind, o in enumerate(one):
        o = mean_best[ind]
    offspring.append(one)

    one_ = _get_elite(temp, 1)[0]
    '''
