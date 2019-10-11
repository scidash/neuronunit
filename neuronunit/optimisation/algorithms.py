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
    invalid_pop,fitnesses = toolbox.evaluate(invalid_ind)
    
    for j, ind in enumerate(invalid_pop):
        ind.fitness.values = fitnesses[j]
        ind.dtc.get_ss()

    
    return invalid_pop

def strip_object(p):
    state = super(type(p)).state
    p.unpicklable = []
    print('i suspect a brian module is in the deap gene, and it should not be')
    import pdb
    pdb.set_trace()
    return p._state(state=state, exclude=['unpicklable','verbose'])


def _update_history_and_hof(halloffame,pf, history, population,td,mu):
    '''Update the hall of fame with the generated individuals

    Note: History and Hall-of-Fame behave like dictionaries
    '''

    if halloffame is not None:
        try:
            
            halloffame.update(population[0:mu])
        except:
            print('mostly not relevant')
    if history is not None:
        try:
            history.update(population[0:mu])
        except:
            for p in population:
                print(p.dtc.from_imputation, 'from imputation')
                #p = strip_object(p)
            pass
    if pf is not None:
        for ind in population:
            for i,j in enumerate(ind):
                ind[i] = float(j)
        try:
            pf.update(population[0:mu])
        except:
            pass
            #import pdb
            #pdb.set_trace()
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
        offspring = deap.algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

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
        # Start a new evolution
        start_gen = 1
        #parents = population#[:]

        gen_vs_pop.append(population)
        logbook = deap.tools.Logbook()
        logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
        history = deap.tools.History()
        toolbox.register("select", tools.selNSGA2)
        def experimental():
            NOBJ = len(population[0].fitness.values)
            #print('stuck_here uniform')
            import pdb; pdb.set_trace()
        #experimental()
        # TODO this first loop should be not be repeated !
        parents = _evaluate_invalid_fitness(toolbox, population)
        #import pdb; pdb.set_trace()
        invalid_count = len(parents)
        gen_vs_hof = []
        hof, pf,history = _update_history_and_hof(hof, pf, history, parents, td,mu)

        gen_vs_hof.append(hof)
        _record_stats(stats, logbook, start_gen, parents, invalid_count)
    #toolbox.register("select",selNSGA2)
    #from deap.tools import selNSGA3

    fronts = []

    # Begin the generational    process
    for gen in range(start_gen + 1, ngen + 1):
        delta = len(parents[0]) - len(toolbox.Individual())

        offspring = _get_offspring(parents, toolbox, cxpb, mutpb)
        offspring = [ toolbox.clone(ind) for ind in offspring ]
        _record_stats(stats, logbook, gen, offspring, invalid_count)

        assert len(offspring)>0
        gen_vs_pop.append(offspring)

        fitness = [ np.sum(list(i[0].fitness.values)[0]) for i in gen_vs_pop if len(i[0].fitness.values)>0 ]
        #if gen>1:
        rec_lenf = [ i for i in range(0,len(fitness))]
        try:
            scores = [ list(i[0].dtc.scores.values())[0] for i in gen_vs_pop]

            rec_len = [ i for i in range(0,len(scores))]
                    
        except:
            dtcs_ = [j.dtc for i in gen_vs_pop for j in i]
            
        names = offspring[0].dtc.scores.keys()
        if gen>1:
            if str('rec_len') in locals().keys():
                try:
                    fig = apl.figure()
                
                    fig.plot(rec_lenf,fitness, label=str('evolution fitness: '), width=100, height=20)
                    fitness1 = [ np.sum(list(i[0].fitness.values)) for i in gen_vs_pop if len(i[0].fitness.values)>1 ]
                    fig.plot(rec_lenf,fitness1, label=str('evolution fitness: '), width=100, height=20)

                    fig.show()
                    front = [ np.sum(list(i.dtc.scores.values())) for i in ga_out['pf']]
                    front_lens = [ i for i in range(0,len(front))]

                    fig.plot(front_lens,front, label=str('pareto front: '), width=100, height=20)
                    fig.show()
                except:
                    pass
                    try:
                        fig.plot(rec_len,scores, label=str('evolution scores: '), width=100, height=20)
                        fig.show()
                    except:
                        pass

            else:
                print(fitness)
        #    pass

        invalid_ind = _evaluate_invalid_fitness(toolbox, offspring)
        # something in evaluate fitness has knocked out fitness
        population = parents + invalid_ind
        population = [ p for p in population if len(p.fitness.values)!=0 ]
        invalid_count = len(invalid_ind)

        fronts.append(pf)
        if gen%10==0:
            stag_check1 = np.mean([ p.dtc.get_ss() for pf in fronts for p in pf ])
            if not np.sum(fronts[-1][0].dtc.get_ss()) < stag_check1*0.975:
                print('gene poulation stagnant, no appreciable gains in fitness')
                #return population, hof, pf, logbook, history, gen_vs_pop

        
        try:
            ref_points = tools.uniform_reference_points(len(population[0]), 12)
            #toolbox.register("select", tools.selNSGA3WithMemory, ref_points=population)
            toolbox.register("select", selNSGA3WithMemory(ref_points))
        except:
            #toolbox.register("select",selNSGA2)
            toolbox.register("select", tools.selNSGA2)

        old_max = 0
        for ind in population:
            if len(ind.fitness.values) > old_max:
                old_max = len(ind.fitness.values)
        population = [ ind for ind in population if ind if ind.fitness.values is not type(None) ]

        popp = [ i for i in population if len(i.fitness.values)==old_max ]
        try:
            parents = toolbox.select(popp, mu)
        except:

            parents = toolbox.select(popp, len(population))

        # make new genes that are in the middle of the best and worst.
        # make sure best gene breeds.

        hof, pf,history = _update_history_and_hof(hof, pf, history, parents, td, mu)

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
