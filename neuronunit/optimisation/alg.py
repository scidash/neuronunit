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
from sklearn.cluster import KMeans
import numpy as np

from neuronunit.optimisation import optimization_management as om


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

def cleanse(temp):
    dtc = temp[0].dtc
    if "ADEXP" in temp[0].dtc.backend or "BHH" in temp[0].dtc.backend:
        try:
            import brian as b2
            b2.clear_cache("cython")
            del b2
        except:
            del brian2
        OM = dtc.dtc_to_opt_man()
        temp_,_ = OM.boot_new_genes(len(temp),dtcpop)
        for i,t in enumerate(temp):
            temp_[i].fitness = t.fitness
            for x,j in enumerate(t):
                temp_[i][x] = j 
        temp = temp_
        for t in temp:
            del t.dtc.tests #= None
            del t.dtc #= None
    return temp

def _update_history_and_hof(halloffame,pf, history, population,GEN,MU):
    '''Update the hall of fame with the generated individuals

    Note: History and Hall-of-Fame behave like dictionaries
    '''
    temp = population

    if halloffame is not None:
        try:
            halloffame.update(temp)
        except:
            temp = cleanse(temp)
            halloffame.update(temp)
    if history is not None:
        try:
            history.update(temp)
        except:
            temp = cleanse(temp)
            history.update(temp)
    if pf is not None:
        if GEN ==0:
            pf = deap.tools.ParetoFront()
        try:
            pf.update(temp)
        except:
            temp = cleanse(temp)
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


def remove_and_replace(pool,toolbox):    
    '''
    Remove and replace genes rheobase == None, unstable bifurcation.
    '''
    filtered = [p for p in pool if len(p.fitness.values)==0]
    pool = [p for p in pool if len(p.fitness.values)!=0]
    if len(filtered):
        if True:
            '''
            Amplifies number genes that work at expense of gene diversity.
            '''
            for i in range(0,len(filtered)):
                ind = pool[i]
                ind = toolbox.clone(ind)
                pool.append(ind)

        if False:
            '''
            Trys to re-evaluate broken gene/model.
            '''
            invalid_ind_,fitnesses_ = toolbox.evaluate(filtered)
            for ind, fit in zip(invalid_ind_, fitnesses_):
                assert len(fit) != 0
                ind.fitness.values = fit
                pool.append(ind)

    last = len(pool[0].fitness.values)
    for p in pool: 
        current = len(p.fitness.values)
        assert current==last
        last = current
    return pool

def get_center(pf,pop):
    """ 
    for use in scope of this file object.
    """
    X = np.array(pf[0:-1])
    kmeans = KMeans(n_clusters=1, random_state=0).fit(X)
    piercing = kmeans.cluster_centers_
    pierce = [toolbox.clone(ind) for ind in pop][0]
    pierce.rheobase = None
    for j,param in enumerate(pierce):
        pierce[j] = piercing[0][j]
    temp_invalid_ind,temp_fitnesses = toolbox.evaluate([pierce])
    temp_invalid_ind[0].fitness.values = temp_fitnesses[0]
    pop.append(temp_invalid_ind[0])
    return pop
import copy
def get_center_nb(pf,pop,OM):
    """
    same as above but works out of scope, like in notebooks
    """
    X = np.array(pf[0:-1])
    kmeans = KMeans(n_clusters=1, random_state=0).fit(X)
    piercing = kmeans.cluster_centers_
    pierce = copy.copy(pop[0])
    for j,param in enumerate(pierce):
        pierce[j] = piercing[0][j]
    OM.td = results['td']
    dtc = OM.update_dtc_pop([pierce])[0]
    dtc.tests = OM.tests
    dtc.self_evaluate()
    dtc.SA
    return dtc


from tqdm import tqdm


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
        cp_frequency = 20,
        cp_filename = 'big_run.p',
        continue_cp = False,
        selection = 'selNSGA2',
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
        hof, pf,history = _update_history_and_hof(hof, pf, history, pop ,0,MU)



        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit


        # Compile statistics about the population
        pop = [p for p in pop if len(p.fitness.values) ]

        record = stats.compile(pop)
        logbook.record(gen=0, evals=len(invalid_ind), **record)
        temp_pop = copy.copy(pop)
        for p in pop:
            assert hasattr(p,'dtc')



        # Begin the generational process
        
        for gen in tqdm(range(1, NGEN), desc='GA Generation Progress'):
            #offspring = tools.selTournamentDCD(pop, len(pop))
            offspring = toolbox.select(pop, int(MU/2))
            offspring = [toolbox.clone(ind) for ind in offspring]
            #CXPB = 0.9
            cnt = 0
            # using the line below would lead to higher gene mutation
            # but less stabile increases in error score.
	    # for ind1, ind2 in zip(offspring[::int(MU/4)], offspring[1::int(MU/4)]):
            for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
                toolbox.mate(ind1, ind2)
                toolbox.mutate(ind1)
                toolbox.mutate(ind2)
                del ind1.fitness.values, ind2.fitness.values
                cnt+=1
            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            invalid_ind,fitnesses = toolbox.evaluate(invalid_ind)

            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
                assert len(fit) != 0

            pool = pop
            pool.extend(offspring)
            
            pool = remove_and_replace(pool,toolbox)

            pop = toolbox.select(pool, MU)
            record = stats.compile(pop)
            logbook.record(gen=gen, evals=len(invalid_ind), **record)
            
            hof, pf,history = _update_history_and_hof(hof, pf, history, pop ,gen,MU)
            if(cp_filename and cp_frequency and gen % cp_frequency == 0):
                cp = dict(population=population,
                        generation=gen,
                        parents=parents,
                        halloffame=hof,
                        history=history,
                        logbook=logbook,
                        rndstate=random.getstate())
                pickle.dump(cp, open(cp_filename, "wb"))
                #print('Wrote checkpoint to %s', cp_filename)
                logger.debug('Wrote checkpoint to %s', cp_filename)        

            #print(logbook.stream)
    hof, pf,history = _update_history_and_hof(hof, pf, history, invalid_ind,gen,MU)
    return pop, hof, pf, logbook, history, gen_vs_pop
