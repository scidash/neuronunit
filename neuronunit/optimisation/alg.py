"""Optimisation class
Copyright (c) 2016, EPFL/Blue Brain Project
 This file is part of BluePyOpt <https://github.com/BlueBrain/BluePyOpt>
"""
from deap.benchmarks.tools import diversity, convergence
import deap.tools as tools
from numba import jit
import copy
import logging
import math
import pdb
import pickle
import random
from tqdm import tqdm

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
    return invalid_pop

def cleanse(temp):
    temp = population
    for t in temp:
        if hasattr(t,'dtc'):
            if "ADEXP" in t.dtc.backend or "BHH" in t.dtc.backend:
                del t.dtc 
        # try:
        #     import brian2 as b2
        #     b2.clear_cache("cython")
        #     del b2
        # except:
        #     pass

    #dtc = temp[0].dtc
    if "ADEXP" in temp[0].dtc.backend or "BHH" in temp[0].dtc.backend:
        # try:
        #     import brian2 as b2
        #     b2.clear_cache("cython")
        #     del b2
        # except:
        #     del brian2
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
    for t in temp:
        if hasattr(t,'dtc'):
            if "ADEXP" in t.dtc.backend or "BHH" in t.dtc.backend:
                del t.dtc 
        try:
            import brian2 as b2
            b2.clear_cache("cython")
            del b2
        except:
            pass
    
    else:
        try:
            import brian2 as b2
            b2.clear_cache("cython")
            del b2
        except:
            pass

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
    '''
    try:
        repf = set([ind[0:-1] for ind in pf])
        #repf = [Individual(ind[0:-1]) for ind in repf])

        pf.update(repf)
        print('passed')

    except:
        print('failed')
    # pf = set(pf)
    #halloffame = set(halloffame)
    '''
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

def get_center(pf,pop,toolbox):
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

def update_custom_NSGA2(individuals, k, nd='standard'):
    """Apply NSGA-II selection operator on the *individuals*. Usually, the
    size of *individuals* will be larger than *k* because any individual
    present in *individuals* will appear in the returned list at most once.
    Having the size of *individuals* equals to *k* will have no effect other
    than sorting the population according to their front rank. The
    list returned contains references to the input *individuals*. For more
    details on the NSGA-II operator see [Deb2002]_.

    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select.
    :param nd: Specify the non-dominated algorithm to use: 'standard' or 'log'.
    :returns: A list of selected individuals.

    .. [Deb2002] Deb, Pratab, Agarwal, and Meyarivan, "A fast elitist
       non-dominated sorting genetic algorithm for multi-objective
       optimization: NSGA-II", 2002.
    """
    if nd == 'standard':
        pareto_fronts = sortNondominated(individuals, k)
    elif nd == 'log':
        pareto_fronts = sortLogNondominated(individuals, k)
    else:
        raise Exception('selNSGA2: The choice of non-dominated sorting '
                        'method "{0}" is invalid.'.format(nd))


    for front in pareto_fronts:
        assignCrowdingDist(front)
    return pareto_fronts
    

def sel_custom_NSGA2(pareto_fronts):
    chosen = list(chain(*pareto_fronts[:-1]))
    k = k - len(chosen)
    if k > 0:
        sorted_front = sorted(pareto_fronts[-1], key=attrgetter("fitness.crowding_dist"), reverse=True)
        chosen.extend(sorted_front[:k])

    # need to now.
    # artificially keep the chosen front small.
    # put some of the best solutions back in, even if they are very crowded together.
    # constellation.
    
    return chosen
    '''

class WeightedSumFitness(deap.base.Fitness):

   
    def __init__(self, values=(), obj_size=None):
        self.weights = [1.0] * obj_size if obj_size is not None else [1]

        super(WeightedSumFitness, self).__init__(values)
        #obj_size=len(values)

    @property
    def weighted_sum(self):
        """Weighted sum of wvalues"""
        return sum(self.wvalues)

    @property
    def sum(self):
        """Weighted sum of values"""
        return sum(self.values)

    @property
    def norm(self):
        """Frobenius norm of values"""
        return numpy.linalg.norm(self.values)

    def __le__(self, other):
        return self.weighted_sum <= other.weighted_sum

    def __lt__(self, other):
        return self.weighted_sum < other.weighted_sum

    def __deepcopy__(self, _):
        """Override deepcopy"""

        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result
'''
def best(x):
    weight_names, weight_functions, weight_values = zip(*weights)
    scores = [np.dot(hof.keys[i].values, weight_values)
                for i in range(len(hof))]
    i = np.argmax(scores)
    return np.array([scores[i]] + list(hof.keys[i].values))



def select_best(pop,size):
    sum_list = []
    cnt = 0
    for p in pop:
        sum_list.append((np.sum([v for v in p.fitness.values ]),cnt))
        cnt+=1
    sorted_sum_list = sorted(sum_list,key=lambda tup: tup[0])
    pre_selection = [sel[1] for sel in sorted_sum_list[0:size]]
    selection = [pop[i] for i in pre_selection]
    return selection

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

        def min_fitness(pop):
            sum_list = []
            for i,p in enumerate(pop):
                sum_list.append((np.sum([v for v in p ]),i))
            sorted_sum_list = sorted(sum_list,key=lambda tup: tup[0])
            pre_selection = sorted_sum_list[0][1]            
            selected_fitness_values = pop[pre_selection]
            return selected_fitness_values

        def max_fitness(pop):
            sum_list = []
            for i,p in enumerate(pop):
                sum_list.append((np.sum([v for v in p ]),i))
            sorted_sum_list = sorted(sum_list,key=lambda tup: tup[0])
            pre_selection = sorted_sum_list[-1][1]
            selected_fitness_values = pop[pre_selection]
            return selected_fitness_values


        stats = tools.Statistics(key=lambda ind: ind.fitness.values)

        stats.register("avg", numpy.mean)#, axis=0)
        stats.register("std", numpy.std)#, axis=0)
        stats.register("min", min_fitness)
        stats.register("max", max_fitness)

        logbook = tools.Logbook()
        logbook.header = "gen", "evals", "std", "min", "avg", "max"
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]


        invalid_ind,fitnesses = toolbox.evaluate(invalid_ind)

        #weighted_fitnesses = []       
        #for f in fitnesses:
        #    weighted_fitnesses.append(WeightedSumFitness(values=f))
    
        
        for ind, fit in zip(invalid_ind, fitnesses): 
            ind.fitness.values = fit
        

        # Compile statistics about the population
        pop = [p for p in pop if len(p.fitness.values) ]
        record = stats.compile(pop)
        logbook.record(gen=0, evals=len(invalid_ind), **record)
        #temp_pop = copy.copy(pop)
        
        hof, pf,history = _update_history_and_hof(hof, pf, history, pop ,0,MU)

        # Begin the generational process
        
        for gen in tqdm(range(1, NGEN), desc='GA Generation Progress'):

            #offspring = tools.selTournamentDCD(pop, len(pop))
            
            #offspring = toolbox.select(pop, int(MU/2))
            offspring = select_best(pop,int(MU/2))
            offspring = [toolbox.clone(ind) for ind in offspring]
            cnt = 0
            # using the line below would lead to higher gene mutation
            # but less stabile increases in error score.
	        # for ind1, ind2 in zip(offspring[::int(MU/4)], offspring[1::int(MU/4)]):
            temp = select_best(offspring,1)
            offspring.insert(0,temp[0]) # force the best gene to always bread.

            # [::4] means slicing list in reverse.
            # means start at beggining of list.
            for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
                toolbox.mate(ind1, ind2)
                toolbox.mutate(ind1)
                toolbox.mutate(ind2)
                del ind1.fitness.values, ind2.fitness.values
                cnt+=1
            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            invalid_ind,fitnesses = toolbox.evaluate(invalid_ind)
            record = stats.compile(pop)
            logbook.record(gen=gen, evals=len(invalid_ind), **record)
            #weighted_fitnesses = []       
            #for f in fitnesses:
            #   weighted_fitnesses.append(WeightedSumFitness(values=f))
                    
            for ind, fit in zip(invalid_ind, fitnesses): 
                ind.fitness.values = fit

            pool = pop
            pool.extend(offspring)
            
            pool = remove_and_replace(pool,toolbox)
            pop = select_best(pool,MU)
            #fronts = update_custom_NSGA2(pop)
            #if gen == NGEN:
            #    custom_front =sel_custom_NSGA2(fronts)
            #pop = toolbox.select(pool, MU)

            hof, pf,history = _update_history_and_hof(hof, pf, history, pop ,gen,MU)
    return pop, hof, pf, logbook, history, gen_vs_pop
