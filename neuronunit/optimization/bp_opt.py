"""Optimisation class"""

"""
Copyright (c) 2016, EPFL/Blue Brain Project

 This file is part of BluePyOpt <https://github.com/BlueBrain/BluePyOpt>

 This library is free software; you can redistribute it and/or modify it under
 the terms of the Lesser General Public License version 3.0 as published
 by the Free Software Foundation.

 This library is distributed in the hope that it will be useful, but WITHOUT
 ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 details.

 You should have received a copy of the GNU Lesser General Public License
 along with this library; if not, write to the Free Software Foundation, Inc.,
 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
"""

# pylint: disable=R0912, R0914
from neuronunit.optimization import optimization_management

import random
import logging
import functools
import numpy

import deap
import deap.base
import deap.algorithms
import deap.tools

from . import algorithms
from bluepyopt.deapext.optimisations import tools

import numpy
from numba import jit
logger = logging.getLogger('__main__')

# TODO decide which variables go in constructor, which ones go in 'run' function
# TODO abstract the algorithm by creating a class for every algorithm, that way
# settings of the algorithm can be stored in objects of these classes

from neuronunit.optimization.optimization_management import evaluate, update_deap_pop
from neuronunit.optimization import optimization_management
import numpy as np

class WeightedSumFitness(deap.base.Fitness):

    """Fitness that compares by weighted sum"""

    def __init__(self, values=(), obj_size=None):
        self.weights = [-1.0] * obj_size if obj_size is not None else [-1]

        super(WeightedSumFitness, self).__init__(values)

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


class WSListIndividual(list):

    """Individual consisting of list with weighted sum field"""

    def __init__(self, *args, **kwargs):
        """Constructor"""
        self.fitness = WeightedSumFitness(obj_size=kwargs['obj_size'])
        self.dtc = None
        self.rheobase = None
        del kwargs['obj_size']
        super(WSListIndividual, self).__init__(*args, **kwargs)

    def set_fitness(self,obj_size):
        self.fitness = WeightedSumFitness(obj_size=obj_size)


class WSFloatIndividual(float):
    """Individual consisting of list with weighted sum field"""
    def __init__(self, *args, **kwargs):
        """Constructor"""
        self.dtc = None
        self.rheobase = None
        super(WSFloatIndividual, self).__init__()

    def set_fitness(self,obj_size):
        self.fitness = WeightedSumFitness(obj_size=obj_size)

import bluepyopt.optimisations
class SciUnitOptimization(bluepyopt.optimisations.Optimisation):
    
    """DEAP Optimisation class"""
    def __init__(self, error_criterion = None, evaluator = None,
                 selection = 'selIBEA',
                 benchmark = False,
                 seed=1,
                 offspring_size=15,
                 elite_size=3,
                 eta=10,
                 mutpb=1.0,
                 cxpb=1.0,
                 map_function=None,
                 backend=None,
                 nparams = 10,
                 provided_dict= {}):
        """Constructor"""

        super(SciUnitOptimization, self).__init__()
        self.selection = selection
        self.benchmark = benchmark

        self.error_criterion = error_criterion
        self.seed = seed
        self.offspring_size = offspring_size
        self.elite_size = elite_size
        self.eta = eta
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.backend = backend
        # Create a DEAP toolbox
        self.toolbox = deap.base.Toolbox()
        self.setnparams(nparams = nparams, provided_dict = provided_dict)
        self.setup_deap()
        #assert len(self.params.items()) == 3
        #assert len(self.pop.dtc.attrs.items()) == 3

    def transdict(self,dictionaries):
        from collections import OrderedDict
        mps = OrderedDict()
        sk = sorted(list(dictionaries.keys()))
        for k in sk:
            mps[k] = dictionaries[k]
        tl = [ k for k in mps.keys() ]
        return mps, tl

    def setnparams(self, nparams = 10, provided_dict = None):
        self.params = optimization_management.create_subset(nparams = nparams,provided_dict = provided_dict)
        self.nparams = len(self.params)
        not_list , self.td = self.transdict(self.params)
        import pdb
        pdb.set_trace()
        return self.params, self.td


    def set_evaluate(self):
        if self.benchmark == True:
            self.toolbox.register("evaluate", benchmarks.zdt1)
        else:
            self.toolbox.register("evaluate", optimization_management.evaluate)


    def grid_sample_init(self, nparams):
        from neuronunit.optimization import exhaustive_search as es
        npoints = self.offspring_size ** (1.0/len(list(self.params)))
        npoints = np.ceil(npoints)
        nparams = len(self.params)
        provided_keys = list(self.params.keys())
        dic_grid, _ = es.create_grid(npoints = npoints, provided_keys = self.params)
        delta = int(np.abs(len(dic_grid) - (npoints ** len(list(self.params)))))
        pop = []


        for dg in dic_grid:
            temp = list(dg.values())
            pop.append(temp)
            
        for d in range(0,delta):
            impute = []
            for i in range(0,len(pop[0])):
                impute.append(np.mean([ p[i] for p in pop ]))
            pop.append(impute)
        assert len(pop) == int(npoints ** len(list(self.params)))

        return pop

    def setup_deap(self):
        """Set up optimisation"""
        # Set random seed
        random.seed(self.seed)

        # Eta parameter of crossover / mutation parameters
        # Basically defines how much they 'spread' solution around
        # The lower this value, the more spread
        ETA = self.eta

        # Number of parameters
        # Bounds for the parameters
        IND_SIZE = len(list(self.params.values()))

        OBJ_SIZE = len(self.error_criterion)
        LOWER = [ np.min(self.params[v]) for v in self.td ]
        UPPER = [ np.max(self.params[v]) for v in self.td ]

        if self.backend == 'glif':
            for index, i in enumerate(UPPER):
                if i == LOWER[index]:
                    LOWER[index]-=2.0
                    i+=2.0

        self.grid_init = self.grid_sample_init(self.params)#(LOWER, UPPER, self.offspring_size)

        def uniform_params(lower_list, upper_list, dimensions):
            if hasattr(lower_list, '__iter__'):
                other = [random.uniform(lower, upper) for lower, upper in zip(lower_list, upper_list)]
            else:
                other = [random.uniform(lower_list, upper_list)
                    for _ in range(dimensions)]
            return other
        # Register the 'uniform' function
        self.toolbox.register("uniform_params", uniform_params, LOWER, UPPER, IND_SIZE)



        self.toolbox.register(
            "Individual",
            deap.tools.initIterate,
            functools.partial(WSListIndividual, obj_size=OBJ_SIZE),
            self.toolbox.uniform_params)

        # Register the population format. It is a list of individuals
        self.toolbox.register(
            "population",
            deap.tools.initRepeat,
            list,
            self.toolbox.Individual)


        # Register the evaluation function for the individuals
        #@jit
        def custom_code(invalid_ind, as_log=None):
            if type(as_log) is not type(None):
                for p in invalid_ind:
                    for gene in p:
                        gene = np.log(gene)

            if self.backend is None:
                invalid_pop = update_deap_pop(invalid_ind, self.error_criterion, td = self.td)
            else:
                
                invalid_pop = update_deap_pop(invalid_ind, self.error_criterion, td = self.td, backend = self.backend)
            assert len(invalid_pop) != 0
            invalid_dtc = [ i.dtc for i in invalid_pop if hasattr(i,'dtc') ]
            fitnesses = list(map(evaluate, invalid_dtc))
            return (invalid_pop,fitnesses)

        self.toolbox.register("evaluate", custom_code)
        # Register the mate operator
        self.toolbox.register(
            "mate",
            deap.tools.cxSimulatedBinaryBounded,
            eta=ETA,
            low=LOWER,
            up=UPPER)

        # Register the mutation operator
        self.toolbox.register(
            "mutate",
            deap.tools.mutPolynomialBounded,
            eta=ETA,
            low=LOWER,
            up=UPPER,
            indpb=0.5)

        # Register the variate operator
        self.toolbox.register("variate", deap.algorithms.varAnd)

        #self.toolbox.register("select", tools.selIBEA)

    #@jit    
    def set_pop(self):
        IND_SIZE = len(list(self.params.values()))
        OBJ_SIZE = len(self.error_criterion)
        if IND_SIZE == 1:
            pop = [ WSListIndividual([g],obj_size=OBJ_SIZE) for g in self.grid_init ]
        else:
            pop = [ WSListIndividual(g, obj_size=OBJ_SIZE) for g in self.grid_init ]
        return pop

    def run(self,
            max_ngen=25,
            offspring_size=None,
            continue_cp=False,
            cp_filename=None,
            cp_frequency=0):
        """Run optimisation"""
        # Allow run function to override offspring_size
        # TODO probably in the future this should not be an object field anymore
        # keeping for backward compatibility
        if offspring_size is None:
            offspring_size = self.offspring_size

        pop = self.toolbox.population(n=offspring_size)
        pop = self.set_pop()
        hof = deap.tools.HallOfFame(offspring_size)
        pf = deap.tools.ParetoFront(offspring_size)

        stats = deap.tools.Statistics(key=lambda ind: ind.fitness.sum)
        stats.register("avg", numpy.mean)
        stats.register("std", numpy.std)
        stats.register("min", numpy.min)
        stats.register("max", numpy.max)
        pop, hof, pf, log, history, gen_vs_pop = algorithms.eaAlphaMuPlusLambdaCheckpoint(
            pop,
            self.toolbox,
            offspring_size,
            self.cxpb,
            self.mutpb,
            max_ngen,
            stats=stats,
            halloffame=hof,
            pf=pf,
            nelite=self.elite_size,
            cp_frequency=cp_frequency,
            continue_cp=continue_cp,
            cp_filename=cp_filename,
            selection = self.selection,
            td = self.td)

        # insert the initial HOF value back in.
        td = self.td
        return pop, hof, pf, log, history, td, gen_vs_pop

