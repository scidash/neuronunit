"""Optimisation class This file is part of BluePyOpt <https://github.com/BlueBrain/BluePyOpt>"""
# https://deap.readthedocs.io/en/master/tutorials/basic/part3.html
from neuronunit.optimisation import exhaustive_search as es
from neuronunit.optimisation import optimization_management as om

import random
import functools
import numpy

import deap
import deap.base
import deap.algorithms
import deap.tools as tools
import array
from deap import base

toolbox = base.Toolbox()

from . import algorithms
from . import alg
import numpy
from numba import jit
import dask

import logging
logger = logging.getLogger('__main__')
from neuronunit.optimisation.data_transport_container import DataTC

# TODO decide which variables go in constructor, which ones go in 'run' function
# TODO abstract the algorithm by creating a class for every algorithm, that way
# settings of the algorithm can be stored in objects of these classes

class BetterFitness(base.Fitness):
    def __le__(self, other):
        return sum(self.wvalues) <= sum(other.wvalues)

    def __lt__(self, other):
        return sum(self.wvalues) < sum(other.wvalues)


import numpy as np
from collections import Iterable, OrderedDict

import copy
from deap import base
class SciUnitOptimisation(object):
    from bluepyopt.optimisations import DEAPOptimisation

    """DEAP Optimisation class"""
    def __init__(self, tests = None, 
                 benchmark = False,
                 seed=None,
                 MU=15,
                 elite_size=3,
                 eta=20,
                 mutpb=0.7,
                 cxpb=0.9,
                 backend = None,
                 nparams = 10,
                 boundary_dict= {},
                 hc = None,
                 verbose = 0,
                 seed_pop = None,
                 protocol={'allen':False,'elephant':True},
                 simulated_obs=None):
        self.seed_pop = seed_pop
        self.verbose = verbose
        """Constructor"""
        #self.simulated_obs = simulated_obs
        self.protocol = protocol
        #super(SciUnitOptimisation, self).__init__()
        self.benchmark = benchmark

        self.tests = tests
        self.OBJ_SIZE = 1#len(self.tests)#+1
        self.seed = seed
        self.MU = MU
        try:
            self.DO_other = DEAPOptimisation(offspring_size=self.MU, \
                selector_name='IBEA')
        except:
            pass
        self.elite_size = elite_size
        self.eta = eta
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.backend = backend
        # Create a DEAP toolbox
        self.toolbox = deap.base.Toolbox()
        self.hc = hc
        self.boundary_dict = boundary_dict
        #self.OBJ_SIZE = None
        self.setnparams(nparams = nparams, boundary_dict = boundary_dict)
        self.setup_deap()

    def dask_map(self, f, x):
        x = db.from_sequence(x, npartitions=self.npartitions) 
        return db.map(f, x).compute()


    def transdict(self,dictionaries):
        mps = OrderedDict()
        sk = sorted(list(dictionaries.keys()))
        for k in sk:
            mps[k] = dictionaries[k]
        tl = [ k for k in mps.keys() ]
        return mps, tl

    def setnparams(self, nparams = 10, boundary_dict = None):
        self.params = om.create_subset(nparams = nparams,boundary_dict = boundary_dict)
        self.params, self.td = self.transdict(boundary_dict)
        return self.params, self.td

    def set_evaluate(self):
        self.toolbox.register("evaluate", om.evaluate)

    def grid_sample_init(self, nparams):
        '''
        the number of points, should be 2**n, where n is the number of dimensions
        but 2**n can be such a big number it's not even computationally tractible.
        Therefore if 2**n is greater than offspring size, sparsify the grid initialisation
        and only use a sparse sampling of points.
        1 -self.offsping size/len(dic_grid).
        '''

        npoints = 2 ** len(list(self.params))
        npoints = np.ceil(npoints)
        if len(self.params)>1:
            dic_grid = es.create_grid(mp_in = self.params,npoints = self.MU, free_parameters = self.params)
        else:
            npoints = self.MU
            values = np.linspace(np.min(list(self.params.values())[0]),np.max(list(self.params.values())[0]),npoints)
            single_key = list(self.params.keys())[0]
            dic_grid = [{single_key:v} for v in values ]
            #import pdb; pdb.set_trace()
        dic_grid = list(dic_grid)
        size = len(dic_grid)

        '''

        This code causes memory errors for some population sizes
        The grid is now defined, the rest of code just makes sure that the size of the grid is a reasonable size
        And computationally tractable. When I write sparse, think 'Down sample' a big, overly sampled list of coordinates.
        '''
        if size > self.MU:
            sparsify = np.linspace(0,len(dic_grid)-1,self.MU)
            pop = []
            for i in sparsify:
                d = dic_grid[int(i)]
                pop.append([float(d[k]) for k in self.td])


        elif size <= self.MU:
            delta = self.MU - size
            pop = []
            dic_grid = es.create_grid(mp_in = self.params,npoints = self.MU+delta, free_parameters = self.params)
            size = len(dic_grid)
            delta = self.MU - size

            for i in dic_grid:
                 pop.append([float(i[k]) for k in self.td])


            cnt=0
            while delta:# and cnt<2:
                dic_grid = list(copy.copy(dic_grid))

                pop.append(copy.copy(pop[0]))
                size = len(pop)
                delta = self.MU - size
                cnt+=1

        elif size == self.MU:
            pop = []
            for i in dic_grid:
                pop.append([float(i[k]) for k in self.td])

        assert len(pop)==self.MU
        for p in pop:
            for i,j in enumerate(p):
                p[i] = float(j)
        return pop



    def setup_deap(self):
        """Set up optimisation"""
        # Set random seed
        from datetime import datetime
        if self.seed is None:
            random.seed(datetime.now())
        else:
            random.seed(self.seed)

        # Eta parameter of crossover / mutation parameters
        # Basically defines how much they 'spread' solution around
        # The lower this value, the more spread
        ETA = self.eta

        # Number of parameters
        # Bounds for the parameters
        IND_SIZE = len(list(self.params.values()))

        #OBJ_SIZE = len(self.error_criterion)
        #self.OBJ_SIZE = OBJ_SIZE
        def glif_modifications(UPPER,LOWER):
            for index, i in enumerate(UPPER):
                if i == LOWER[index]:
                    LOWER[index]-=2.0
                    i+=2.0
            return LOWER

        if self.backend == 'GLIF':
            del self.td[-1]
            self.params.pop('type',None)

            self.td = [ param for param in self.td if type(self.params[param][0]) is type(float(0.0)) ]
            self.params = { param:self.params[param] for param in self.td if type(self.params[param][0]) is type(float(0.0)) }


            LOWER = [ np.min(self.params[v]) for v in self.td ]
            UPPER = [ np.max(self.params[v]) for v in self.td ]
            LOWER = glif_modifications(UPPER,LOWER)
        else:
            LOWER = [ np.min(self.params[v]) for v in self.td ]
            UPPER = [ np.max(self.params[v]) for v in self.td ]
        # in other words the population
        if type(self.seed_pop) is type([]):
            self.grid_init = self.seed_pop
            ordered = OrderedDict(self.params)
            self.td = list(ordered.keys())

        elif type(self.seed_pop) is type({}):
            '''
            If there is only one point in parameter space, as oppossed to a collection of points:
            '''
            ordered = OrderedDict(self.params)
            ind = []

            self.grid_init = self.grid_sample_init(self.params)
            for k,v in ordered.items():
                ind.append(self.seed_pop[k])
            self.grid_init.append(ind)
            self.td = list(ordered.keys())


    
        elif type(self.seed_pop) is None:
            #.grid_init = self.grid_sample_init(self.params)#(LOWER, UPPER, self.MU)
            from deap import creator

            creator.create("FitnessMin", base.Fitness, weights=tuple(-1.0 for i in range(0,self.OBJ_SIZE)))
            creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)
            self.toolbox.register("population", tools.initRepeat, list, creator.Individual)

            #self.grid_init = self.set_pop(boot_new_random=self.MU)
        if not hasattr(self,'grid_init'):
            #self.grid_init = self.grid_sample_init(self.params)#(LOWER, UPPER, self.MU)
            from deap import creator
            creator.create("FitnessMin", base.Fitness, weights=tuple(-1.0 for i in range(0,self.OBJ_SIZE)))
            creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)
            self.toolbox.register("population", tools.initRepeat, list, creator.Individual)

           # self.grid_init = self.set_pop(boot_new_random=self.MU)


        def uniform_params(lower_list, upper_list, dimensions):
            if hasattr(lower_list, '__iter__'):
                other = [random.uniform(lower, upper) for lower, upper in zip(lower_list, upper_list)]
            else:
                other = [random.uniform(lower_list, upper_list)
                    for _ in range(dimensions)]
            return other

        from deap import creator

        #creator.create("FitnessMin", base.Fitness, weights=tuple(-1.0 for i in range(0,self.OBJ_SIZE)))
        creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)
        NDIM = len(LOWER)
        self.toolbox.register("attr_float", uniform_params, LOWER, UPPER, NDIM)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.attr_float)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        self.toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=LOWER, up=UPPER, eta=20.0)
        self.toolbox.register("mutate", tools.mutPolynomialBounded, low=LOWER, up=UPPER, eta=20.0, indpb=1.0/NDIM)
        self.toolbox.register("select", tools.selNSGA2)

        self.OM = om.OptMan(self.tests,self.td, backend = self.backend, \
                                              hc = self.hc,boundary_dict = self.boundary_dict, \
                                              error_length=self.OBJ_SIZE,protocol=self.protocol)

        def custom_code(invalid_ind):

            if self.backend is None:
                self.backend = 'IZHI'
                print('should not happen')
            if self.verbose:
                print(self.tests.use_rheobase_score)
            invalid_pop = list(self.OM.update_deap_pop(invalid_ind, self.tests, \
                                                  td = self.td, backend = self.backend, \
                                                  hc = self.hc,boundary_dict = self.boundary_dict, \
                                                  error_length=self.OBJ_SIZE))


            invalid_dtc = [ i.dtc for i in invalid_pop if hasattr(i,'dtc') ]
            assert len(invalid_pop) == len(invalid_dtc)
            fitnesses = []
            for i in invalid_dtc:
                fitnesses.append(om.evaluate(i))
            """
            Add this bit:
            for i in invalid_dtc:
                fitnesses.append(om.evaluate_new(i))
            """
            
            return (invalid_pop,fitnesses)

        self.toolbox.register("evaluate", custom_code)
        # Register the mate operator
        self.toolbox.register(
            "mate",
            deap.tools.cxSimulatedBinaryBounded,
            eta=self.eta,
            low=LOWER,
            up=UPPER)


        # Register the mutation operator
        self.toolbox.register("variate", deap.algorithms.varAnd)
        IND_SIZE = len(list(self.params.values()))
        self.toolbox.register(
            "mutate",
            deap.tools.mutPolynomialBounded,
            eta=self.eta,
            low=LOWER,
            up=UPPER,
            indpb=1.0/IND_SIZE)

        return self.OM



    def set_pop(self, boot_new_random=0):
        IND_SIZE = len(list(self.params.values()))
        OBJ_SIZE = self.OBJ_SIZE
        if boot_new_random == 0:
            #pop = []
            
            pop = self.toolbox.population(n=len(self.grid_init))
            for i,g in enumerate(self.grid_init):   
                for j,param in enumerate(pop[i]): 
                    pop[i][j] = g[j]
        else:
            pop = self.toolbox.population(n=boot_new_random)
        return pop

    def run(self,
            continue_cp=False,
            cp_filename=None,
            cp_frequency=2,
            NGEN = 10):
        """Run optimisation"""
        # Allow run function to override MU
        # TODO probably in the future this should not be an object field anymore
        # keeping for backward compatibility
        #if MU is None:
        MU = self.MU

        #pop = self.set_pop()
        pop = self.set_pop(boot_new_random=MU)
        hof = deap.tools.HallOfFame(MU)
        pf = deap.tools.ParetoFront()
        #pf = deap.tools.ParetoFront(MU) # Wrong because first arg to ParetoFront is similarity metric not pop size

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        pop, hof, pf, log, history, min_gene = alg.eaAlphaMuPlusLambdaCheckpoint(
            pop,
            self.toolbox,
            MU,
            self.cxpb,
            self.mutpb,
            NGEN,
            stats=stats,
            hof=hof,
            pf=pf,
            td = self.td)

        # insert the initial HOF value back in.
        td = self.td
        self.ga_out = {'pop':pop,'hof':hof,'pf':pf,'log':log,'history':history,'td':td,'min_gene':min_gene}
        return self.ga_out

def run_ga(explore_edges, NGEN, test, \
        free_parameters = None, hc = None,
        MU = None, seed_pop = None, \
           backend = str('IZHI'),protocol={'allen':False,'elephant':True}):
    ss = {}
    try:
        free_parameters.pop('dt')
    except:
        pass
    #if 'Iext' in explore_edges:
    try:
        explore_edges.pop('Iext')
    except:
        pass
    for k in free_parameters:
        if k not in "Iext":
            if not k in explore_edges.keys() and k not in str('Iext') and k not in str('dt'):
                ss[k] = explore_edges[str(free_parameters)]
            else:
                ss[k] = explore_edges[k]
    if type(MU) == type(None):
        MU = 2**len(list(free_parameters))
    NGEN = int(np.floor(NGEN))
    if not isinstance(test, Iterable):
        test = [test]
    DO = SciUnitOptimisation(MU = MU, tests = test,\
     boundary_dict = ss, backend = backend, hc = hc, \
                             protocol=protocol)

    if seed_pop is not None:
        # This is a re-run condition.
        DO.setnparams(nparams = len(free_parameters), boundary_dict = ss)

        DO.seed_pop = seed_pop
        DO.setup_deap()
        DO.error_length = len(test)
    ga_out = DO.run(NGEN = NGEN)

    return ga_out, DO
