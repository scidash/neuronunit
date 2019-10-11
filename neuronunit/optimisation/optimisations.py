"""Optimisation class This file is part of BluePyOpt <https://github.com/BlueBrain/BluePyOpt>

"""

from neuronunit.optimisation import optimization_management

import random
import functools
import numpy

import deap
import deap.base
import deap.algorithms
import deap.tools

from . import algorithms
#from . import tools

#import bluepyopt.optimisations
import numpy
from numba import jit
#import bluepyopt.optimisations

import logging
logger = logging.getLogger('__main__')

# TODO decide which variables go in constructor, which ones go in 'run' function
# TODO abstract the algorithm by creating a class for every algorithm, that way
# settings of the algorithm can be stored in objects of these classes

#from neuronunit.optimisation.optimization_management import evaluate#, update_deap_pop

import numpy as np
from collections import OrderedDict

from neuronunit.optimisation import exhaustive_search as es
from neuronunit.optimisation import optimization_management as om

#import neuronunit.optimisation.optimization_management as om

import copy
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


class SciUnitOptimisation(object):#bluepyopt.optimisations.Optimisation):

    """DEAP Optimisation class"""
    def __init__(self, error_criterion = None, evaluator = None,
                 selection = 'selIBEA',
                 benchmark = False,
                 seed=None,
                 offspring_size=15,
                 elite_size=3,
                 eta=20,
                 mutpb=0.7,
                 cxpb=0.9,
                 backend = None,
                 nparams = 10,
                 boundary_dict= {},
                 hc = None,
                 seed_pop = None, protocol={'allen':False,'elephant':True},simulated_obs=None):
        self.seed_pop = seed_pop
        """Constructor"""
        #self.simulated_obs = simulated_obs
        self.protocol = protocol
        #super(SciUnitOptimisation, self).__init__()
        self.selection = selection
        self.benchmark = benchmark

        self.error_criterion = error_criterion
        self.error_length = len(error_criterion)
        #import pdb; pdb.set_trace()
        self.seed = seed
        self.offspring_size = offspring_size
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

    def transdict(self,dictionaries):
        mps = OrderedDict()
        sk = sorted(list(dictionaries.keys()))
        for k in sk:
            mps[k] = dictionaries[k]
        tl = [ k for k in mps.keys() ]
        return mps, tl

    def setnparams(self, nparams = 10, boundary_dict = None):
        self.params = optimization_management.create_subset(nparams = nparams,boundary_dict = boundary_dict)
        self.params, self.td = self.transdict(boundary_dict)
        return self.params, self.td


    def set_evaluate(self):
        if self.benchmark == True:
            self.toolbox.register("evaluate", benchmarks.zdt1)
        else:
            self.toolbox.register("evaluate", optimization_management.evaluate)

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
            dic_grid = es.create_grid(mp_in = self.params,npoints = self.offspring_size, free_params = self.params)
        else:
            npoints = self.offspring_size
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
        if size > self.offspring_size:
            sparsify = np.linspace(0,len(dic_grid)-1,self.offspring_size)
            pop = []
            for i in sparsify:
                d = dic_grid[int(i)]
                pop.append([d[k] for k in self.td])


        elif size <= self.offspring_size:
            delta = self.offspring_size - size
            pop = []
            dic_grid = es.create_grid(mp_in = self.params,npoints = self.offspring_size+delta, free_params = self.params)
            size = len(dic_grid)
            delta = self.offspring_size - size

            for i in dic_grid:
                 pop.append([i[k] for k in self.td])


            cnt=0
            while delta:# and cnt<2:
                dic_grid = list(copy.copy(dic_grid))

                pop.append(copy.copy(pop[0]))
                size = len(pop)
                delta = self.offspring_size - size
                cnt+=1

        elif size == self.offspring_size:
            pop = []
            for i in dic_grid:
                pop.append([i[k] for k in self.td])

        assert len(pop)==self.offspring_size
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

        OBJ_SIZE = len(self.error_criterion)
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
            self.grid_init = self.grid_sample_init(self.params)#(LOWER, UPPER, self.offspring_size)
        if not hasattr(self,'grid_init'):
             self.grid_init = self.grid_sample_init(self.params)

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
        #self.error_criterion['protocol'] = self.protocol # ={'allen':False,'elephant':True,'dm':False}
        OM = om.OptMan(self.error_criterion,self.td, backend = self.backend, \
                                              hc = self.hc,boundary_dict = self.boundary_dict, \
                                              error_length=self.error_length,protocol=self.protocol)
        #OM.siulated_obs = self.simulated_obs
        #assert type(OM.simulated_obs) is not type(None)
        def custom_code(invalid_ind):

            if self.backend is None:
                self.backend = 'RAW'

            invalid_pop = list(OM.update_deap_pop(invalid_ind, self.error_criterion, \
                                                  td = self.td, backend = self.backend, \
                                                  hc = self.hc,boundary_dict = self.boundary_dict, \
                                                  error_length=self.error_length))


            invalid_dtc = [ i.dtc for i in invalid_pop if hasattr(i,'dtc') ]

            fitnesses = list(map(om.evaluate, invalid_dtc))
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

        return OM
    
    def set_pop(self, boot_new_random=0):
        if boot_new_random == 0:
            IND_SIZE = len(list(self.params.values()))
            OBJ_SIZE = len(self.error_criterion)
            if IND_SIZE == 1:
                pop = [ WSListIndividual(g,obj_size=OBJ_SIZE) for g in self.grid_init ]
            else:
                pop = [ WSListIndividual(g, obj_size=OBJ_SIZE) for g in self.grid_init ]
        else:
            pop = self.toolbox.population(n=boot_new_random)
        return pop


    def run(self,
            continue_cp=False,
            cp_filename=None,
            cp_frequency=2,
            max_ngen = 10):
        """Run optimisation"""
        # Allow run function to override offspring_size
        # TODO probably in the future this should not be an object field anymore
        # keeping for backward compatibility
        #if offspring_size is None:
        offspring_size = self.offspring_size

        #pop = self.toolbox.population(n=offspring_size)
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
            hof=hof,
            pf=pf,
            nelite=self.elite_size,
            cp_frequency=cp_frequency,
            continue_cp=continue_cp,
            cp_filename=cp_filename,
            selection = self.selection,
            td = self.td)

        # insert the initial HOF value back in.
        td = self.td
        self.results = {'pop':pop,'hof':hof,'pf':pf,'log':log,'history':history,'td':td,'gen_vs_pop':gen_vs_pop}
        return self.results
from collections import Iterable, OrderedDict

def run_ga(explore_edges, max_ngen, test, \
        free_params = None, hc = None,
        selection = None, MU = None, seed_pop = None, \
        backend = str('RAW'),protocol={'allen':False,'elephant':True}):
    ss = {}
    try:
        free_params.pop('dt')
    except:
        pass
    if 'Iext' in free_params:
        free_params.pop('Iext')
    for k in free_params:
        if not k in explore_edges.keys() and k not in str('Iext') and k not in str('dt'):
            ss[k] = explore_edges[str(free_params)]
        else:
            ss[k] = explore_edges[k]
    if type(MU) == type(None):
        MU = 2**len(list(free_params))
    # make sure that the gene population size is divisible by 4.
    #if selection == True:
    #    selection = str('selNSGA')
    #else:
    #   selection = str('selIBEA')
    max_ngen = int(np.floor(max_ngen))
    if not isinstance(test, Iterable):
        test = [test]
    DO = SciUnitOptimisation(offspring_size = MU, error_criterion = test,\
     boundary_dict = ss, backend = backend, hc = hc, \
     selection = selection,protocol=protocol)

    if seed_pop is not None:
        # This is a re-run condition.
        DO.setnparams(nparams = len(free_params), boundary_dict = ss)

        DO.seed_pop = seed_pop
        DO.setup_deap()
        DO.error_length = len(test)
    ga_out = DO.run(max_ngen = max_ngen)
    return ga_out, DO

'''
temp = [ v.dtc for k,v in history.genealogy_history.items() ]
temp = [ i for i in temp if type(i) is not type(None)]
temp = [ i for i in temp if len(list(i.attrs.values())) != 0.0 ]
true_history = [ (dtc, dtc.get_ss()) for dtc in temp ]
true_mins = sorted(true_history, key=lambda h: h[1])

#sorted(student_tuples, key=lambda student: student[2])
hof = [ h for h in hof if len(h)==len(pop[0])]
hof = [ h for h in hof if type(h.dtc) is not type(None)]
pf = [ p for p in pf if len(p)==len(pop[0])]
pf = [ p for p in pf if type(p.dtc) is not type(None)]
try:
    assert true_mins[0][1] <  hof[0].dtc.get_ss()
    if true_mins[0][1] <  hof[0].dtc.get_ss():
        #print('hall of fame unreliable, compared to history')
        hof = [i[0] for i in true_mins]
        best = hof[0]
        best_attrs = best.attrs
except:
    pass
'''
'''
try:
    attr_keys = list(hof[0].dtc.attrs.keys())


    us = {} # GA utilized_space
    for key in attr_keys:
        #temp = [ v.dtc for k,v in history.genealogy_history.items() ]
        temp = [ i.attrs[key] for i in temp if type(i) is not type(None)]
        #temp = [ v.dtc.attrs[key] for k,v in history.genealogy_history.items() ]
        us[key] = ( np.min(temp), np.max(temp))
        self.us = us
except:
    attr_keys = list(pf[0].dtc.attrs.keys())
    #pass

try:
    self.results['dhof'] = [ h.dtc for h in self.results['hof'] ]
    self.results['bd'] = self.results['hof'][0].dtc
except:
    try:
        self.results['bd'] = self.results['hof'][0].dtc
        self.results['dpf'] = [ h.dtc for h in self.results['pf'] ]
    except:
        self.results['dhof'] = [ p.dtc for p in pop ]
        self.results['bd'] = pop[0].dtc
'''
