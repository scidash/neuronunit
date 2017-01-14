

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
#from inspyred.ec import terminators as term



import ipyparallel as ipp
rc = ipp.Client()
v = rc[:]


class DeapContainer:
    '''
    Just a container for hiding implementation, not a very sophisticated one at that.
    '''
    def __init__(self,*args):
        self.tb = base.Toolbox()
        self.ngen=None
        self.pop_size=None
        self.param=None
        self.last_params=None
        #Warning, the algorithm below is sensitive to certain multiples in the population size
        #which is denoted by MU.
        #The mutiples of 100 work, many numbers will not work
        #TODO write a proper exception handling method.
        #TODO email the DEAP list about this issue too.
        #TODO refactor MU into pop_size
                             #self.ff,pop_size,ngen,NDIM=1,OBJ_SIZE=1,self.range_of_values

    def sciunit_optimize_nsga(self,test_or_suite,model,pop_size,ngen,rov, param,
                              NDIM=1,OBJ_SIZE=6,seed_in=1):

        self.model=model
        self.parameters=param
        self.ngen = ngen#250
        self.pop_size = pop_size#population size
        self.rov = rov # Range of values

        toolbox = base.Toolbox()
        creator.create("FitnessMax", base.Fitness, weights=(-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,))#Final comma here, important, not a typo, must be a tuple type.
        creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMax)


        class DModel():
            #TODO delete
            #def __init__():
            self.name=''
            self.attrs={}
            self.results={}

        class Individual(list):
            '''
            When instanced the object from this class is used as one unit of chromosome or allele by DEAP.
            Extends list via polymorphism.
            '''
            def __init__(self, *args):
                list.__init__(self, *args)
                self.sciunitscore=[]
                self.model=None
                self.error=None
                self.results=None
                self.name=''
                self.attrs={}
                self.params=None
                self.score=None
                self.error=None


                #self.dmodel=DModel()
                #pdb.set_trace()

        def uniform(low, up, size=None):
            '''
            This is the PRNG distribution that defines the initial
            allele population. Inputs are the maximum and minimal numbers that the PRNG can generate.
            '''
            try:
                return [random.uniform(a, b) for a, b in zip(low, up)]
            except TypeError:
                return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]



        name=None
        attrs=None
        name_value=None
        error=None
        score=None
        BOUND_LOW=[ np.min(i) for i in rov ]
        BOUND_UP=[ np.max(i) for i in rov ]
        NDIM = len(rov)

        #BOUND_LOW=np.min(rov)
        #BOUND_UP=np.max(rov)

        #import multiprocessing
        #pool = multiprocessing.Pool()
        from ipyparallel import Client
        rc = Client(profile=os.getenv('IPYTHON_PROFILE'))
        lview = rc.load_balanced_view()

        map_function = lview.map_sync
        toolbox.register("map", map_function)
        #toolbox.register("map", pool.map)
        toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM)
        #assert NDIM==2
        #toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        def callsciunitjudge(individual):#This method must be pickle-able for scoop to work.
            #pdb.set_trace()
            print('hello from before error')
            error=individual.error
            #pdb.set_trace()

            return (error[0],error[1],error[2],error[3],error[4],)
        import dill

        def paramap(the_func,pop):

            import mpi4py
            from mpi4py import MPI
            COMM = MPI.COMM_WORLD
            SIZE = COMM.Get_size()
            RANK = COMM.Get_rank()


            import copy
            pop1=copy.copy(pop)
            #ROund robin distribution
            psteps = [ pop1[i] for i in range(RANK, len(pop1), SIZE) ]
            pop=[]

            #Do the function to list element mapping
            pop=list(map(the_func,pop1))
            #gather all the resulting lists onto rank0
            print('code hangs here why1 ?')

            pop2 = COMM.gather(pop, root=0)
            print('code hangs here why2?')
            #COMM.barrier()
            if RANK == 0:
                print('got to past rank0 block')
                pop=[]
                #merge all of the lists into one list on rank0
                for p in pop2:
                    pop.extend(p)
                print('hangs here 2')

            else:
                pop=[]
                print('hangs here 3')

            if RANK==0:
                #broadcast the results back to all hosts.
                print('stuck 3')
                pop = COMM.bcast(pop, root=0)
                print('hangs here 4')
            #COMM.barrier()
            #if RANK!=0:
            #        COMM.Abort()
            #dir(COMM)
            return pop

        '{} {}'.format('why it cant pickle',dill.pickles(callsciunitjudge))
        '{} {}'.format('why it cant pickle',dill.detect.badtypes(callsciunitjudge))
        #import pdb; pdb.set_trace()
        toolbox.register("evaluate",callsciunitjudge)

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

#        #for i in range(RANK, len(pop), SIZE):#
        #    psteps1.append(pop[i])
        #assert len(psteps)==len(psteps1)
        #pop=[]
        def func2map(ind):
            print('\n')
            print('\n')
            print('\n')

            #print('hello from rank: ',RANK)
            print('\n')
            print('\n')
            print('\n')

            #print(RANK)
            ind.sciunitscore={}
            self.model=self.model.load_model()
            for i, p in enumerate(param):
                name_value=str(ind[i])
                #reformate values.
                self.model.name=name_value
                if i==0:
                    attrs={'//izhikevich2007Cell':{p:name_value }}
                else:
                    attrs['//izhikevich2007Cell'][p]=name_value

            self.model.update_run_params(attrs)
            import copy

            ind.results=copy.copy(self.model.results)
            print(type(ind))
            ind.attrs=attrs
            ind.name=name_value
            ind.params=p
            score = test_or_suite.judge(model)
            ind.error = score.sort_keys.values[0]
            return ind

        #pop=paramap(func2map,pop)
        #pop=v.map(func2map,pop)
        pop=list(map(func2map,pop))

        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        for ind in invalid_ind:
            print(hasattr(ind,'error'))
        print('this is where I should check what attributes ind in pop has')

        #pdb.set_trace()

        #pop1=copy.copy(pop)
        #psteps = [ pop1[i] for i in range(RANK, len(pop), SIZE) ]

        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        #fitnesses = paramap(callsciunitjudge,invalid_ind)
        #fitnesses = v.map(callsciunitjudge,invalid_ind)
        #pdb.set_trace()
        for ind, fit in zip(invalid_ind, fitnesses):
            print(ind,fit)
            ind.fitness.values = fit
            print(ind.fitness.values)


        # This is just to assign the crowding distance to the individuals
        # no actual selection is done
        pop = toolbox.select(pop, len(pop))
        #pop = toolbox.select(pop, len(pop))

        record = stats.compile(pop)
        logbook.record(gen=0, evals=len(invalid_ind), **record)
        print(logbook.stream)

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
                print('ind1 is updating: ')
                print(ind1,ind2)
                toolbox.mutate(ind1)
                toolbox.mutate(ind2)
                del ind1.fitness.values, ind2.fitness.values
                print('stuck y')
                #pdb.set_trace()


            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            print('stuck x')
            #pdb.set_trace()
            #fitnesses = list(map(callsciunitjudge,invalid_ind))
            #fitnesses = paramap(callsciunitjudge,invalid_ind)
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)

            #fitnesses=v.map(callsciunitjudge,invalid_ind)

            #fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)

            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            print('gen is updating: ', gen)
            print(gen)
            print(record)

            #COMM.barrier()


            '''
            pop2 = COMM.gather(pop, root=0)
            if RANK == 0:
                print('got to past rank0 block')
                pop=[]
                for p in pop2:
                    pop.extend(p)
                    # Select the next generation population
                    #was this: pop = toolbox.select(pop + offspring, MU)
                pop = toolbox.select(offspring, self.pop_size)

                print('stuck 1')
            else:
                print('stuck 2')
                pop=None
            if RANK==0:
                print('stuck 3')
                pop = COMM.bcast(pop, root=0)
                print('stuck 4')

            COMM.barrier()
            print('stuck 5')
            '''
        return pop

        #pdb.set_trace()
        #return (pop[0][0],pop[0].sciunitscore)


    def sciunit_optimize(self,test_or_suite,model,pop_size,ngen,rov, param,
                              NDIM=1,OBJ_SIZE=1,seed_in=1):

        self.model=model
        self.parameters=param
        self.ngen = ngen#250
        self.pop_size = pop_size#population size
        self.rov = rov # Range of values

        toolbox = base.Toolbox()
        creator.create("FitnessMax", base.Fitness, weights=(-1.0,-1.0,-1.0,-1.0,-1.0,-1.0))#Final comma here, important, not a typo, must be a tuple type.
        creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMax)

        class Individual(list):
            '''
            When instanced the object from this class is used as one unit of chromosome or allele by DEAP.
            Extends list via polymorphism.
            '''
            def __init__(self, *args):
                list.__init__(self, *args)
                self.sciunitscore=[]


        def uniform(low, up, size=None):
            '''
            This is the PRNG distribution that defines the initial
            allele population. Inputs are the maximum and minimal numbers that the PRNG can generate.
            '''
            try:
                return [random.uniform(a, b) for a, b in zip(low, up)]
            except TypeError:
                return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]



        toolbox.register("map", futures.map)
        #assert NDIM==2

        #for i in rov:
        BOUND_LOW=[ np.min(i) for i in rov ]
        BOUND_UP=[ np.max(i) for i in rov ]
        NDIM = len(rov)
        toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM)


        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        def callsciunitjudge(individual):

            name={}
            attrs={}
            # Individual and units needs to change
            # name_value and attrs need to change depending on the test being taken.
            #
            #self.model.attrs=self.param
            #TODO make it such that
            name_value= str(individual[0])+str('mV')
            name={'V_rest': name_value }
            attrs={'//izhikevich2007Cell':{'vr':name_value }}

            self.model.name=name
            self.model.load_model()
            self.model.attrs=attrs
            self.model.h.psection()
            #self.new_params=str(self.model.psection())
            #self.last_params=str(self.model.psection())
            score = test_or_suite.judge(model)
            individual.sciunit_score=score.sort_keys.mean().mean()
            print(individual[0])
            print(score.sort_keys)
            #print("V_rest = %.1f; SortKey = %.3f" % (float(individual[0]),float(score.sort_key)))
            if type(score) != None:

                if type(score.sort_keys) != None:

                    error = -score.sort_keys.mean().mean()

                    print(score)

            return error,

        toolbox.register("evaluate",callsciunitjudge)
        toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
        toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM)
        toolbox.register("select", tools.selTournament, tournsize=3)
        seed=1
        random.seed(seed)

        CXPB = 0.9#cross over probability
        MUTPB= 0.2
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", numpy.mean, axis=0)
        stats.register("std", numpy.std, axis=0)
        stats.register("min", numpy.min, axis=0)
        stats.register("max", numpy.max, axis=0)

        logbook = tools.Logbook()
        logbook.header = "gen", "evals", "std", "min", "avg", "max"


        pop = toolbox.population(n=self.pop_size)

        print("Start of evolution")

        # Evaluate the entire population
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        print("Evaluated individuals",len(pop))

        # Begin the evolution
        for gen in range(self.ngen):
            g=gen#TODO refactor
            print("-- Generation %i --" % g)

            # Select the next generation individuals
            offspring = toolbox.select(pop, len(pop))
            # Clone the selected individuals
            offspring = list(map(toolbox.clone, offspring))

            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):

                # cross two individuals with probability CXPB
                if random.random() < CXPB:
                    toolbox.mate(child1, child2)

                    # fitness values of the children
                    # must be recalculated later
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:

                # mutate an individual with probability MUTPB
                if random.random() < MUTPB:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            print("  Evaluated %i individuals" % len(invalid_ind))

            # The population is entirely replaced by the offspring
            pop[:] = offspring
            #error_surface(pop,gen,ff=self.ff)
            # Gather all the fitnesses in one list and print the stats
            fits = [ind.fitness.values[0] for ind in pop]
            #TODO terminate DEAP learning when the population converges to save computation
            #this will probably involve using term as defined by the import statement above.
            #To be specific using term attributes in a conditional that evaluates to break if true.

        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        print(logbook.stream)
        #error_surface(pop,gen,ff=self.ff)
        return pop#(pop[0][0],pop[0].sus0,ff)


        #Depreciated
        def error_surface(pop,gen):
            '''
            Plot the population on the error surface at generation number gen.
            solve a trivial parabola by brute force
            plot the function to verify the maxima
            Inputs are DEAP GA population of chromosomes and generation number
            no outputs.

            plot the GAs genes parameter values, against the error for each genes parameter.
            '''
            import matplotlib
            matplotlib.use('agg')
            import matplotlib.pyplot as plt
            plt.hold(True)

            scatter_pop=np.array([ind[0] for ind in pop])
            #note the error score is inverted bellow such that it aligns with the error surface.
            scatter_score=np.array([-ind.sus0 for ind in pop])
            #pdb.set_trace()
            plt.scatter(scatter_pop,scatter_score)
            plt.hold(False)
            plt.savefig('simple_function'+str(gen)+'.png')
