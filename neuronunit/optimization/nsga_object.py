#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.
class NSGA(object):
    def __init__(self, CXPB, nparams=10):
        self.nparams = nparams
        self.subset = None

        from deap import base
        toolbox = base.Toolbox()
        from deap import tools
        #toolbox.register("select", tools.selNSGA2)
        #from bluepyopt.deapext import tools
        from bluepyopt.deapext import tools
        toolbox.register("select", tools.selIBEA)
        #toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox = toolbox
        self.history = None #to be initiated elsewhere
        self.pf = None
        self.CXPB = CXPB



    def set_map(self):
        import ipyparallel as ipp
        self.ipp = ipp
        rc = ipp.Client(profile='default')
        dview = rc[:]
        rc[:].use_cloudpickle()
        #rc.Client.become_dask()
        self.dview = rc[:]
        self.toolbox.register("map", dview.map_sync)
        return dview


    def set_evaluate(self):
        from neuronunit.optimization import nsga_parallel
        self.toolbox.register("evaluate", nsga_parallel.evaluate)
        #return toolbox
    def evolve(self,pop,MU,gen):
        import array
        import random
        import numpy
        from deap import algorithms
        from deap import base
        from deap import tools
        from neuronunit.optimization import nsga_parallel

        # Vary the population
        toolbox = self.toolbox #= toolbox
        creator = self.creator# = creator
        tools = self.tools# = tools
        logbook = self.logbook
        #offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = toolbox.select(pop, len(pop))
        #offspring = toolbox.select(pop, len(pop))

        offspring = [toolbox.clone(ind) for ind in offspring]


        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= self.CXPB:
                toolbox.mate(ind1, ind2)

            toolbox.mutate(ind1)
            toolbox.mutate(ind2)
            del ind1.fitness.values, ind2.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        # To understand the code below you need to understand that invalid_ind is
        # a perservering reference to a part of offspring.
        import copy
        #invalid_dtc = list(nsga_parallel.update_pop(invalid_ind,self.td))
        return_package = list(nsga_parallel.update_pop(invalid_ind,self.td))
        invalid_dtc = []
        for i,r in enumerate(return_package):
            invalid_dtc.append(r[0])# = return_package[0][:]
            invalid_ind[i] = r[1]


        #assert set(list(invalid_ind[:])) in set(list(offspring[:]))
        '''
        for i,o in enumerate(offspring):
            set_check_off.update(list(o))
            if i < len(invalid_ind):
                set_check_inv.update(list(invalid_ind[i]))
        if len(set_check_inv.intersection(set_check_off)) > 0:
            print(set_check_inv.intersection(set_check_off))
            print(len(set_check_inv.intersection(set_check_off)), ' < size of intersection:')
            print('size of non intersection')
            print(abs(len(offspring) - len(set_check_inv.intersection(set_check_off))))
        '''



        from neuronunit.optimization.nsga_parallel import evaluate
        fitnesses = list(map(evaluate,invalid_dtc))
        #for i,d in enumerate(invalid_dtc):
        #    invalid_ind[i] =
        for ind, dtc in zip(invalid_ind, invalid_dtc):
            ind.scores = None
            ind.scores = dtc.scores
        #fitnesses = list(toolbox.map(toolbox.evaluate, copy.copy(invalid_dtc)))

        for ind, fit in zip(offspring, fitnesses):
            ind.fitness.values = fit

        # Select the next generation population
        #pop = toolbox.select(pop + offspring, MU)
        # select among best ever individuals, offspring, and the initial pop
        #if len(self.pf)>0:
        #    pop = toolbox.select(list(self.pf) + offspring + pop, MU)
        #else:
        #import pdb; pdb.set_trace()
        '''
        if hasattr(self.pf[0],'fitness') and hasattr(self.pf[1],'fitness'):
            if float(sum(self.pf[0].fitness.values)) > 0 and float(sum(self.pf[1].fitness.values)) > 0:
                select_out_of = self.pf[0]+ self.pf[1] + offspring[0:-3]+ pop[0]
                print('gets here b')
            else:
                select_out_of = offspring[0:-3] + pop[0:2]
        else:
        select_out_of = offspring[0:-3] + pop[0:2]
        '''

        pop = toolbox.select(invalid_ind + pop, len(pop))
        #pop = toolbox.select(invalid_ind + pop , MU)

        self.history.update(pop)
        self.pf.update(pop)
        #if gen > 1:
        #    pop[0] = self.pf[0]

        #pop = toolbox.select(offspring + pop , MU)

        record = self.stats.compile(pop)
        self.logbook.record(gen=gen, evals=len(invalid_ind), **record)

        invalid_dtc = list(copy.copy(invalid_dtc))
        self.fitnesses = list(fitnesses)
        return invalid_dtc, pop, self.logbook, self.fitnesses, self.pf

    def setnparams(self,nparams=10, provided_keys=None):
        from neuronunit.optimization import nsga_parallel
        self.nparams = nparams
        self.subset = nsga_parallel.create_subset(nparams=self.nparams,provided_keys=provided_keys)


    def main(self, MU, NGEN, seed=None):

        self.set_map()
        #dview = self.set_evaluate()
        from neuronunit.optimization import evaluate_as_module
        from neuronunit.optimization.evaluate_as_module import import_list
        from neuronunit.optimization import nsga_parallel
        from deap.benchmarks.tools import diversity, convergence

        import array
        import random
        import numpy
        from deap import algorithms
        from deap import base

        import numpy
        from numpy import random
        from neuronunit.optimization.nsga_parallel import evaluate
        Individual = evaluate_as_module.Individual
        self.dview.push({'Individual':Individual})

        numb_err_f = 8 # eight error functions
        NDIM = 10 # ten parameters
        #ind = self.ipp.Reference('Individual')
        subset = self.subset

        toolbox, tools, self.history, creator, base, self.pf = evaluate_as_module.import_list(self.ipp, subset,numb_err_f,NDIM)
        self.toolbox = toolbox
        self.creator = creator
        self.tools = tools
        Individual = evaluate_as_module.Individual
        self.dview.push({'Individual':Individual})
        #print(import_list,subset,numb_err_f)
        self.dview.apply_sync(import_list,self.ipp,subset,numb_err_f,NDIM)
        get_trans_dict = evaluate_as_module.get_trans_dict
        self.td = get_trans_dict(self.subset)
        self.dview.push({'td':self.td })

        pop = toolbox.population(n = MU)
        pop = [ toolbox.clone(i) for i in pop ]
        self.history.update(pop)
        self.pf.update(pop)

        self.dview.scatter('Individual',pop)

        #from neuronunit.optimization import nsga_parallel

        toolbox = self.toolbox
        dview = self.set_map()
        self.set_evaluate()

        random.seed(seed)


        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", numpy.mean, axis=0)
        stats.register("std", numpy.std, axis=0)
        stats.register("min", numpy.min, axis=0)
        stats.register("max", numpy.max, axis=0)
        self.stats = stats
        logbook = tools.Logbook()
        logbook.header = "gen", "evals", "std", "min", "avg", "max"

        pop = toolbox.population(n=MU)
        self.pf.update(pop)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        #invalid_dtc
        return_package = list(nsga_parallel.update_pop(invalid_ind,self.td))
        invalid_dtc = []
        for i,r in enumerate(return_package):
            invalid_dtc.append(r[0])# = return_package[0][:]
            invalid_ind[i] = r[1]

        fitnesses = list(map(evaluate,invalid_dtc))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        for ind, dtc in zip(invalid_ind, invalid_dtc):
            ind.scores = None
            ind.scores = dtc.scores
        # This is just to assign the crowding distance to the individuals
        # no actual selection is done
        pop = toolbox.select(pop, len(pop))

        record = stats.compile(pop)
        logbook.record(gen=0, evals=len(invalid_ind), **record)
        self.logbook = logbook
        #print(self.logbook.stream)

        old_minimum = 2.0
        old_std = 0.0
        for gen in range(1, NGEN):
            invalid_dtc, pop, logbook, fitnesses, self.pf = self.evolve(pop,MU,gen)
            print(gen)
            if 'dtcpopg' not in dir():
                import numpy as np

                import pickle
                #grid_dump_second_3rd.p
                first_third = pickle.load(open('grid_dump_first_3rd.p','rb'))
                second_third = pickle.load(open('grid_dump_second_3rd.p','rb'))
                final_third = pickle.load(open('grid_dump_final_3rd.p','rb'))

                second_third.extend(first_third)
                second_third.extend(final_third)
                dtcpopg = second_third
                print(dtcpopg)

            def error(dtc):
                #import numpy as np

                """
                Overall error function for a DTC
                Returns the root-mean-square error over all the tests
                """
                #return sum(dtc.scores.values())
                return np.sqrt(np.mean(np.square(list(dtc.scores.values()))))

            def sorted_dtcs(dtcpop):
                """
                Returns dtc,error tuples sorted from low to high error
                """
                return sorted([(dtc,error(dtc)) for dtc in dtcpop],key=lambda x:x[1])


            dtcpopg = [ d for d in dtcpopg if not d.scores['RheobaseTestP'] == None ]
            minimagr_dtc, maxi = sorted_dtcs(dtcpopg)[-3]
            minimagr_dtc, mini = sorted_dtcs(dtcpopg)[0]
            quantize_distance = list(np.linspace(mini,maxi,10))

            print(invalid_dtc[0].scores)
            print(minimagr_dtc.scores)
            most_accurate_dtc = sum(list(invalid_dtc[0].scores.values()))
            most_accurate_grid = sum(list(minimagr_dtc.scores.values()))
            print('most accurate score indicator looking good:',bool(most_accurate_dtc<most_accurate_grid ))

            current_mean = 0
            errors = [ (i, error(i)) for i in invalid_dtc if hasattr(i,'scores') ]
            cnte = 0
            cntbetter = 0
            if old_minimum > sum(logbook.select('min')[-1]):
                old_minimum = sum(logbook.select('min')[-1])
                print('decreasing best')

            if old_std < sum(logbook.select('std')[-1]):
                old_std = sum(logbook.select('std')[-1])
                print('increasing stdev')
            print(old_std, 'new standard dev')

            print(old_minimum, 'new minimum')
            oldf = 1.2
            ii = 0
            for i,f in enumerate(fitnesses):
                if oldf > sum(f):
                    oldf = sum(f)
                    ii = i
            print(invalid_dtc[ii].scores, ' via maximum fitness')


            for i,e in errors:
                if e < mini:
                    cntbetter+=1
                    print(i.scores)
                if e < quantize_distance[1]:
                    cnte+=1

            print(cntbetter,'much better than brute force')
            print(cnte,'number of individual aproaching as good as brute force')
            #if
            '''
            pfe = np.sqrt(np.mean(np.square(list((self.pf[0].fitness.values())))))
            pfl = np.sqrt(np.mean(np.square(list((self.pf[-1].fitness.values())))))

            print(self.pf[0],self.pf[-1],pfe,pfl)
            if pfl < mini:
                print('at least the pareto front value is respectable', pfe, mini)
            if pfe < mini:
                print('at least the pareto front value is respectable', pfe, mini)
            '''
            current_mean = np.mean([i for _,i in errors])
            print(current_mean , mini)
            print(current_mean  < mini)


        #removed spurious logbook entry.
        #self.logbook.record(gen=gen, evals=len(invalid_ind), **record)
        import copy
        self.invalid_dtc = list(copy.copy(invalid_dtc))
        self.fitnesses = fitnesses
        return self.invalid_dtc, pop, self.logbook, self.fitnesses, self.history, self.pf
