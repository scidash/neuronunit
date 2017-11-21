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
    def __init__(self,nparams=10):
        self.nparams = nparams

        from deap import base
        toolbox = base.Toolbox()
        from deap import tools
        toolbox.register("select", tools.selNSGA2)
        self.toolbox = toolbox



    def set_map(self):
        import ipyparallel as ipp
        self.ipp = ipp
        rc = ipp.Client(profile='default')
        rc[:].use_cloudpickle()
        dview = rc[:]
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
        #tournament select is buggy and population size dependent.
        #offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = toolbox.select(pop, len(pop))

        offspring = [toolbox.clone(ind) for ind in offspring]
        if len(offspring)==0:
            import pdb; pdb.set_trace()
        CXPB = 0.9

        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= CXPB:
                toolbox.mate(ind1, ind2)

            toolbox.mutate(ind1)
            toolbox.mutate(ind2)
            del ind1.fitness.values, ind2.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        if len(invalid_ind)==0:
            import pdb; pdb.set_trace()


        #dtcpop = nsga_parallel.update_pop(pop,td)
        import copy
        invalid_dtc = list(nsga_parallel.update_pop(invalid_ind,self.td))
        from neuronunit.optimization.nsga_parallel import evaluate
        fitnesses = list(map(evaluate,invalid_dtc))
        #fitnesses = list(toolbox.map(toolbox.evaluate, copy.copy(invalid_dtc)))

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Select the next generation population
        pop = toolbox.select(pop + offspring, MU)
        record = self.stats.compile(pop)
        self.logbook.record(gen=gen, evals=len(invalid_ind), **record)
        #self.logbook
        #print(self.logbook.stream)
        #return dtcpop, pop, self.logbook, fitnesses
        self.invalid_dtc = list(copy.copy(invalid_dtc))
        self.fitnesses = list(fitnesses)
        return copy.copy(self.invalid_dtc), copy.copy(pop), copy.copy(self.logbook), copy.copy(self.fitnesses)

    def setnparams(self,nparams=10):
        self.nparams = nparams


    def main(self, MU, NGEN, seed=None):

        self.set_map()
        #dview = self.set_evaluate()
        from neuronunit.optimization import evaluate_as_module
        from neuronunit.optimization import nsga_parallel
        import array
        import random
        import numpy
        from deap import algorithms
        from deap import base

        subset = nsga_parallel.create_subset(nparams=self.nparams)

        #subset = nsga_parallel.create_subset(nparams=10)
        numb_err_f = 8
        toolbox, tools, history, creator, base = evaluate_as_module.import_list(self.ipp,subset,numb_err_f)
        self.toolbox = toolbox
        self.creator = creator
        self.tools = tools

        self.dview.push({'Individual':evaluate_as_module.Individual})
        self.dview.apply_sync(evaluate_as_module.import_list,self.ipp,subset,numb_err_f)
        get_trans_dict = evaluate_as_module.get_trans_dict
        self.td = get_trans_dict(subset)
        self.dview.push({'td':self.td })

        pop = toolbox.population(n = MU)
        pop = [ toolbox.clone(i) for i in pop ]
        self.dview.scatter('Individual',pop)

        #from neuronunit.optimization import nsga_parallel

        toolbox = self.toolbox
        dview = self.set_map()
        self.set_evaluate()
        import numpy
        from numpy import random
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
        #import pdb; pdb.set_trace()

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        #if len(invalid_ind)!=0:
        #    import pdb; pdb.set_trace()
        invalid_dtc = list(nsga_parallel.update_pop(invalid_ind,self.td))
        print(invalid_dtc)
        #import pdb; pdb.set_trace()

        invalid_dtc = list(filter(lambda dtc: float(dtc.rheobase['value']) > 0.0, invalid_dtc))
        assert len(invalid_dtc)>0
        #print(invalid_dtc)
        #import pdb; pdb.set_trace()

        #import pdb; pdb.set_trace()
        from neuronunit.optimization.nsga_parallel import evaluate
        fitnesses = list(map(evaluate,invalid_dtc))

        #import pdb; pdb.set_trace()

        #fitnesses = list(self.dview.map_sync(evaluate, invalid_dtc))
        #import pdb; pdb.set_trace()
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # This is just to assign the crowding distance to the individuals
        # no actual selection is done
        pop = toolbox.select(pop, len(pop))

        record = stats.compile(pop)
        logbook.record(gen=0, evals=len(invalid_ind), **record)
        self.logbook = logbook
        print(self.logbook.stream)


        for gen in range(1, NGEN):
            invalid_dtc, pop, logbook, fitnesses = self.evolve(pop,MU,gen)

        self.logbook.record(gen=gen, evals=len(invalid_ind), **record)
        import copy

        self.invalid_dtc = list(copy.copy(invalid_dtc))

        #assert type(self.invalid_dtc) is type(list)
        self.fitnesses = fitnesses
        #assert type(self.fitnesses) is type(list)

        return copy.copy(self.invalid_dtc), copy.copy(pop), copy.copy(self.logbook), copy.copy(self.fitnesses)



#import pdb; pdb.set_trace()

'''
NGEN = 2
for gen in range(1, NGEN):
    dtcpop, pop, logbook, fitnesses = N.evolve()
    df = zip(dtcpop,fitnesses)
    for d in df:
        print(d)
'''
