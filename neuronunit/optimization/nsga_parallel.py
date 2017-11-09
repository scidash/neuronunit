##
# Assumption that this file was executed after first executing the bash: ipcluster start -n 8 --profile=default &
##
import matplotlib # Its not that this file is responsible for doing plotting, but it calls many modules that are, such that it needs to pre-empt
# setting of an appropriate backend.
matplotlib.use('agg')
import os
from numpy import random
import numpy as np

#import sys
#os.system('ipcluster start -n 8 --profile=default & sleep 15 ; python stdout_worker.py &')

import ipyparallel as ipp
rc = ipp.Client(profile='default')
#rc.Client.become_dask()

rc[:].use_cloudpickle()
dview = rc[:]

# scatter 'id', so id=0,1,2 on engines 0,1,2
dview.scatter('id', rc.ids, flatten=True)
print("Engine IDs: ", dview['id'])
# create a Reference to `id`. This will be a different value on each engine
ref = ipp.Reference('id')


from ipyparallel import depend, require, dependent
# Import get_neab has to happen exactly here. It has to be called only on
# controller (rank0, it has)
from neuronunit.optimization import get_neab

from neuronunit import tests

def dtc_to_rheo(dtc):
    from neuronunit.models.reduced import ReducedModel
    from neuronunit.optimization import get_neab
    from neuronunit.optimization import evaluate_as_module
    model = ReducedModel(get_neab.LEMS_MODEL_PATH,name=str('vanilla'),backend='NEURON')

    model.set_attrs(dtc.attrs)
    #dtc.cell_name = model._backend.cell_name
    #dtc.current_src_name = model._backend.current_src_name
    dtc.scores = None
    dtc.scores = {}
    dtc.differences = None
    dtc.differences = {}
    score = get_neab.tests[0].judge(model,stop_on_error = False, deep_error = True)
    observation = score.observation
    prediction = score.prediction
    delta = evaluate_as_module.difference(observation,prediction)
    dtc.differences[str(get_neab.tests[0])] = delta
    dtc.scores[str(get_neab.tests[0])] = score.sort_key
    dtc.rheobase = score.prediction
    return dtc

def bind_score_to_dtc(dtc):
    #import evaluate_as_module
    from neuronunit.optimization import evaluate_as_module

    #from neuronunit.models import backends
    from neuronunit.models.reduced import ReducedModel
    import quantities as pq
    import numpy as np
    from neuronunit.optimization import get_neab

    model = ReducedModel(get_neab.LEMS_MODEL_PATH,name=str('vanilla'),backend='NEURONMemory')


    model.set_attrs(dtc.attrs)
    get_neab.tests[0].prediction = dtc.rheobase
    model.rheobase = dtc.rheobase['value']
    if dtc.rheobase['value'] <= 0.0:
        return dtc

    for k,t in enumerate(get_neab.tests):
        if k>0:

            t.params = dtc.vtest[k]
            score = t.judge(model,stop_on_error = False, deep_error = False)
            #import pdb; pdb.set_trace()
            dtc.scores[str(t)] = score.sort_key
            try:
                observation = score.observation
                prediction = score.prediction
                delta = evaluate_as_module.difference(observation,prediction)
                dtc.differences[str(t)] = delta
            except:
                pass
    return dtc

def evaluate(dtc):

    from neuronunit.optimization import get_neab
    import numpy as np
    fitness = [ -100.0 for i in range(0,8)]
    for k,t in enumerate(dtc.scores.keys()):
        if dtc.rheobase['value'] > 0.0:
            fitness[k] = dtc.scores[str(t)]
        else:
            fitness[k] = -100.0

    #print(fitness)
    return fitness[0],fitness[1],\
           fitness[2],fitness[3],\
           fitness[4],fitness[5],\
           fitness[6],fitness[7],
'''
def federate_attribute(dtcpop,attribute):
    dtc = dtcpop[0]
    for dtc in dtcpop:
        dtc.cached_attrs = None
        dtc.cached_attrs = attribute
    # add all elments into the dictionary thats the 1st element of the list
    for dtci in dtcpop:
        dtc.cached_attrs.update(dtci.cached_attrs)

    # add all elments into every dictionary belonging to every element in the element of the list
    for dtcj in dtcpop:
        dtcj.cached_attrs.update(dtc.cached_attrs)

    for k, dtck in enumerate(dtcpop):
        current = len(dtck.cached_attrs)
        assert current == previous
        previous = current
    return dtcpop
'''
def update_pop(pop,td):
    '''
    Inputs a population of genes (pop).
    Returned neuronunit scored DTCs (dtcpop).
    '''
    # this method converts a population of genes to a population of Data Transport Containers,
    # Which act as communicatable data types for storing model attributes.
    # Rheobase values are found on the DTCs
    # DTCs for which a rheobase value of x (pA)<=0 are filtered out
    # DTCs are then scored by neuronunit, using neuronunit models that act in place.

    from neuronunit.optimization import model_parameters as modelp
    from neuronunit.optimization import evaluate_as_module
    update_dtc_pop = evaluate_as_module.update_dtc_pop
    pre_format = evaluate_as_module.pre_format
    dtcpop = list(update_dtc_pop(pop, td))
    for d in dtcpop:
        assert type(d) is not type(None)
    assert len(dtcpop) != 0
    dtcpop = list(map(dtc_to_rheo,dtcpop))
    print('\n\n\n\n\n rheobase complete \n\n\n\n')
    for d in dtcpop:
        assert type(d) is not type(None)
    assert len(dtcpop) != 0
    dtcpop = list(map(pre_format,dtcpop))
    print('\n\n\n\n\n preformat complete \n\n\n\n')
    for d in dtcpop:
        assert type(d) is not type(None)
    assert len(dtcpop) != 0
    from neuronunit.optimization.exhaustive_search import parallel_method
    dtcpop = list(dview.map_sync(parallel_method,dtcpop))
    print('\n\n\n\n\n score calculation complete \n\n\n\n')
    #import pdb; pdb.set_trace()

    import copy
    for d in dtcpop:
        assert type(d) is not type(None)
    assert len(dtcpop) != 0
    return copy.copy(dtcpop)


def create_subset(nparams=10):
    from neuronunit.optimization import model_parameters as modelp
    import numpy as np
    mp = modelp.model_params
    key_list = list(mp.keys())
    reduced_key_list = key_list[0:nparams]
    subset = { k:mp[k] for k in reduced_key_list }
    return subset
'''
Deprecated
def main(MU=12, NGEN=4, CXPB=0.9, nparams=10):
    import deap
    import copy

    from neuronunit.optimization import evaluate_as_module
    from neuronunit.optimization import model_parameters

    scores = []

    subset = create_subset(nparams=10)
    numb_err_f = 8
    toolbox, tools, history, creator, base = evaluate_as_module.import_list(ipp,subset,numb_err_f)

    dview.push({'Individual':evaluate_as_module.Individual})
    dview.apply_sync(evaluate_as_module.import_list,ipp,subset,numb_err_f)
    get_trans_dict = evaluate_as_module.get_trans_dict
    td = get_trans_dict(subset)
    dview.push({'td':td })

    pop = toolbox.population(n = MU)
    pop = [ toolbox.clone(i) for i in pop ]
    dview.scatter('Individual',pop)

    dtcpop = update_pop(pop,td)
    assert len(dtcpop) == len(pop)

    #for dtc in dtcpop:
    #    scores.append(dtc.scores)
    fitnesses = list(dview.map_sync(evaluate,dtcpop))
    assert len(fitnesses) == len(pop)

    print(dtcpop,fitnesses)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    pop = tools.selNSGA2(pop, MU)

    # only update the history after crowding distance has been assigned
    deap.tools.History().update(pop)
    # After an evaluation of error its appropriate to display error statistics
    pf = tools.ParetoFront()
    pf.update([toolbox.clone(i) for i in pop])

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)
    stats.register("avg", np.mean, axis=0)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(pop), **record)
    means = np.array(logbook.select('avg'))
    difference_progress = []
    gen = 1
    difference_progress.append(np.mean([v for dtc in dtcpop for v in dtc.differences.values()  ]))

    verbose = True
    difference_progress = []
    while (gen < NGEN):# and means[-1] > 0.05):
        # Although the hypervolume is not actually used here, it can be used
        # As a terminating condition.
        # hvolumes.append(hypervolume(pf))
        gen += 1
        print(gen)
        offspring = tools.selNSGA2(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]

        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= CXPB:
                toolbox.mate(ind1, ind2)
            toolbox.mutate(ind1)
            toolbox.mutate(ind2)
            # deleting the fitness values is what renders them invalid.
            # The invalidness is used as a flag for recalculating them.
            # Their fitneess needs deleting since the attributes which generated these values have been mutated
            # and hence they need recalculating
            # Mutation also implies breeding, if a gene is mutated it was also recently recombined.
            del ind1.fitness.values, ind2.fitness.values

        invalid_ind = []
        for ind in offspring:
            if ind.fitness.valid == False:
                invalid_ind.append(ind)
        # Need to make sure that _pop does not replace instances of the same model
        # Thus waisting computation.

        dtcpop = update_pop(invalid_ind,td)
        assert len(dtcpop) == len(invalid_ind)

        for dtc in dtcpop:
            scores.append(dtc.scores)
        fitnesses = list(dview.map_sync(evaluate,dtcpop))
        assert len(dtcpop) == len(fitnesses)

        difference_progress.append(np.mean([v for dtc in dtcpop for v in dtc.differences.values()  ]))
        mf = np.mean(fitnesses)


        for ind, fit in zip(copy.copy(invalid_ind), fitnesses):
            ind.fitness.values = fit

        # Its possible that the offspring are worse than the parents of the penultimate generation
        # Its very likely for an offspring population to be less fit than their parents when the pop size
        # is less than the number of parameters explored. However this effect should stabelize after a
        # few generations, after which the population will have explored and learned significant error gradients.
        # Selecting from a gene pool of offspring and parents accomodates for that possibility.
        # There are two selection stages as per the NSGA example.
        # https://github.com/DEAP/deap/blob/master/examples/ga/nsga2.py

        #pop = toolbox.select(pop + offspring, MU)


        pop = tools.selNSGA2(offspring + pop, MU)
        #assert len(dtcpop) == len(pop)

        record = stats.compile(pop)
        history.update(pop)
        logbook.record(gen=gen, evals=len(pop), **record)
        pf.update([toolbox.clone(i) for i in pop])
        means = np.array(logbook.select('avg'))
        pf_mean = np.mean([ i.fitness.values for i in pf ])
        #return difference_progress, fitnesses, pf, logbook, pop, dtcpop, stats

        # if the means are not decreasing at least as an overall trend something is wrong.
        print('means from logbook: {0} from manual meaning the fitness: {1}'.format(means,mf))
        print('means: {0} pareto_front first: {1} pf_mean {2}'.format(logbook.select('avg'), \
                                                            np.sum(np.mean(pf[0].fitness.values)),\
                                                            pf_mean))
        if gen==NGEN:
            return difference_progress, fitnesses, pf, logbook, pop, dtcpop, stats, scores
'''
