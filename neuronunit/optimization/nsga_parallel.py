##
# Assumption that this file was executed after first executing the bash: ipcluster start -n 8 --profile=default &
##
import matplotlib # Its not that this file is responsible for doing plotting, but it calls many modules that are, such that it needs to pre-empt
# setting of an appropriate backend.
matplotlib.use('agg')
import os
import quantities as pq
from numpy import random
import numpy as np

import sys
os.system('ipcluster start -n 8 --profile=default & sleep 15 ; python stdout_worker.py &')


import ipyparallel as ipp
rc = ipp.Client(profile='default')
rc[:].use_cloudpickle()
dview = rc[:]

from ipyparallel import depend, require, dependent
from neuronunit.tests import get_neab
THIS_DIR = os.path.dirname(os.path.realpath('nsga_parallel.py'))
this_nu = os.path.join(THIS_DIR,'../../')
sys.path.insert(0,this_nu)
from neuronunit import tests
#from deap import hypervolume
import deap



###
# GA parameters
MU = 12
NGEN = 4
CXPB = 0.9
###



import evaluate_as_module
toolbox, tools, history, creator, base = evaluate_as_module.import_list(ipp)
dview.push({'Individual':evaluate_as_module.Individual})
dview.apply(evaluate_as_module.import_list,ipp)


import model_parameters
param_dict = model_parameters.model_params
get_trans_dict = evaluate_as_module.get_trans_dict
td = get_trans_dict(param_dict)

dview.push({'td':td })

pop = toolbox.population(n = MU)
pop = [ toolbox.clone(i) for i in pop ]
dview.scatter('Individual',pop)
update_dtc_pop = evaluate_as_module.update_dtc_pop
dtcpop = update_dtc_pop(pop, td)



def dtc_to_rheo(dtc):
    from neuronunit.models import backends
    from neuronunit.models.reduced import ReducedModel
    import quantities as pq
    import numpy as np
    from neuronunit.tests import get_neab
    import evaluate_as_module
    model = ReducedModel(get_neab.LEMS_MODEL_PATH,name=str('vanilla'),backend='NEURON')
    model.set_attrs(dtc.attrs)
    dtc.scores = None
    dtc.scores = {}
    dtc.differences = None
    dtc.differences = {}
    score = get_neab.tests[0].judge(model,stop_on_error = False, deep_error = True)
    observation = score.observation
    prediction = score.prediction
    delta = evaluate_as_module.difference(observation,prediction)
    dtc.scores[str(get_neab.tests[0])] = score.sort_key
    dtc.rheobase = score.prediction
    return dtc

def map_wrapper(dtc):
    from neuronunit.models import backends
    from neuronunit.models.reduced import ReducedModel
    import quantities as pq
    import numpy as np
    from neuronunit.tests import get_neab
    model = ReducedModel(get_neab.LEMS_MODEL_PATH,name=str('vanilla'),backend='NEURON')
    model.set_attrs(dtc.attrs)
    get_neab.tests[0].prediction = dtc.rheobase
    model.rheobase = dtc.rheobase['value']
    for k,t in enumerate(get_neab.tests):
        if k>1:
            t.params = dtc.vtest[k]
            score = t.judge(model,stop_on_error = False, deep_error = True)
            dtc.scores[str(t)] = score.sort_key
            observation = score.observation
            prediction = score.prediction
            delta = evaluate_as_module.difference(observation,prediction)
            dtc.differences[str(t)] = delta
    return dtc

def evaluate(dtc):
    from neuronunit.tests import get_neab
    fitness = [-100.0 for i in range(0,8)]
    for k,t in enumerate(get_neab.tests):
        try:
            assert type(dtc.scores[str(t)]) is not type(None)
            fitness[k] = dtc.scores[str(t)]
        except:
            fitness[k] = -100.0

    print(fitness)
    return fitness[0],fitness[1],\
           fitness[2],fitness[3],\
           fitness[4],fitness[5],\
           fitness[6],fitness[7],


def update_pop(dtcpop):
    dtcpop = list(map(dtc_to_rheo,dtcpop))
    dtcpop = [ dtc for dtc in dtcpop if type(dtc.scores[str(get_neab.tests[0])]) is not type(None) ]

    dtcpop = list(map(evaluate_as_module.pre_format,dtcpop))
    dtcpop = list(dview.map(map_wrapper,dtcpop).get())
    return dtcpop

dtcpop = update_pop(dtcpop)

fitnesses = list(dview.map(evaluate,dtcpop).get())
print(dtcpop,fitnesses)
for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = fit
pop = tools.selNSGA2(pop, MU)

# only update the history after crowding distance has been assigned
deap.tools.History().update(pop)
### After an evaluation of error its appropriate to display error statistics
pf = tools.ParetoFront()
pf.update([toolbox.clone(i) for i in pop])
#hvolumes = []
#hvolumes.append(hypervolume(pf))

stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("min", np.min, axis=0)
stats.register("max", np.max, axis=0)
logbook = tools.Logbook()
logbook.header = "gen", "evals", "std", "min", "avg", "max"

record = stats.compile(pop)
logbook.record(gen=0, evals=len(pop), **record)
print(logbook.stream)


verbose = True
means = np.array(logbook.select('avg'))
gen = 1
verbose = True
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
    dtcpop = update_pop(dtcpop)
    fitnesses = list(dview.map(evaluate,dtcpop).get())
    print(dtcpop,fitnesses)
    print(gen)
    mf = np.mean(fitnesses)
    print(mf)
    import copy
    for ind, fit in zip(copy.copy(invalid_ind), fitnesses):
        ind.fitness.values = fit

    # Its possible that the offspring are worse than the parents of the penultimate generation
    # Its very likely for an offspring population to be less fit than their parents when the pop size
    # is less than the number of parameters explored. However this effect should stabelize after a
    # few generations, after which the population will have explored and learned significant error gradients.
    # Selecting from a gene pool of offspring and parents accomodates for that possibility.
    # There are two selection stages as per the NSGA example.
    # https://github.com/DEAP/deap/blob/master/examples/ga/nsga2.py
    # pop = toolbox.select(pop + offspring, MU)


    pop = tools.selNSGA2(offspring + pop, MU)

    record = stats.compile(pop)
    history.update(pop)

    logbook.record(gen=gen, evals=len(pop), **record)
    pf.update([toolbox.clone(i) for i in pop])
    means = np.array(logbook.select('avg'))
    pf_mean = np.mean([ i.fitness.values for i in pf ])


    # if the means are not decreasing at least as an overall trend something is wrong.
    print('means from logbook: {0} from manual meaning the fitness: {1}'.format(means,mf))
    print('means: {0} pareto_front first: {1} pf_mean {2}'.format(logbook.select('avg'), \
                                                        np.sum(np.mean(pf[0].fitness.values)),\
                                                        pf_mean))
os.system('conda install graphviz plotly cufflinks')
from neuronunit import plottools
plottools.surfaces(history,td)
best, worst = plottools.best_worst(history)
listss = [best , worst]
best_worst = update_dtc_pop(listss,td)
#import evaluate_as_module as em

best_worst , _ = check_rheobase(best_worst)
rheobase_values = [v.rheobase for v in dtcoffspring ]
dtchistory = update_dtc_pop(history.genealogy_history.values(),td)

import pickle
with open('complete_dump.p','wb') as handle:
   pickle.dump([pf,dtcoffspring,history,logbook,rheobase_values,best_worst,dtchistory,hvolumes],handle)

lists = pickle.load(open('complete_dump.p','rb'))
#dtcoffspring2,history2,logbook2 = lists[0],lists[1],lists[2]
plottools.surfaces(history,td)
import plottools
#reload(plottools)
#dtchistory = _pop(history.genealogy_history.values(),td)
#best, worst = plottools.best_worst(history)
#listss = [best , worst]
#best_worst = _pop(listss,td)
#best_worst , _ = check_rheobase(best_worst)
best = dtc
unev = pickle.load(open('un_evolved.p','rb'))
unev, rh_values_unevolved = unev[0], unev[1]
for x,y in enumerate(unev):
    y.rheobase = rh_values_unevolved[x]
dtcoffpsring.append(unev)
plottools.shadow(dtcoffspring,best_worst[0])
plottools.plotly_graph(history,dtchistory)
#plottools.graph_s(history)
plottools.plot_log(logbook)
plottools.not_just_mean(hvolumes,logbook)
plottools.plot_objectives_history(logbook)

#Although the pareto front surely contains the best candidate it cannot contain the worst, only history can.
#best_ind_dict_dtc = _pop(pf[0:2],td)
#best_ind_dict_dtc , _ = check_rheobase(best_ind_dict_dtc)



print(best_worst[0].attrs,' == ', best_ind_dict_dtc[0].attrs, ' ? should be the same (eyeball)')
print(best_worst[0].fitness.values,' == ', best_ind_dict_dtc[0].fitness.values, ' ? should be the same (eyeball)')

# This operation converts the population of virtual models back to DEAP individuals
# Except that there is now an added 11th dimension for rheobase.
# This is not done in the general GA algorithm, since adding an extra dimensionality that the GA
# doesn't utilize causes a DEAP error, which is reasonable.

plottools.bar_chart(best_worst[0])
plottools.pca(final_population,dtcpop,fitnesses,td)
#test_dic = bar_chart(best_worst[0])
plottools.plot_evaluate( best_worst[0],best_worst[1])
#plottools.plot_db(best_worst[0],name='best')
#plottools.plot_db(best_worst[1],name='worst')

plottools.plot_evaluate( best_worst[0],best_worst[1])
plottools.plot_db(best_worst[0],name='best')
plottools.plot_db(best_worst[1],name='worst')
