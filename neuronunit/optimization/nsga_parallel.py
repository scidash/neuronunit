##
# Assumption that this file was executed after first executing the bash: ipcluster start -n 8 --profile=default &
##
import matplotlib # Its not that this file is responsible for doing plotting, but it calls many modules that are, such that it needs to pre-empt
# setting of an appropriate backend.
matplotlib.use('agg')
import os
import quantities as pq
from numpy import random

import sys
import ipyparallel as ipp
#from ipyparallel import Client
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
MU = 4
NGEN = 3
CXPB = 0.9
###


def check_paths():
    '''
    import paths and test for consistency
    '''
    import neuronunit
    from neuronunit.models.reduced import ReducedModel
    from neuronunit.tests import get_neab
    model = ReducedModel(get_neab.LEMS_MODEL_PATH,name='vanilla',backend='NEURON')
    model.backend = 'NEURON'
    return neuronunit.models.__file__

path_serial = check_paths()
paths_parallel = dview.apply_async(check_paths).get_dict()
assert path_serial == paths_parallel[0]

import evaluate_as_module
toolbox, tools, history, creator, base = evaluate_as_module.import_list(ipp)
dview.push({'Individual':evaluate_as_module.Individual})
dview.apply(evaluate_as_module.import_list,ipp)
#print(returns.get())

def get_trans_dict(param_dict):
    trans_dict = {}
    for i,k in enumerate(list(param_dict.keys())):
        trans_dict[i]=k
    return trans_dict
import model_parameters
param_dict = model_parameters.model_params

def dt_to_ind(dtc,td):
    '''
    Re instanting data transport container at every update dtcpop
    is Noneifying its score attribute, and possibly causing a
    performance bottle neck.
    '''
    ind =[]
    for k in td.keys():
        ind.append(dtc.attrs[td[k]])
    ind.append(dtc.rheobase)
    return ind
@require('copy')
def update_dtc_pop(pop, td):
    '''
    inputs a population of genes/alleles, the population size MU, and an optional argument of a rheobase value guess
    outputs a population of genes/alleles, a population of individual object shells, ie a pickleable container for gene attributes.
    Rationale, not every gene value will result in a model for which rheobase is found, in which case that gene is discarded, however to
    compensate for losses in gene population size, more gene samples must be tested for a successful return from a rheobase search.
    If the tests return are successful these new sampled individuals are appended to the population, and then their attributes are mapped onto
    corresponding virtual model objects.
    '''
    import copy
    import numpy as np
    pop = [toolbox.clone(i) for i in pop ]
    import evaluate_as_module

    def transform(ind):
        dtc = evaluate_as_module.DataTC()
        print(dtc)
        #dtc = DataTC()
        #print(dtc)
        param_dict = {}
        for i,j in enumerate(ind):
            param_dict[td[i]] = str(j)
        dtc.attrs = param_dict
        dtc.evaluated = False
        return dtc


    if len(pop) > 0:
        dtcpop = list(dview.map_sync(transform, pop))
        #dtcpop = list(copy.copy(dtcpop))
    else:
        # In this case pop is not really a population but an individual
        # but parsimony of naming variables
        # suggests not to change the variable name to reflect this.
        dtcpop = transform(pop)
    return dtcpop
td = get_trans_dict(param_dict)

dview.push({'td':td })

pop = toolbox.population(n = MU)
pop = [ toolbox.clone(i) for i in pop ]
dview.scatter('Individual',pop)

dtcpop = update_dtc_pop(pop, td)



def dtc_to_rheo(dtc):
    from neuronunit.models import backends
    from neuronunit.models.reduced import ReducedModel
    import quantities as pq
    import numpy as np
    from neuronunit.tests import get_neab
    import evaluate_as_module
    model = ReducedModel(get_neab.LEMS_MODEL_PATH,name=str('vanilla'),backend='NEURON')
    #model.load_model()
    model.set_attrs(dtc.attrs)
    print(model)
    print(model.attrs)
    dtc.scores = None
    dtc.scores = {}
    #get_neab.tests[0].dview = dview
    dtc.differences = None
    dtc.differences = {}
    score = get_neab.tests[0].judge(model,stop_on_error = False, deep_error = True)
    observation = score.observation
    prediction = score.prediction
    delta = evaluate_as_module.difference(observation,prediction)
    #dtc.differences[str(get_neab.tests[0])] = delta
    #dtc.scores[str(get_neab.tests[0])] = score.sort_key
    dtc.rheobase = score.prediction
    return dtc

def map_wrapper(dtc):
    from neuronunit.models import backends
    from neuronunit.models.reduced import ReducedModel
    import quantities as pq
    import numpy as np
    from neuronunit.tests import get_neab
    model = ReducedModel(get_neab.LEMS_MODEL_PATH,name=str('vanilla'),backend='NEURON')
    #model.load_model()
    model.set_attrs(dtc.attrs)
    print(model)
    print(model.attrs)
    get_neab.tests[0].prediction = dtc.rheobase
    model.rheobase = dtc.rheobase['value']
    #differences = []
    for k,t in enumerate(get_neab.tests):
        if k>1:
            t.params = dtc.vtest[k]
            score = t.judge(model,stop_on_error = False, deep_error = True)
            dtc.scores[str(t)] = score.sort_key
            #observation = score.observation
            #prediction = score.prediction
            #delta = np.abs(observation-prediction)
            #dtc.differences[str(t)] = delta

    return dtc

dtcpop = list(map(dtc_to_rheo,dtcpop))
dtcpop = list(map(evaluate_as_module.pre_format,dtcpop))
dtcpop = list(dview.map(map_wrapper,dtcpop).get())



print(scores)
rh_values_unevolved = [v.rheobase for v in dtcpop ]


new_checkpoint_path = str('un_evolved')+str('.p')
import pickle
with open(new_checkpoint_path,'wb') as handle:#
    pickle.dump([dtcpop,rh_values_unevolved], handle)


for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = fit


pop = tools.selNSGA2(pop, MU)

# only update the history after crowding distance has been assigned
deap.tools.History().update(pop)


### After an evaluation of error its appropriate to display error statistics
#pf = tools.ParetoFront()
pf.update([toolbox.clone(i) for i in pop])
#hvolumes = []
#hvolumes.append(hypervolume(pf))

record = stats.compile(pop)
logbook.record(gen=0, evals=len(pop), **record)
print(logbook.stream)

def rh_difference(unit_predictions):
    unit_observations = get_neab.tests[0].observation['value']
    to_r_s = unit_observations.units
    unit_predictions = unit_predictions.rescale(to_r_s)
    unit_delta = np.abs( np.abs(unit_observations)-np.abs(unit_predictions) )
    print(unit_delta)
    return float(unit_delta)

verbose = True
means = np.array(logbook.select('avg'))
gen = 1
rh_mean_status = np.mean([ v.rheobase for v in dtcpop ])
rhdiff = rh_difference(rh_mean_status * pq.pA)
print(rhdiff)
verbose = True
while (gen < NGEN and means[-1] > 0.05):
    # Although the hypervolume is not actually used here, it can be used
    # As a terminating condition.
    # hvolumes.append(hypervolume(pf))
    gen += 1
    print(gen)
    offspring = tools.selNSGA2(pop, len(pop))
    if verbose:
        for ind in offspring:
            print('what do the weights without values look like? {0}'.format(ind.fitness.weights[:]))
            print('what do the weighted values look like? {0}'.format(ind.fitness.wvalues[:]))
            #print('has this individual been evaluated yet? {0}'.format(ind.fitness.valid[0]))
            print(rhdiff)
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
    dtcoffspring = update_dtc_pop(copy.copy(invalid_ind), trans_dict) #(copy.copy(invalid_ind), td)
    #import evaluate_as_module as em
    #dtcpop , _ = check_rheobase(dtcpop)
    dtcoffspring , _ =  check_rheobase(copy.copy(dtcoffspring))
    rh_mean_status = np.mean([ v.rheobase for v in dtcoffspring ])
    rhdiff = rh_difference(rh_mean_status * pq.pA)
    print('the difference: {0}'.format(rh_difference(rh_mean_status * pq.pA)))
    # sometimes fitness is assigned in serial, although slow gives access to otherwise hidden
    # stderr/stdout
    # fitnesses = []
    # for v in dtcoffspring:
    #    fitness.append(evaluate(v))
    #import evaluate_as_module
    dtcpop = [ v for v in dtcpop if v.rheobase > 0.0 ]
    for d in dtcpop:
        print('testing dubious rheobase values \n\n\n')
        d = check_current(d.rheobase, d)
    fitnesses = list(dview.map_sync(evaluate_as_module.evaluate, copy.copy(dtcoffspring)))
    #fitnesses = list(dview.map_sync(evaluate, copy.copy(dtcoffspring)))

    mf = np.mean(fitnesses)

    for ind, fit in zip(copy.copy(invalid_ind), fitnesses):
        ind.fitness.values = fit
        if verbose:
            print('what do the weights without values look like? {0}'.format(ind.fitness.weights))
            print('what do the weighted values look like? {0}'.format(ind.fitness.wvalues))
            print('has this individual been evaluated yet? {0}'.format(ind.fitness.valid))

    # Its possible that the offspring are worse than the parents of the penultimate generation
    # Its very likely for an offspring population to be less fit than their parents when the pop size
    # is less than the number of parameters explored. However this effect should stabelize after a
    # few generations, after which the population will have explored and learned significant error gradients.
    # Selecting from a gene pool of offspring and parents accomodates for that possibility.
    # There are two selection stages as per the NSGA example.
    # https://github.com/DEAP/deap/blob/master/examples/ga/nsga2.py
    # pop = toolbox.select(pop + offspring, MU)

    # keys = history.genealogy_tree.keys()
    # Optionally
    # Grab evaluated history its and chuck them into the mixture.
    # This may cause stagnation of evolution however.
    # We want to select among the best from the whole history of the GA, not just penultimate and present generations.
    # all_hist = [ history.genealogy_history[i] for i in keys if history.genealogy_history[i].fitness.valid == True ]
    # pop = tools.selNSGA2(offspring + all_hist, MU)

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
