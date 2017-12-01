
# coding: utf-8

# In[1]:

################
# GA parameters:
################
MU = 10; NGEN = 20; CXPB = 0.7
USE_CACHED_GA = False
# about 8 models will be made, excluding rheobase search.

################
# Grid search parameters:
# only 5 models, will be made excluding rheobase search
################
npoints = 2
nparams = 10
from neuronunit.optimization.model_parameters import model_params
provided_keys = list(model_params.keys())
#provided_keys = ['b','a','vpeak','v0','vt'] #implied number parameters is 2

USE_CACHED_GS = False

# !ulimit -n 2048
# There is on issue on the Mac with the number of open file handles.
# I think there is a file leak in LEMS or somewhere else and the command above needs to be run
# in the shell that spawned this notebook before this code can be run to completion.

import ipyparallel as ipp
c = ipp.Client(profile='default')
#c = Client()  # connect to IPyParallel cluster
#e = c.become_dask()  #
#print(e,'gets here')


#e.start_ipython_scheduler()
#info = e.start_ipython_workers()
#rc = ipp.Client(profile='default')
#from ipyparallel import depend, require, dependent
#dview = rc[:]
#import matplotlib as mpl
#mpl.use('Agg')
# setting of an appropriate backend.

import pickle
import numpy as np


# In[3]:

from neuronunit.optimization.nsga_object import NSGA
from neuronunit.optimization import exhaustive_search as es
from neuronunit.optimization import evaluate_as_module as eam
#from neuronunit import plottools

if USE_CACHED_GA:
    from deap import creator
    from deap import base
    from neuronunit.optimization import evaluate_as_module as eam

    NSGAO = NSGA()
    NSGAO.setnparams(nparams=nparams,provided_keys=provided_keys)
    td = eam.get_trans_dict(NSGAO.subset)

    weights = tuple([-1.0 for i in range(0,8)])
    creator.create("FitnessMin", base.Fitness, weights=weights)
    creator.create("Individual", list, fitness=creator.FitnessMin)
    [invalid_dtc, pop, logbook, fitnesses, history, pf] = pickle.load(open('ga_dump.p','rb'))
else:
    NSGAO = NSGA()
    NSGAO.setnparams(nparams=nparams,provided_keys=provided_keys)
    #import pdb; pdb.set_trace()
    invalid_dtc, pop, logbook, fitnesses, history, pf = NSGAO.main(MU, NGEN)
    with open('ga_dump.p','wb') as f:
       pickle.dump([invalid_dtc, pop, logbook, fitnesses, history, pf],f)
from neuronunit.plottools import dtc_to_plotting
from neuronunit import plottools
invalid_dtc = dview.map_sync(dtc_to_plotting,invalid_dtc)
plottools.use_dtc_to_plotting(invalid_dtc)
plottools.plot_log(logbook)
plottools.plot_objectives_history(logbook)




# In[5]:

if USE_CACHED_GS:
    dtcpopg = pickle.load(open('grid_dump.p','rb'))
else:
    dtcpopg = es.run_grid(npoints,nparams,provided_keys=provided_keys)
    with open('grid_dump.p','wb') as f:
       pickle.dump(dtcpopg,f)




# In[6]:

def error(dtc):
    """
    Overall error function for a DTC
    Returns the root-mean-square error over all the tests
    """
    return np.sqrt(np.mean(np.square(list(dtc.scores.values()))))

def sorted_dtcs(dtcpop):
    """
    Returns dtc,error tuples sorted from low to high error
    """
    return sorted([(dtc,error(dtc)) for dtc in dtcpop],key=lambda x:x[1])


def sorted_history(pop):
    """
    Returns dtc,error tuples sorted from low to high error
    """
    return sorted([ind.fitness for ind in pop],key=lambda x:x[1])


minimagr, mini = sorted_dtcs(dtcpopg)[0]
maximagr, maxi = sorted_dtcs(dtcpopg)[-1]

def pop2dtc(pop,NSGAO):
    '''
    dview.push({'Individual':eam.Individual})
    Individual = ipp.Reference('Individual')
    pop = [toolbox.clone(i) for i in pop ]
    '''
    from deap import base, creator
    toolbox = base.Toolbox()
    NDIM = 10
    weights = tuple([-1.0 for i in range(0,NDIM)])
    creator.create("FitnessMin", base.Fitness, weights=weights)
    creator.create("Individual", list, fitness=creator.FitnessMin)
    from neuronunit.optimization import evaluate_as_module as eam
    td = eam.get_trans_dict(NSGAO.subset)
    dtc_pop = eam.update_dtc_pop(pop,td)

    for i,p in enumerate(pop):
        dtc_pop[i].scores = p.fitness.values
        dtc_pop[i].error = None
        dtc_pop[i].error = np.sqrt(np.mean(np.square(list(p.fitness.values))))
    sorted_list  = sorted([(dtc,dtc.error) for dtc in dtc_pop],key=lambda x:x[1])
    return sorted_list, dtc_pop

sorted_list_pf, pareto_dtc = pop2dtc(pf,NSGAO)

h = list(history.genealogy_history.values())
sorted_list_h, dtc_pop = pop2dtc(h,NSGAO)

minimaga = sorted_list_pf[0][0]
maximaga = sorted_list_pf[-1][0]

#history_fitness = sorted_history(h)
miniga = sorted_list_pf[0][1]
maxiga = sorted_list_pf[-1][1]

# In[7]:

# quantize distance between minimimum error and maximum error.
quantize_distance = list(np.linspace(mini,maxi,10))

# check that the nsga error is in the bottom 1/5th of the entire error range.
print('Report: ')
print("Success" if bool(miniga < quantize_distance[2]) else "Failure")
print("The nsga error %f is in the bottom 1/5th of the entire error range" % miniga)
print("Minimum = %f; 20th percentile = %f; Maximum = %f" % (miniga,quantize_distance[2],maxiga))
#print("Solution Space Error Distribution well explored?" if bool(maxiga > quantize_distance[-2]) else "Failure")


# In[8]:

# This function reports on the deltas brute force obtained versus the GA found attributes.
from neuronunit.optimization import model_parameters as modelp
from deap.benchmarks.tools import hypot

mp = modelp.model_params
for k,v in minimagr.attrs.items():
    dimension_length = hypot(float(np.max(mp[k])),float(np.min(mp[k])))
    sdi1D = hypot(float(minimaga.attrs[k]),float(v))
    relative_distance = sdi1D/dimension_length
    print('the difference between brute force candidates model parameters and the GA\'s model parameters:')
    print('parameter values: ',float(v),float(minimaga.attrs[k]),'parameter names: ',v,k)
    print('the relative distance scaled by the length of the parameter dimension of interest:')
    print(relative_distance)


    #unittest.test_5_agreement()


# In[9]:

print('the difference between the bf error and the GA\'s error:')
print('grid search:')
from numpy import square, mean, sqrt
rmsg = sqrt(mean(square(list(minimagr.scores.values()))))
print(rmsg)
print('ga:')
try:
    rmsga = sqrt(mean(square(list(minimaga.scores.values()))))
except:
    rmsga = sqrt(mean(square(list(minimaga.scores))))

print(rmsga)
print('maximum error:', maximagr)
#dtc = invalid_dtc[0]
#from neuronunit.optimization.evaluate_as_module import pre_format
#dtc = pre_format(dtc)
#import pdb; pdb.set_trace()
#parameter_list = list(dtc.vtest.values())
#print(parameter_list[0])

import copy
#dtc = copy.copy(invalid_dtc[0])
#plottools.plot_suspicious(dtc)


"""Tests of NeuronUnit test classes"""
import unittest
#import os
#os.system('ipcluster start -n 8 --profile=default & sleep 5;')



class TestBackend(unittest.TestCase):
    def setup(self):
        self.dtcpopg = pickle.load(open('grid_dump.p','rb'))
        from neuronunit.optimization import evaluate_as_module
        from deap import creator
        from deap import base
        from neuronunit.optimization import evaluate_as_module as eam

        NSGAO = NSGA()
        NSGAO.setnparams(nparams=nparams,provided_keys=provided_keys)
        td = eam.get_trans_dict(NSGAO.subset)

        weights = tuple([-1.0 for i in range(0,8)])
        creator.create("FitnessMin", base.Fitness, weights=weights)
        creator.create("Individual", list, fitness=creator.FitnessMin)
        [invalid_dtc, pop, logbook, fitnesses, history, pf] = pickle.load(open('ga_dump.p','rb'))
        self.invalid_dtc = invalid_dtc
        self.pop = pop
        self.logbook = logbook
        self.fitnesses = fitnesses
        self.history = history
        self.pf = pf

    def test_pyNN_neuron0(self):


        from neuronunit.models.reduced import ReducedModel
        from neuronunit.optimization.get_neab import tests as T
        from neuronunit.optimization import get_neab

        from neuronunit.optimization import evaluate_as_module
        from neo import AnalogSignal
        dtc = self.invalid_dtc[0]
        model = ReducedModel(get_neab.LEMS_MODEL_PATH,name=str('vanilla'),backend='NEURON')
        model.set_attrs(dtc.attrs)
        model.rheobase = dtc.rheobase['value']
        #model2 = ReducedModel(get_neab.LEMS_MODEL_PATH,name=str('vanilla'),backend='NEURON',DTC=dtc)
        score = T[-1].judge(model,stop_on_error = False, deep_error = True)
        #dtc.vm1 = list(model.get_membrane_potential())
        dtc.vm0 = list(model.results['vm'])
        model = ReducedModel(get_neab.LEMS_MODEL_PATH,name=str('vanilla'),backend='pyNN')
        model.set_attrs(dtc.attrs)
        model.rheobase = dtc.rheobase['value']
        score = T[1].judge(model,stop_on_error = False, deep_error = True)
        #dtc.vm0 = list(model.get_membrane_potential())
        dtc.vm1 = list(model.results['vm'])
        dtc.tvec = list(model.results['t'])
        return dtc

    def tests_ga_vs_grid1(self):
        dtcpopg = pickle.load(open('grid_dump.p','rb'))
        [invalid_dtc, pop, logbook, fitnesses, history, pf] = pickle.load(open('ga_dump.p','rb'))
        provided_keys=list(dtcpopg[0].attrs.keys())
        NSGAO = NSGA()
        NSGAO.setnparams(nparams=nparams,provided_keys=provided_keys)
        sorted_list_pf, pareto_dtc = pop2dtc(pf,NSGAO)
        minimagr, mini = sorted_dtcs(dtcpopg)[0]
        maximagr, maxi = sorted_dtcs(dtcpopg)[-1]
        h = list(history.genealogy_history.values())
        sorted_list_h, dtc_pop = pop2dtc(h,NSGAO)
        miniga = sorted_list_pf[0][1]
        # In[7]:

        # quantize distance between minimimum error and maximum error.
        quantize_distance = list(np.linspace(mini,maxi,10))
        self.assertGreater(miniga, quantize_distance[2])
        self.assertGreater(maxiga, quantize_distance[-3])
        # check that the nsga error is in the bottom 1/5th of the entire error range.
        print('Report: ')
        print("Success" if bool(miniga < quantize_distance[2]) else "Failure")
        print("The nsga error %f is in the bottom 1/5th of the entire error range" % miniga)
        print("Minimum = %f; 20th percentile = %f; Maximum = %f" % (mini,quantize_distance[2],maxi))
        print("Solution Space Error Distribution well explored?" if bool(maxiga > quantize_distance[-2]) else "Failure")




#if USE_CACHED_GS and USE_CACHED_GA:
#    unittest.main()

# Two things to find:
# How close togethor are the model parameters in parameter space (hyper volume), relative to the dimensions of the HV?
# ie get the euclidian distance between the two sets of model parameters.

#
#
#

#exit()
