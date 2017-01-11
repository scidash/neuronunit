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
from deap.benchmarks.tools import diversity, convergence, hypervolume
from deap import creator
from deap import tools

import os
import sys

#Over ride any neuron units in the PYTHON_PATH with this one.
#only appropriate for development.

thisnu = str(os.getcwd())+'/../../'
sys.path.insert(0,thisnu)
print(sys.path)


# coding: utf-8

# In[17]:
#Retain jupyter notebook cell #s and comments, such that the Notebook can quickly be remade with a conversion.
#get_ipython().magic('load_ext autoreload')
#get_ipython().magic('autoreload 2')
#get_ipython().magic('matplotlib notebook')
import os,sys
import numpy as np
import matplotlib.pyplot as plt
import quantities as pq


import sciunit


import neuronunit
from neuronunit import aibs
from neuronunit.models.reduced import ReducedModel
import pdb
import pickle
# In[16]:

# This example is from https://github.com/OpenSourceBrain/IzhikevichModel.
#model_path = os.getcwd()+str('/neuronunit/software_tests/NeuroML2') # Replace this the path to your
                                                                       # working copy of
                                                                       # github.com/OpenSourceBrain/IzhikevichModel.
#file_path=model_path+str('/LEMS_2007One.xml')

IZHIKEVICH_PATH = os.getcwd()+str('/NeuroML2') # Replace this the path to your
LEMS_MODEL_PATH = IZHIKEVICH_PATH+str('/LEMS_2007One.xml')


# In[8]:

import quantities as pq
from neuronunit import tests as nu_tests, neuroelectro
neuron = {'nlex_id': 'nifext_50'} # Layer V pyramidal cell
tests = []

dataset_id = 354190013  # Internal ID that AIBS uses for a particular Scnn1a-Tg2-Cre
                        # Primary visual area, layer 5 neuron.
observation = aibs.get_observation(dataset_id,'rheobase')


if os.path.exists(str(os.getcwd())+"/neuroelectro.pickle"):
    print('attempting to recover from pickled file')
    with open('neuroelectro.pickle', 'rb') as handle:
        tests = pickle.load(handle)

else:
    print('checked path:')
    print(str(os.getcwd())+"/neuroelectro.pickle")
    print('no pickled file found. Commencing time intensive Download')

    #(nu_tests.TimeConstantTest,None),                           (nu_tests.InjectedCurrentAPAmplitudeTest,None),
    tests += [nu_tests.RheobaseTest(observation=observation)]
    test_class_params = [(nu_tests.InputResistanceTest,None),
                         (nu_tests.RestingPotentialTest,None),
                         (nu_tests.InjectedCurrentAPWidthTest,None),
                         (nu_tests.InjectedCurrentAPThresholdTest,None)]



    for cls,params in test_class_params:
        #use of the variable 'neuron' in this conext conflicts with the module name 'neuron'
        #at the moment it doesn't seem to matter as neuron is encapsulated in a class, but this could cause problems in the future.


        observation = cls.neuroelectro_summary_observation(neuron)
        tests += [cls(observation,params=params)]

    with open('neuroelectro.pickle', 'wb') as handle:
        pickle.dump(tests, handle)

def update_amplitude(test,tests,score):
    rheobase = score.prediction['value']#first find a value for rheobase
    #then proceed with other optimizing other parameters.


    print(len(tests))
    #pdb.set_trace()
    for i in [2,3,4]:
        # Set current injection to just suprathreshold
        tests[i].params['injected_square_current']['amplitude'] = rheobase*1.01


#Do the rheobase test. This is a serial bottle neck that must occur before any parallel optomization.
#Its because the optimization routine must have apriori knowledge of what suprathreshold current injection values are for each model.


hooks = {tests[0]:{'f':update_amplitude}} #This is a trick to dynamically insert the method
#update amplitude at the location in sciunit thats its passed to, without any loss of generality.
suite = sciunit.TestSuite("vm_suite",tests,hooks=hooks)


class Test:
    def __init__(self):
        pass

    def judge(self,model=None):
        pass # already implemented, returns a score

    def optimize(self,model=None):

        best_params = None
        best_score = None#-np.inf
        #call to the GA.
        #import deap_config
        from deap_config_nsga2 import deap_capsule
        dc=deap_capsule()

        pop_size=12
        ngen=10
        #These parameters are over written
        NDIM=2
        OBJ_SIZE=2
        #range_of_values=self.range_of_values
        seed_in=1


        import os

        IZHIKEVICH_PATH = os.getcwd()+str('/NeuroML2') # Replace this the path to your
        LEMS_MODEL_PATH = IZHIKEVICH_PATH+str('/LEMS_2007One.xml')
        from neuronunit.models import backends
        from neuronunit.models.reduced import ReducedModel



        #Its because Reduced model is the base class that calling super on SingleCellModel does not work.
        #model = ReducedModel(LEMS_MODEL_PATH,name='vanilla',backend='NEURON')
        dc.model=ReducedModel(LEMS_MODEL_PATH,name='vanilla',backend='NEURON')
        param=['vr','a','b']
        rov=[]
        rov0 = np.linspace(-67,-50,10)
        rov1 = np.linspace(0.015,0.045,10)
        half=0.0019999999/2.0
        rov2 = np.linspace(-0.0019999999-half,-0.0019999999+half,10)
        rov.append(rov0)
        rov.append(rov1)
        rov.append(rov2)

        best_params, best_score, model =dc.sciunit_optimize(suite,param,pop_size,ngen,rov,NDIM,OBJ_SIZE,seed_in=1)
        return (best_params, best_score, model)




import matplotlib as matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
if __name__ == "__main__":

    #def ff(xx):
    #        return 3-(xx-2)**2

    #def gg(xx):
    #        return 15-(xx-2)**6

    #brute_force_optimize(ff)
    #logbook,y,x=sciunit_optimize(ff,3)
    #best_params, best_score, model = sciunit_optimize(ff,3)

    #range_of_values=np.linspace(-65.0,-55.0,1000)
    t=Test()
    pop, best_score, model=t.optimize()
    plt.hold(True)
    for i in xrange(0,9):
        plt.plot(pop[i].time_trace,pop[i].voltage_trace)
    plt.savefig('best 10')
    #(self.model,pop[0],pop[0].sciunitscore)
    print('pareto front top value in pf hall of fame')
    #print('best params',best_params,'best_score',best_score, 'model',model)
