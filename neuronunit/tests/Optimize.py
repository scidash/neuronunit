
# coding: utf-8

# In[17]:

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


# In[16]:

# This example is from https://github.com/OpenSourceBrain/IzhikevichModel.
IZHIKEVICH_PATH = os.getcwd()+str('/neuronunit/tests/NeuroML2') # Replace this the path to your 
                                                                       # working copy of 
                                                                       # github.com/OpenSourceBrain/IzhikevichModel.  
#LEMS_MODEL_PATH = os.path.join(IZHIKEVICH_PATH,)
LEMS_MODEL_PATH=IZHIKEVICH_PATH+str('/LEMS_2007One.xml')

model = ReducedModel(IZHIKEVICH_PATH+str('/LEMS_2007One.xml'),name='vanilla')

#some testing of functionality
#TODO rm later.

from neuronunit.models import backends
NeuronObject=backends.NEURONBackend(IZHIKEVICH_PATH)
NeuronObject.load_model()


# In[8]:

import quantities as pq
from neuronunit import tests as nu_tests, neuroelectro
neuron = {'nlex_id': 'nifext_50'} # Layer V pyramidal cell
tests = []

dataset_id = 354190013  # Internal ID that AIBS uses for a particular Scnn1a-Tg2-Cre 
                        # Primary visual area, layer 5 neuron.
observation = aibs.get_observation(dataset_id,'rheobase')
tests += [nu_tests.RheobaseTest(observation=observation)]
    
test_class_params = [(nu_tests.InputResistanceTest,None),
                     (nu_tests.TimeConstantTest,None),
                     (nu_tests.CapacitanceTest,None),
                     (nu_tests.RestingPotentialTest,None),
                     (nu_tests.InjectedCurrentAPWidthTest,None),
                     (nu_tests.InjectedCurrentAPAmplitudeTest,None),
                     (nu_tests.InjectedCurrentAPThresholdTest,None)]

for cls,params in test_class_params:
    observation = cls.neuroelectro_summary_observation(neuron)
    tests += [cls(observation,params=params)]
    
def update_amplitude(test,tests,score):
    rheobase = score.prediction['value']
    for i in [5,6,7]:
        print(tests[i])
        # Set current injection to just suprathreshold
        tests[i].params['injected_square_current']['amplitude'] = rheobase*1.01 
    
hooks = {tests[0]:{'f':update_amplitude}}
suite = sciunit.TestSuite("vm_suite",tests,hooks=hooks)


# In[4]:
model = ReducedModel(IZHIKEVICH_PATH+str('/LEMS_2007One.xml'),name='vanilla')
#model = ReducedModel(LEMS_MODEL_PATH,name='vanilla',backend='NEURONbackend')


# In[5]:

SUO = '/home/mnt/scidash/sciunitopt'
if SUO not in sys.path:
    sys.path.append(SUO)


# In[6]:

from types import MethodType
def optimize(self,model,rov):
    best_params = None
    best_score = None
    from sciunitopt.deap_config_simple_sum import DeapCapsule
    dc=DeapCapsule()
    pop_size=12
    ngen=5                                  
    pop = dc.sciunit_optimize(self,LEMS_MODEL_PATH,pop_size,ngen,rov,
                                                         NDIM=2,OBJ_SIZE=2)
    return pop#(best_params, best_score, model)

my_test = tests[0]
my_test.verbose = False
my_test.optimize = MethodType(optimize, my_test) # Bind to the score.


# In[7]:

rov = np.linspace(-100,-40,1000)
pop = my_test.optimize(model,rov)
#print('pareto front top value in pf hall of fame')
#print('best params',best_params,'best_score',best_score, 'model',model)


# In[13]:

print("%.2f mV" % np.mean([p[0] for p in pop]))

