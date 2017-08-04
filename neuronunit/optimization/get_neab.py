import os,sys
import numpy as np
import matplotlib as matplotlib
matplotlib.use('Qt5Agg',warn=False)
import quantities as pq
import sciunit
import os
#Over ride any neuron units in the PYTHON_PATH with this one.
#only appropriate for development.
#THIS_DIR = os.path.dirname(os.path.realpath(__file__))
this_nu = os.path.join(os.getcwd(),'../..')
sys.path.insert(0,this_nu)

import neuronunit
from neuronunit import aibs
import pickle
try:
    IZHIKEVICH_PATH = os.path.join(os.getcwd(),'NeuroML2')
    assert os.path.isdir(IZHIKEVICH_PATH)
except AssertionError:
    # Replace this with the path to your Izhikevich NeuroML2 directory.
    IZHIKEVICH_PATH = os.path.join(os.getcwd(),'NeuroML2')

LEMS_MODEL_PATH = os.path.join(IZHIKEVICH_PATH,'LEMS_2007One.xml')
import time
#from pyneuroml import pynml
#import quantities as pq
from neuronunit import tests as nu_tests, neuroelectro
neuron = {'nlex_id': 'nifext_50'} # Layer V pyramidal cell
tests = []

dataset_id = 354190013  # Internal ID that AIBS uses for a particular Scnn1a-Tg2-Cre
                        # Primary visual area, layer 5 neuron.
observation = aibs.get_observation(dataset_id,'rheobase')

print(observation)
ne_pickle = os.path.join(os.getcwd(),"neuroelectro.pickle")

if os.path.isfile(ne_pickle):
    print('attempting to recover from pickled file')
    with open(ne_pickle, 'rb') as f:
        tests = pickle.load(f)
'''

else:
print('Checked path %s and no pickled file found. Commencing time intensive Download' % ne_pickle)
tests += [nu_tests.RheobaseTest(observation=observation)]#,name=None)]
test_class_params = [(nu_tests.InputResistanceTest),
                     (nu_tests.TimeConstantTest),
                     (nu_tests.CapacitanceTest),
                     (nu_tests.RestingPotentialTest),
                     (nu_tests.InjectedCurrentAPWidthTest),
                     (nu_tests.InjectedCurrentAPAmplitudeTest),
                     (nu_tests.InjectedCurrentAPThresholdTest)]


for nu in test_class_params:
    #use of the variable 'neuron' in this conext conflicts with the module name 'neuron'
    #at the moment it doesn't seem to matter as neuron is encapsulated in a class, but this could cause problems in the future.

    observation = nu.neuroelectro_summary_observation(neuron)
    nu.observation = observation
    print(observation)
    #print(params)
    #tests += [nu(observation=observation)]

with open('neuroelectro.pickle', 'wb') as handle:
    pickle.dump(tests, handle)
def update_amplitude(test,tests,score):
    rheobase = score.prediction['value']#first find a value for rheobase
    #then proceed with other optimizing other parameters.
    #for i in


    for i in [tests[-3],tests[-2],tests[-1]]:
        # Set current injection to just suprathreshold

        i.params['injected_square_current']['amplitude'] = rheobase*1.01
#Don't do the rheobase test. This is a serial bottle neck that must occur before any parallel optomization.
#Its because the optimization routine must have apriori knowledge of what suprathreshold current injection values are for each model.
#hooks = {tests[0]:{'f':update_amplitude}} #This is a trick to dynamically insert the method
'''
#update amplitude at the location in sciunit thats its passed to, without any loss of generality.
suite = sciunit.TestSuite("vm_suite",tests)#,hooks=hooks)
