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
#observation = aibs.get_observation(dataset_id,'rheobase')

ne_pickle = os.path.join(os.getcwd(),"neuroelectro.pickle")

if os.path.isfile(ne_pickle):
    print('attempting to recover from pickled file')
    with open(ne_pickle, 'rb') as f:
        tests = pickle.load(f)
#print(observation)
for t in tests:
    print(t.observation)

#update amplitude at the location in sciunit thats its passed to, without any loss of generality.
suite = sciunit.TestSuite("vm_suite",tests)#,hooks=hooks)
