import os
#import sys
#import numpy as np
#import pdb
#import quantities as pq
#import time
#from pyneuroml import pynml
#import quantities as pq

import matplotlib as matplotlib
matplotlib.use('agg',warn=False)
import sciunit
import neuronunit
from neuronunit import aibs

import pickle
THIS_DIR = os.path.dirname(os.path.realpath(__file__))
LEMS_MODEL_PATH = os.path.realpath(os.path.join(THIS_DIR,'..','models','NeuroML2','LEMS_2007One.xml'))
from neuronunit import tests as _, neuroelectro
from neuronunit.tests import fi
from neuronunit.tests import passive
from neuronunit.tests import waveform


neuron = {'nlex_id': 'nifext_50'} # Layer V pyramidal cell
tests = []

#http://api.brain-map.org/api/v2/data/query.xml?criteria=
#model::ApiCellTypesSpecimenDetail
#,rma::options[num_rows$eqall]

dataset_id = 354190013  # Internal ID that AIBS uses for a particular Scnn1a-Tg2-Cre
                        # Primary visual area, layer 5 neuron.


# this file writing operation needs to be threadlocked
observation = aibs.get_observation(dataset_id,'rheobase')
ne_pickle = os.path.join(THIS_DIR,"neuroelectro.pickle")

if os.path.isfile(ne_pickle):
    print('attempting to recover from pickled file')
    with open(ne_pickle, 'rb') as f:
        tests = pickle.load(f)

else:

    #print('Checked path %s and no pickled file found. Commencing time intensive Download' % ne_pickle)
    tests += [fi.RheobaseTestP(observation=observation)]
    test_class_params = [(passive.InputResistanceTest,None),
                     (passive.TimeConstantTest,None),
                     (passive.CapacitanceTest,None),
                     (passive.RestingPotentialTest,None),
                     (waveform.InjectedCurrentAPWidthTest,None),
                     (waveform.InjectedCurrentAPAmplitudeTest,None),
                     (waveform.InjectedCurrentAPThresholdTest,None)]


    for cls,params in test_class_params:
        print(cls,params)
        #use of the variable 'neuron' in this conext conflicts with the module name 'neuron'
        #at the moment it doesn't seem to matter as neuron is encapsulated in a class, but this could cause problems in the future.
        observation = cls.neuroelectro_summary_observation(neuron)
        tests += [cls(observation)]

#with open(ne_pickle, 'wb') as f:
#    pickle.dump(tests, f)

def update_amplitude(test,tests,score):
    rheobase = score.prediction['value']#first find a value for rheobase
    #then proceed with other optimizing other parameters.

    for i in [4,5,6]:
        # Set current injection to just suprathreshold

        tests[i].params['injected_square_current']['amplitude'] = rheobase * 1.01 # I feel that 1.01 may lead to more than one spike
        # in marginal cases.

#Don't do the rheobase test. This is a serial bottle neck that must occur before any parallel optomization.
#Its because the optimization routine must have apriori knowledge of what suprathreshold current injection values are for each model.
hooks = {tests[0]:{'f':update_amplitude}} #This is a trick to dynamically insert the method
#update amplitude at the location in sciunit thats its passed to, without any loss of generality.
suite = sciunit.TestSuite("vm_suite",tests)#,hooks=hooks)
