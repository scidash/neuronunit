import numpy as np
import time
import inspect
from types import MethodType
import quantities as pq
from quantities.quantity import Quantity
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sciunit
import os, sys
thisnu = str(os.getcwd())+'/../..'
sys.path.insert(0,thisnu)
from scoop import futures
import sciunit.scores as scores
import neuronunit.capabilities as cap
import get_neab
from neuronunit.models import backends
import sciunit.scores as scores
from neuronunit.models import backends
from neuronunit.models.reduced import ReducedModel
import neuronunit.capabilities as cap


import sciunit.scores as scores
import quantities as qt
import pdb
#vm = VirtualModel()
#import matplotlib.plot as plt
import matplotlib.pyplot as plt
AMPL = 0.0*pq.pA
DELAY = 100.0*pq.ms
DURATION = 1000.0*pq.ms
from scipy.optimize import curve_fit

import os,sys
import numpy as np
import matplotlib as matplotlib
matplotlib.use('agg')
import quantities as pq
import sciunit

#Over ride any neuron units in the PYTHON_PATH with this one.
#only appropriate for development.
thisnu = str(os.getcwd())+'/../..'
sys.path.insert(0,thisnu)
print(sys.path)

import neuronunit
from neuronunit import aibs
import pdb
import pickle
from scoop import futures
from scoop import utils
IZHIKEVICH_PATH = os.getcwd()+str('/NeuroML2') # Replace this the path to your
LEMS_MODEL_PATH = IZHIKEVICH_PATH+str('/LEMS_2007One.xml')
import time
from pyneuroml import pynml
import quantities as pq
from neuronunit import tests as nu_tests, neuroelectro

neural_data = {'nlex_id': 'nifext_50'} #Layer V pyramidal cell
# Don't use the label neuron
#that label will be needed by the HOC/NEURON object which also needs to occupy the same name space


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
    tests += [nu_tests.RheobaseTest(observation=observation)]
    test_class_params = [(nu_tests.InputResistanceTest,None),
                         (nu_tests.TimeConstantTest,None),
                         (nu_tests.CapacitanceTest,None),
                         (nu_tests.RestingPotentialTest,None),
                         (nu_tests.InjectedCurrentAPWidthTest,None),
                         (nu_tests.InjectedCurrentAPAmplitudeTest,None),
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
    #for i in


    for i in [4,5,6]:

required_capabilities = (cap.ReceivesSquareCurrent,
                         cap.ProducesSpikes)
params = {'injected_square_current':
            {'amplitude':100.0*pq.pA, 'delay':DELAY, 'duration':DURATION}}
name = "Rheobase test"
description = ("A test of the rheobase, i.e. the minimum injected current "
               "needed to evoke at least one spike.")
score_type = scores.RatioScore
guess=None
lookup = {} # A lookup table global to the function below.
verbose=True
import quantities as pq
units = pq.pA


model = ReducedModel(get_neab.LEMS_MODEL_PATH,name='vanilla',backend='NEURON')
import pickle
with open('score_matrix.pickle', 'rb') as handle:
    matrix=pickle.load(handle)


matrix3=[]
for x,y,rheobase in matrix:
    for i in x:
        matrix2=[]
        for j in i:
            if j==None:
                j=10.0
            matrix2.append((j,rheobase))
            #print(j,rheobase)
        matrix3.append(matrix2)
storagei = [ np.sum(i) for i in matrix3 ]
storagesmin=np.where(storagei==np.min(storagei))
storagesmax=np.where(storagei==np.max(storagei))
score0,attrs0,rheobase0=matrix[storagesmin[0][0]]
score1,attrs1,rheobase1=matrix[storagesmin[0][1]]
score0max,attrs0max,rheobase0=matrix[storagesmax[0][0]]
score1max,attrs1max,rheobase1=matrix[storagesmax[0][1]]


class VirtualModel:
    '''
    This is a pickable dummy clone
    version of the NEURON simulation model
    It does not contain an actual model, but it can be used to
    wrap the real model.
    This Object class serves as a data type for storing rheobase search
    attributes and other useful parameters,
    with the distinction that unlike the NEURON model this class
    can be transported across HOSTS/CPUs
    '''
    def __init__(self):
        self.lookup={}
        self.rheobase=None
        self.previous=0
        self.run_number=0
        self.attrs=None
        self.name=None
        self.s_html=None
        self.results=None



def build_single(indexs):
    attrs,name=indexs
    #This method is only used to check singlular sets of hard coded parameters.]
    #This medthod is probably only useful for diagnostic purposes.

    model.attrs=attrs
    model.update_run_params(attrs)
    model.h.psection()
    print('!!!!!\n\n')
    print(model.attrs)
    model.update_run_params(model.attrs)
    model.h.psection()
    print(model.attrs)
    print('!!!!!\n\n')

    model.name = str(attrs)
    #rh_value=searcher2(f,rh_param,vms)
    get_neab.suite.tests[0].prediction={}
    get_neab.suite.tests[0].prediction['value']=52.22222222222222 *qt.pA
    score = get_neab.suite.judge(model)#passing in model, changes model
    print('this is the spike count!!!!!!!!!!!: \n \n \n')
    print('this is the spike count!!!!!!!!!!!: \n \n \n')
    print('this is the spike count!!!!!!!!!!!: \n \n \n')
    print(model.get_spike_count())
    plt.plot(model.results['t'],model.results['vm'])
    plt.savefig(name+'.png')
    plt.clf()


print(score0,attrs0)
name='min_one'
list_of_tups=[]
list_of_tups.append((attrs0,name))
name='min_two'
list_of_tups.append((attrs1,name))
name='max_one'
list_of_tups.append((attrs0max,name))
name='max_two'
list_of_tups.append((attrs1max,name))
from scoop import futures

if __name__ == "__main__":

    completed=list(futures.map(build_single,list_of_tups))


#sbuild_single(attrs1,rheobase1)
