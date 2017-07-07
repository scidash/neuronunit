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
#global model
#model = ReducedModel(get_neab.LEMS_MODEL_PATH,name='vanilla',backend='NEURON')

import grid_search as gs
model=gs.model

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


if os.path.exists(str(os.getcwd())+"/neuroelectro.pickle"):
    print('attempting to recover from pickled file')
    with open('neuroelectro.pickle', 'rb') as handle:
        tests = pickle.load(handle)
for i, j in enumerate(tests):
    print(i,j)
    for k,v in j.observation.items():
        print(k,v)

with open('vmpop.pickle', 'rb') as handle:
    vmpop = pickle.load(handle)


with open('score_matrixt.pickle', 'rb') as handle:
    score_matrixt = pickle.load(handle)
    pdb.set_trace()
#pdb.set_trace()


#with open('nsga_matrix_worst.pickle', 'rb') as handle:
#    nsga_matrix=pickle.load(handle)

#parameters_min=nsga_matrix[1][1]
#pdb.set_trace()
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

#import pickle
#with open('score_matrix.pickle', 'rb') as handle:
#    matrix=pickle.load(handle)


#with open('nsga_vmpop_worst.pickle', 'rb') as handle:
#    vmpop=pickle.load(handle)

pdb.set_trace()

from scoop import futures

def test_to_model(local_test_methods,attrs):
    import matplotlib.pyplot as plt
    import copy
    global model
    global nsga_matrix
    model.local_run()
    model.update_run_params(nsga_matrix[1][1])
    model.re_init(nsga_matrix[1][1])
    tests = None
    tests = get_neab.suite.tests
    tests[0].prediction={}
    tests[0].prediction['value']=nsga_matrix[0][2]*qt.pA
    tests[0].params['injected_square_current']['amplitude']=nsga_matrix[1][2]*qt.pA
    #score = get_neab.suite.judge(model)#pass
    if local_test_methods in [4,5,6]:
        tests[local_test_methods].params['injected_square_current']['amplitude']=nsga_matrix[1][2]*qt.pA
    #model.results['vm'] = [ 0 ]
    model.re_init(nsga_matrix[0][1])
    tests[local_test_methods].generate_prediction(model)
    #tests[local_test_methods].judge(model)
    #print(local_test_methods)
    #pdb.set_trace()
    #pdb.set_trace()
    '''
    if local_test_methods in [0,4,6,7]:

        #print(tests[local_test_methods].observation)
        #tests[local_test_methods].prediction['value']
        if 'value' in tests[local_test_methods].observation.keys() and 'value' in tests[local_test_methods].prediction.keys():
            observ_vector = [tests[local_test_methods].observation['value'].item() for i in model.results['t']]
            pred_vector = [tests[local_test_methods].prediction['value'].item() for i in model.results['t']]

            plt.plot(model.results['t'], observ_vector, label='observation')
            plt.plot(model.results['t'], pred_vector, label='prediction')
    '''
    plt.plot(model.results['t'], model.results['vm'], label='best candidate of GA')

    model.update_run_params(nsga_matrix[1][1])
    tests = None

    tests = get_neab.suite.tests
    #tests[0].prediction={}
    tests[0].prediction={}
    tests[0].prediction['value']=nsga_matrix[1][2]*qt.pA
    tests[0].params['injected_square_current']['amplitude']=nsga_matrix[1][2]*qt.pA

    if local_test_methods in [4,5,6]:
        tests[local_test_methods].params['injected_square_current']['amplitude']=nsga_matrix[1][2]*qt.pA

    #model.results['vm'] = [ 0 ]
    model.re_init(nsga_matrix[1][1])
    #tests[local_test_methods].judge(model)
    tests[local_test_methods].generate_prediction(model)
    '''
    if local_test_methods in [0,4,6,7]:
        if 'value' in tests[local_test_methods].observation.keys() and 'value' in tests[local_test_methods].prediction.keys():
            observ_vector = [tests[local_test_methods].observation['value'].item() for i in model.results['t']]
            pred_vector = [tests[local_test_methods].prediction['value'].item() for i in model.results['t']]

            plt.plot(model.results['t'], observ_vector, label='observation')
            plt.plot(model.results['t'], pred_vector, label='prediction')
    '''
    plt.plot(model.results['t'],model.results['vm'],label='worst candidate of GA')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)

    plt.title(str(tests[local_test_methods]))
    plt.savefig(str(tests[local_test_methods])+str('.png'))
    plt.clf()
    model.results['vm']=None
    model.results['t']=None
    tests[local_test_methods].related_data=None
    local_test_methods=None
    return 0


def build_single(indexs):
    attrs,name,rheobase=indexs
    from scoop import futures
    judged = list(futures.map(test_to_model,local_test_methods,repeat(attrs)))
    return 0#judged

local_test_methods = [ i for i,j  in enumerate(get_neab.suite.tests) ]
from itertools import repeat

list_of_tups=[]
#list_of_tups.append((nsga_matrix[1][1],'maximum',nsga_matrix[1][2]))
#get_neab.suite.tests[0].prediction={}
#get_neab.suite.tests[0].prediction['value']=nsga_matrix[1][2]*qt.pA
list_of_tups.append((nsga_matrix[0][1],'minimum',nsga_matrix[0][2]))


if __name__ == "__main__":
    completed1 = list(futures.map(build_single,vmpop))
