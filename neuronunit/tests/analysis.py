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
global model
model = ReducedModel(get_neab.LEMS_MODEL_PATH,name='vanilla',backend='NEURON')
print(model)
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

with open('nsga_matrix.pickle', 'rb') as handle:
    nsga_matrix=pickle.load(handle)

parameters_min=nsga_matrix[1][1]
#parameters_max=nsga_matrix[3]

if os.path.exists(str(os.getcwd())+"/neuroelectro.pickle"):
    print('attempting to recover from pickled file')
    with open('neuroelectro.pickle', 'rb') as handle:
        tests = pickle.load(handle)

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


#model = ReducedModel(get_neab.LEMS_MODEL_PATH,name='vanilla',backend='NEURON')
import pickle
with open('score_matrix.pickle', 'rb') as handle:
    matrix=pickle.load(handle)
'''
print(matrix)
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
score0max,attrs0max,rheobase0max=matrix[storagesmax[0][0]]
score1max,attrs1max,rheobase1max=matrix[storagesmax[0][1]]
'''


from scoop import futures


def test_to_model(local_test_methods,attrs):
    import matplotlib.pyplot as plt
    import copy
    global model
    model.local_run()
    model.update_run_params(attrs)
    model.re_init(attrs)
    tests = get_neab.suite.tests
    tests[local_test_methods].judge(model)
    if hasattr(tests[local_test_methods],'related_data'):
        if tests[local_test_methods].related_data != None:
            print(tests[local_test_methods].related_data['vm'])
    plt.plot(copy.copy(model.results['t']), copy.copy(model.results['vm']))
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

get_neab.suite.tests[0].prediction={}
get_neab.suite.tests[0].prediction['value']=nsga_matrix[0][2]*qt.pA

local_test_methods = [ i for i,j in enumerate(get_neab.suite.tests) ]
from itertools import repeat

list_of_tups=[]
list_of_tups.append((nsga_matrix[1][1],'maximum',nsga_matrix[1][2]))

list_of_tups.append((nsga_matrix[0][1],'minimum',nsga_matrix[0][2]))

'''
list_of_tups.append((attrs0, name, rheobase0))
name='min_two'
list_of_tups.append((attrs1, name, rheobase1))
name='max_one'
list_of_tups.append((attrs0max, name, rheobase0max))
name='max_two'
list_of_tups.append((attrs1max, name, rheobase1max))
'''

if __name__ == "__main__":
    #print(list_of_tups[0])
    #completed3 = map(build_single,list_of_tups)
    completed1 = list(futures.map(build_single,list_of_tups))
    print('\n\n\n a', completed1, '\n\n\n')
    completed2 = list(futures.map(build_single,list_of_tups))
    print('\n\n\n b', completed2, '\n\n\n')


#sbuild_single(attrs1,rheobase1)
