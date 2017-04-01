import os, sys
import time
import inspect
from types import MethodType
import pickle

import pdb
import numpy as np
import quantities as pq
from quantities.quantity import Quantity
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scoop import futures

import sciunit
import sciunit.scores as scores

thisnu = str(os.getcwd())+'/../..'
sys.path.insert(0,thisnu)

import neuronunit.capabilities as cap
from neuronunit.tests import get_neab
from neuronunit.models import backends
from neuronunit.models.reduced import ReducedModel
import neuronunit.capabilities as cap


#vm = VirtualModel()
#import matplotlib.plot as plt
AMPL = 0.0*pq.pA
DELAY = 100.0*pq.ms
DURATION = 1000.0*pq.ms
from scipy.optimize import curve_fit
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
units = pq.pA


model = ReducedModel(get_neab.LEMS_MODEL_PATH,name='vanilla',backend='NEURON')

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(THIS_DIR,'score_matrix.pickle')
with open(path, 'rb') as handle:
    matrix=pickle.load(handle)


matrix3=[]
for x,y,rheobase in matrix:
    for i in x:
        matrix2=[]
        for j in i:
            if j==None:
                j=10.0
            matrix2.append((j,rheobase))
            print(j,rheobase)
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



def build_single(attrs,name):
    #This method is only used to check singlular sets of hard coded parameters.]
    #This medthod is probably only useful for diagnostic purposes.

    model.attrs=attrs
    model.update_run_params(attrs)
    model.name = str(attrs)
    #rh_value=searcher2(f,rh_param,vms)
    get_neab.suite.tests[0].prediction={}
    get_neab.suite.tests[0].prediction['value']=52.22222222222222 *pq.pA
    score = get_neab.suite.judge(model)#passing in model, changes model
    for k,v in score.related_data.items():#.results['vm']
        print(k,v)

        for i,j in v.items():
            print(i,j)
            plt.clf()
            time=np.linspace(0,1600,len(j['vm']))
            plt.xlabel(str(v))
            plt.ylabel(str(i))
            plt.plot(time,j['vm'])
            plt.savefig(str(k)+name+'.png')
            plt.clf()

            #pdb.set_trace()
print(score0,attrs0)
name='min_one'
build_single(attrs0,name)
name='min_two'
build_single(attrs1,name)
name='max_one'
build_single(attrs0max,name)
name='max_two'
build_single(attrs1max,name)

#sbuild_single(attrs1,rheobase1)
