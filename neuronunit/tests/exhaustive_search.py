


#import mpi4py
#from mpi4py import MPI
#COMM = MPI.COMM_WORLD
#SIZE = COMM.Get_size()
#RANK = COMM.Get_rank()

import pdb
import get_neab
#pdb.set_trace()
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
import sciunit.scores as scores

from neuronunit.models import backends
import neuronunit.capabilities as cap
import testsrh
dir(testsrh)
#import neuronunit.capabilities.spike_functions as sf
#from neuronunit import neuroelectro
#from .channel import *


from scoop import futures


###################
# These parameters cause insufficient data.
# run this model once exactly such that I can properly write code that
# catches the exception
#{'//izhikevich2007Cell': {'vpeak': '30.0', 'vr': '-53.4989145966',
#'a': '0.0303440140536', 'b': '-2.38706324769e-08'}}j
from neuronunit.models.reduced import ReducedModel

model = ReducedModel(get_neab.LEMS_MODEL_PATH,name='vanilla',backend='NEURON')
model=model.load_model()


#model = ReducedModel(get_neab.LEMS_MODEL_PATH,name='vanilla',backend='NEURON')
#model=model.load_model()
vr = np.linspace(-75.0,-50.0,10)
a = np.linspace(0.015,0.045,2)
b = np.linspace(-3.5*10E-9,-0.5*10E-9,2)
k = np.linspace(7.0E-4-+7.0E-5,7.0E-4+70E-5,10)
C = np.linspace(1.00000005E-4-1.00000005E-5,1.00000005E-4+1.00000005E-5,10)

c = np.linspace(-55,-60,10)
d = np.linspace(0.050,0.2,10)
v0 = np.linspace(-75.0,-45.0,10)
vt =  np.linspace(-50.0,-30.0,10)
vpeak =np.linspace(30.0,40.0,10)
container=[]
import pdb

def build_single():

    #{'//izhikevich2007Cell': {'vpeak': '30.0', 'vr': '-53.4989145966',
    #'a': '0.0303440140536', 'b': '-2.38706324769e-08'}}j
    attrs={}
    attrs['//izhikevich2007Cell']={}
    attrs['//izhikevich2007Cell']['a']=0.0303440140536
    attrs['//izhikevich2007Cell']['b']=-2.38706324769e-08
    attrs['//izhikevich2007Cell']['vpeak']=30.0
    attrs['//izhikevich2007Cell']['vr']=-53.4989145966
    print(attrs)


    import quantities as qt
    #v=[v for v in get_neab.suite.tests[0].observation.values()][0]
    #get_neab.suite.tests[0].prediction={}
    #get_neab.suite.tests[0].prediction['value']=individual.rheobase*qt.pA

    model.update_run_params(attrs)
    #get_neab.suite.tests[0].prediction={}
    score = get_neab.suite.judge(model)#passing in model, changes model
    error = []
    error = [ abs(i.score) for i in score.unstack() ]
    print(score)
    pdb.set_trace()

    #model.s_html=

build_single()



def func2map(iter_arg,suite):
    attrs={}
    attrs['//izhikevich2007Cell']={}
    param=['a','b']
    i,j=iter_arg
    print(i,j)
    model.name=str(i)+str(j)#+str(k)+str(k)
    attrs['//izhikevich2007Cell']['a']=i
    attrs['//izhikevich2007Cell']['b']=j
    attrs['//izhikevich2007Cell']['vpeak']=40.0

    print(attrs)

    model.update_run_params(attrs)

    score = suite.judge(model)#passing in model, changes model
    error = []
    error = [ abs(i.score) for i in score.unstack() ]
    print(score)
    model.s_html=(score.to_html(),attrs)
    print(attrs)

    model.run_number+=1
    RUN_TIMES='{}{}{}'.format('counting simulation run times on models',model.results['run_number'],model.run_number)
    return model.s_html

if __name__ == "__main__":

    '''
    iter_list=[ (i,j,r,l,m,n,o,p,q) for i in a for j in b \
                                     for r in vr for l in k \
                                     for m in C for n in c \
                                     for o in d for p in v0 for q in vt ]
    '''
    #pdb.set_trace()
    #observation = neuronunit.neuro_electro.neuroelectro_summary_observation(neuron)


    #iter_arg = [ iter_list[i] for i in range(RANK, len(iter_list), SIZE) ]
    bg_bf=time.time()


    from itertools import repeat
    iter_list=[ (i,j) for i in a for j in b ]

    score_matrix=list(futures.map(func2map,iter_list,repeat(suite)))
#
#    main()
'''
COMM.barrier()
score_matrix2 = COMM.gather(score_matrix, root=0)
if RANK == 0:
    score_matrix=[]
    for p in score_matrix:
        score_matrix.extend(p)
    print(score_matrix)
    end_bf=time.time()
    whole_time=end_bf-bg_bf
    f=open('brute_force_time','w')
    f.write(str(whole_time))
    f.close()
    import pickle

    with open('score_matrix.pickle', 'wb') as handle:
        pickle.dump(score_matrix, handle)


else:
   score_matrix=None
'''
