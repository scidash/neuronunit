


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
a = np.linspace(0.015,0.045,10)
b = np.linspace(-3.5*10E-9,-0.5*10E-9,10)
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

#build_single()





def model2map(iter_arg):#This method must be pickle-able for scoop to work.
    vm=VirtualModel()
    attrs={}
    attrs['//izhikevich2007Cell']={}
    param=['a','b']
    param=['a','b','vr','vpeak']#,'k']#,'C']#,'c','d','v0','k','vt','vpeak']#,'d'

    i,j,k=iter_arg
    print(i,j)
    model.name=str(i)+str(j)#+str(k)+str(k)
    attrs['//izhikevich2007Cell']['a']=i
    attrs['//izhikevich2007Cell']['b']=j
    attrs['//izhikevich2007Cell']['vpeak']=k
    #attrs['//izhikevich2007Cell']['vr']=l
    print(attrs)
    vm.attrs=attrs
    #model.update_run_params(attrs)
    return vm



def func2map(iter_arg,suite):#This method must be pickle-able for scoop to work.

    model.update_run_params(iter_arg.attrs)

    import quantities as qt
    get_neab.suite.tests[0].prediction={}
    get_neab.suite.tests[0].prediction['value']=iter_arg.rheobase*qt.pA
    import os
    import os.path
    from scoop import utils

    score = get_neab.suite.judge(model)#passing in model, changes model
    model.run_number+=1
    #results=model.results
    try:
        error = []
        error = [ abs(i.score) for i in score.unstack() ]
        s_html=score.to_html()
    except Exception as e:
        '{}'.format('Insufficient Data')
        f=open('scoop_log_'+str(utils.getHosts()),'w')
        f.write(str(attrs))
        f.close()
        if np.sum(error)!=0:
            error = [ (10.0+i)/2.0 for i in error ]
        else:
            error = [ 10.0 for i in range(0,8) ]

        s_html=score.to_html()
    error=error
    assert results
    return score



class VirtualModel:
    '''
    This is a shell for the real model, it contains
    model parameters and other data which can readily be pickeled.
    Unlike the actual model which contains unpickable HOC code
    .
    '''
    def __init__(self):
        self.lookup={}
        self.previous=0
        self.run_number=0
        self.attrs=None
        self.name=None

param=['a','b']#,'vr','vpeak']#,'k']#,'C']#,'c','d','v0','k','vt','vpeak']#,'d'

import neuronunit.capabilities as cap
#from neuronunit.models import backends
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
import quantities as pq
units = pq.pA
verbose=True



def f(ampl,vm):
    print(vm, ampl)

    if float(ampl) not in vm.lookup:
        current = params.copy()['injected_square_current']
        print('current, previous = ',ampl,vm.previous)
        uc={'amplitude':ampl}
        current.update(uc)
        current={'injected_square_current':current}
        vm.run_number+=1
        print('model run number',vm.run_number)
        #model.attrs=vm.attrs
        model.update_run_params(vm.attrs)
        assert vm.attrs==model.attrs
        print(vm.attrs)
        print(model.attrs)
        model.inject_square_current(current)
        vm.previous=ampl
        n_spikes = model.get_spike_count()
        verbose=True
        if verbose:
            print("Injected %s current and got %d spikes" % \
                    (ampl,n_spikes))
        vm.lookup[float(ampl)] = n_spikes
        #vm3=copy.copy(vm)

        return vm

    if float(ampl) in vm.lookup:
        print('model_in lookup')
        return vm


#rheobase=None
def main2(ind,guess_attrs=None):
    vm=VirtualModel()
    if guess_attrs!=None:
        for i, p in enumerate(param):
            value=str(guess_attrs[i])
            model.name=str(model.name)+' '+str(p)+str(value)
            if i==0:
                attrs={'//izhikevich2007Cell':{p:value }}
            else:
                attrs['//izhikevich2007Cell'][p]=value
        vm.attrs=attrs
        guess_attrs=None#stop reentry into this condition during while,
    else:
        import copy
        vm.attrs=ind.attrs

    begin_time=time.time()
    while_true=True
    while(while_true):
        from itertools import repeat
        if len(vm.lookup)==0:
            steps2 = np.linspace(50,190,4.0)
            steps = [ i*pq.pA for i in steps2 ]
            lookup2=list(map(f,steps,repeat(vm)))#,repeat(model)))
            print(lookup2)
            print(type(lookup2))
            print(len(lookup2))

        m = lookup2[0]

        sub=[]
        supra=[]
        for k,v in m.lookup.items():
            if v==1:
                while_true=False
                print('hit')
                end_time=time.time()
                total_time=end_time-begin_time
                print(total_time)
                #pdb.set_trace()
                return (m.run_number,k,m.attrs)#a
                break
            elif v==0:
                sub.append(k)
            elif v>0:
                supra.append(k)
        sub=np.array(sub)
        supra=np.array(supra)
        if len(sub) and len(supra):
            steps2 = np.linspace(sub.max(),supra.min(),4.0)
            steps = [ i*pq.pA for i in steps2 ]

        elif len(sub):
            steps2 = np.linspace(sub.max(),2*sub.max(),4.0)
            steps = [ i*pq.pA for i in steps2 ]
        elif len(supra):
            steps2 = np.linspace(-1*(supra.min()),supra.min(),4.0)
            steps = [ i*pq.pA for i in steps2 ]

        lookup2=list(map(f,steps,repeat(vm)))


def evaluate2(individual, guess_value=None):#This method must be pickle-able for scoop to work.
    #import rheobase_old2 as rh
    model=VirtualModel()

    if guess_value != None:

        individual.lookup={}
        vm=VirtualModel()
        import copy
        vm.attrs=copy.copy(individual.attrs)
        #pdb.set_trace()
        vm=f(guess_value,vm)
        #pdb.set_trace()
        for k,v in vm.lookup.items():
            if v==1:
                individual.rheobase=k
                #print('succeeds in parallel case \n \n\n\n\n\n\n\n')
                return individual
            if v!=1:
                #pdb.set_trace()
                guess_value = None#more trial and error.
        #if individual.lookup

        #in case first guess no good. enable

    if guess_value == None:
        (run_number,k,attrs)=main2(individual)
    individual.rheobase=0
    individual.rheobase=k
    return individual


if __name__ == "__main__":
    iter_list=[ (i,j,k) for i in a for j in b for k in vr ]#for l in vpeak]

    guess_attrs=[]
    #find rheobase on a model constructed out of the mean parameter values.
    guess_attrs.append(np.mean( [ i for i in a ]))
    guess_attrs.append(np.mean( [ i for i in b ]))

    steps2 = np.linspace(50,190,4.0)
    steps = [ i*pq.pA for i in steps2 ]

    run_number,guess_value,attrs=main2(model,guess_attrs)
    #import copy
    from itertools import repeat
    list_of_models=list(futures.map(model2map,iter_list))
    #list_of_models=list(futures.map(model),iter_list

    iterator=list(futures.map(evaluate2,list_of_models,repeat(guess_value)))
    '''
    invalid_indvm=[]
    for i in iterator:
        if hasattr(i,'rheobase'):
            #vm=VirtualModel()
            vm.rheobase=i.rheobase
            guess_value=i.rheobase
        invalid_indvm.append(vm)
    bg_bf=time.time()
    '''
    score_matrix=list(futures.map(func2map,iterator,repeat(get_neab.suite)))
    with open('score_matrix.pickle', 'wb') as handle:
        pickle.dump(score_matrix, handle)
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



else:
   score_matrix=None
'''
