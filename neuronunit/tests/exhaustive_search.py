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
#from neuronunit.models import LEMSModel

from neuronunit.models.reduced import ReducedModel
model = ReducedModel(get_neab.LEMS_MODEL_PATH,name='vanilla',backend='NEURON')

model=model.load_model()

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


def build_single():
    '''
    This method is only used to check singlular sets of hard coded parameters.
    '''
    attrs={}
    attrs['//izhikevich2007Cell']={}
    attrs['//izhikevich2007Cell']['a']=0.0303440140536
    attrs['//izhikevich2007Cell']['b']=-2.38706324769e-08
    attrs['//izhikevich2007Cell']['vpeak']=30.0
    attrs['//izhikevich2007Cell']['vr']=-53.4989145966
    import quantities as qt
    model.update_run_params(attrs)
    score = get_neab.suite.judge(model)#passing in model, changes model
    error = []
    error = [ abs(i.score) for i in score.unstack() ]
    

def model2map(iter_arg):#This method must be pickle-able for scoop to work.
    vm=VirtualModel()
    attrs={}
    attrs['//izhikevich2007Cell']={}
    param=['a','b']
    param=['a','b','vr','vpeak']
    i,j,k=iter_arg
    model.name=str(i)+str(j)#+str(k)+str(k)
    attrs['//izhikevich2007Cell']['a']=i
    attrs['//izhikevich2007Cell']['b']=j
    attrs['//izhikevich2007Cell']['vpeak']=k
    vm.attrs=attrs
    return vm


def func2map(iter_arg,suite):#This method must be pickle-able for scoop to work.
    model.update_run_params(iter_arg.attrs)
    import quantities as qt


    #iterator=list(futures.map(evaluate2,list_of_models,repeat(guess_value)))
    #evaluate2(iter_arg,repeat(guess_value))
    get_neab.suite.tests[0].prediction={}
    get_neab.suite.tests[0].prediction['value']=iter_arg.rheobase*qt.pA
    import os
    import os.path
    from scoop import utils
    score = get_neab.suite.judge(model)#passing in model, changes model
    model.run_number+=1
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

param=['a','b']
import neuronunit.capabilities as cap
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
    
    if float(ampl) not in vm.lookup:
        current = params.copy()['injected_square_current']
        uc={'amplitude':ampl}
        current.update(uc)
        current={'injected_square_current':current}
        vm.run_number+=1
        model.update_run_params(vm.attrs)
        assert vm.attrs==model.attrs
        model.inject_square_current(current)
        vm.previous=ampl
        n_spikes = model.get_spike_count()
        verbose=False
        if verbose:
            print("Injected %s current and got %d spikes" % \
                    (ampl,n_spikes))
        vm.lookup[float(ampl)] = n_spikes

        return vm

    if float(ampl) in vm.lookup:
        return vm

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

        m = lookup2[0]

        sub=[]
        supra=[]
        for k,v in m.lookup.items():
            if v==1:
                while_true=False
                end_time=time.time()
                total_time=end_time-begin_time
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
    model=VirtualModel()

    if guess_value != None:

        individual.lookup={}
        vm=VirtualModel()
        import copy
        vm.attrs=copy.copy(individual.attrs)
        vm=f(guess_value,vm)
        for k,v in vm.lookup.items():
            if v==1:
                individual.rheobase=k
                return individual
            if v!=1:
                guess_value = None#more trial and error.
    if guess_value == None:
        (run_number,k,attrs)=main2(individual)
    individual.rheobase=0
    individual.rheobase=k
    return individual


if __name__ == "__main__":
    iter_list=[ (i,j,k) for i in a for j in b for k in vr ]#for l in vpeak]
    guess_attrs=[]
    guess_attrs.append(np.mean( [ i for i in a ]))
    guess_attrs.append(np.mean( [ i for i in b ]))

    steps2 = np.linspace(50,190,4.0)
    steps = [ i*pq.pA for i in steps2 ]

    run_number,guess_value,attrs=main2(model,guess_attrs)
    from itertools import repeat
    list_of_models=list(futures.map(model2map,iter_list))

    iterator=list(futures.map(evaluate2,list_of_models,repeat(guess_value)))
    score_matrix=list(futures.map(func2map,iterator,repeat(get_neab.suite)))
    with open('score_matrix.pickle', 'wb') as handle:
        pickle.dump(score_matrix, handle)
