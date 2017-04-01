
import pdb
from neuronunit.tests import get_neab
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

import neuronunit.capabilities as cap
#import neuronunit.capabilities.spike_functions as sf
#from neuronunit import neuroelectro
#from .channel import *

small=None
from scoop import futures
#from neuronunit.models import backends
AMPL = 0.0*pq.pA
DELAY = 100.0*pq.ms
DURATION = 1000.0*pq.ms



import pdb
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
from scoop import futures

import neuronunit.capabilities as cap
#import neuronunit.capabilities.spike_functions as sf
#from neuronunit import neuroelectro
#from .channel import *



vr = np.linspace(-75.0,-50.0,2)
a = np.linspace(0.015,0.045,2)
b = np.linspace(-3.5*10E-9,-0.5*10E-9,2)
k = np.linspace(7.0E-4-+7.0E-5,7.0E-4+70E-5,10)
C = np.linspace(1.00000005E-4-1.00000005E-5,1.00000005E-4+1.00000005E-5,10)

c = np.linspace(-55,-60,10)
d = np.linspace(0.050,0.2,10)
v0 = np.linspace(-75.0,-45.0,10)
vt =  np.linspace(-50.0,-30.0,10)
vpeak =np.linspace(30.0,40.0,2)
container=[]

#from neuronunit.models.reduced import ReducedModel
#model = ReducedModel(get_neab.LEMS_MODEL_PATH,name='vanilla',backend='NEURON')
#model=model.load_model()

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


from itertools import repeat


#from neuronunit.models import backends
AMPL = 0.0*pq.pA
DELAY = 100.0*pq.ms
DURATION = 1000.0*pq.ms

class VirtuaModel:
    def __init__(self):
        self.lookup={}
        self.previous=0
        self.run_number=0
        self.attrs=None


def f(ampl,vm,model):

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

        return vm

    if float(ampl) in vm.lookup:
        print('model_in lookup')
        return vm

#rheobase=None
def main(ind,model):
    '''
    attrs={}
    attrs['//izhikevich2007Cell']={}
    param=['a','b']
    i,j=iter_arg
    print(i,j)
    model.name=str(i)+str(j)#+str(k)+str(k)
    assert model.name != 'vanilla'
    print(type(model))
    model.name=str(i)+str(j)#+str(k)+str(k)
    assert model.name != 'vanilla'
    attrs['//izhikevich2007Cell']['vr']=i
    attrs['//izhikevich2007Cell']['vpeak']=j#40.0
    '''

    #import pdb
    #pdb.set_trace()
    print('individual model')
    print(ind,model)
    import pdb;
    #pdb.set_trace()
    #pdb.set_trace()
    #model=ind.model
    print(ind)
    #model.lookup={}
    #model.run_number=0
    #print(attrs)
    #model.update_run_params(attrs)
    #import pdb
    #from itertools import repeat
    vm=VirtuaModel()
    vm.attrs=model.attrs
    begin_time=time.time()
    #pdb.set_trace()
    while_true=True
    while(while_true):
        from itertools import repeat

        if len(vm.lookup)==0:
            steps2 = np.linspace(50,190,4.0)
            steps = [ i*pq.pA for i in steps2 ]
            lookup2=list(map(f,steps,repeat(vm),repeat(model)))
        #lookup2=list(map(f,steps,repeat(vm)))

        #steps3=[]
        #for m in lookup2:
        m = lookup2[0]
        sub=[]
        supra=[]
        for k,v in m.lookup.items():
            if v==1:
                while_true=False
                print('hit')
                #m.rheobas=k
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

        lookup2=list(map(f,steps,repeat(vm),repeat(model)))
        #steps3.extend(steps)
        #print('this is steps 3: \n\n\n')
        #print(steps3)
        #print('this was steps 3: \n\n\n')



    #return (vm.run_number,k,vm.attrs)

from itertools import repeat
iter_list=[ (i,j) for i in a for j in b ]

if __name__ == "__main__":
    import time
    beggining=time.time()
    score_matrix=[]
    iter_list=[ (i,j) for i in vr for j in vpeak ]

    #gen=list(map(main,iter_list))
    #iter_models= [ m for m in models ]
    gen = list(futures.map(main,iter_list))
    for g in gen:
        score_matrix.append((g))
    print(len(score_matrix))
    print(score_matrix)
    end=time.time()
    whole_time=end-beggining
    f=open('brute_force_time_parallel_inl','w')
    f.write(str(whole_time))
    f.close()
    import pickle

    with open('score_matrix_serial.pickle', 'wb') as handle:
        pickle.dump(score_matrix, handle)

    #Do the rheobase test. This is a serial bottle neck that must occur before any parallel optomization.
    #Its because the optimization routine must have apriori knowledge of what suprathreshold current injection values are for each model.
    #def main(func2map,iter_list,suite):
    #score_matrix = [func2map(i) for i in iter_arg]

    #Contact GitHub API Training Shop Blog About
