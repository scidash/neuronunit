
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

from neuronunit.models.reduced import ReducedModel

model = ReducedModel(get_neab.LEMS_MODEL_PATH,name='vanilla',backend='NEURON')

#from neuronunit.models.reduced import ReducedModel
#model = ReducedModel(get_neab.LEMS_MODEL_PATH,name='vanilla',backend='NEURON')
#model=model.load_model()
'''
def generate_prediction():

    prediction = {'value': None}
    model.rerun = True
    lookup = {} # A lookup table global to the function below.


    sub = np.array([x for x in lookup if lookup[x]==0])*units
    supra = np.array([x for x in lookup if lookup[x]>0])*units
    verbose=False
    if verbose:
        if len(sub):
            print("Highest subthreshold current is %s" \
                  % (float(sub.max().round(2))*units))
        else:
            print("No subthreshold current was tested.")
        if len(supra):
            print("Lowest suprathreshold current is %s" \
                  % supra.min().round(2))
        else:
            print("No suprathreshold current was tested.")

    if len(sub) and len(supra):
        #This means the smallest current to cause spiking.
        rheobase = supra.min()
    else:
        rheobase = None
    prediction['value'] = rheobase

    return prediction
'''
#evaluate once with a current injection at 0pA
'''
def f(ampl):
    lookup = {} # A lookup table global to the function below.

    if float(ampl) not in lookup:
        current = params.copy()['injected_square_current']

        uc={'amplitude':ampl}
        current.update(uc)
        current={'injected_square_current':current}

        model.inject_square_current(current)
        n_spikes = model.get_spike_count()
        verbose=True
        if verbose:
            print("Injected %s current and got %d spikes" % \
                    (ampl,n_spikes))
            '{}{}'.format('models name: ',model.name)
            print('models name: ',model.name)
            print(type(model))

        lookup[float(ampl)] = n_spikes
        if n_spikes==1:
            one=True
        else:
            one=False
    return (ampl,n_spikes,one,lookup)
'''


#from neuronunit.models import backends
AMPL = 0.0*pq.pA
DELAY = 100.0*pq.ms
DURATION = 1000.0*pq.ms


#def update_amplitude(test,tests):


'''
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
'''

def f(ampl,model):

    if float(ampl) not in model.lookup:
        current = params.copy()['injected_square_current']
        print('current, previous = ',ampl,model.previous)
        uc={'amplitude':ampl}
        current.update(uc)
        current={'injected_square_current':current}
        model.run_number+=1
        print('model run number',model.run_number)
        model.inject_square_current(current)
        model.previous=ampl
        n_spikes = model.get_spike_count()
        verbose=True
        if verbose:
            print("Injected %s current and got %d spikes" % \
                    (ampl,n_spikes))
        model.lookup[float(ampl)] = n_spikes
        return model
    if float(ampl) in model.lookup:
        print('model_in lookup')
        return model

rheobase=None
def main(iter_arg,model):
    attrs={}
    attrs['//izhikevich2007Cell']={}
    param=['a','b']
    i,j=iter_arg
    print(i,j)
    #model.name=str(i)+str(j)
    model.name=str(i)+str(j)#+str(k)+str(k)
    assert model.name != 'vanilla'
    #if model.name!='vanilla':
    print(type(model))
    model.name=str(i)+str(j)#+str(k)+str(k)
    assert model.name != 'vanilla'
    attrs['//izhikevich2007Cell']['vr']=i
    attrs['//izhikevich2007Cell']['vpeak']=j#40.0
    model.lookup={}
    model.run_number=0
    print(attrs)
    model.update_run_params(attrs)
    import pdb
    from itertools import repeat

    while_true=True
    while(while_true):
        from itertools import repeat

        if len(model.lookup)==0:
            steps2 = np.linspace(-50,150,8.0)
            steps = [ i*pq.pA for i in steps2 ]

        lookup2=list(futures.map(f,steps,repeat(model)))
        for m in lookup2:
            sub=[]
            supra=[]
            for k,v in m.lookup.items():
                #pdb.set_trace()
                if v==1:
                    while_true=False
                    print('hit')
                    #m.rheobas=k
                    return (k,m.attrs)
                    break
                elif v==0:
                    sub.append(k)
                elif v>0:
                    supra.append(k)
            sub=np.array(sub)
            supra=np.array(supra)
            if len(sub) and len(supra):
                steps2 = np.linspace(supra.min(),sub.max(),8.0)
                steps = [ i*pq.pA for i in steps2 ]

            elif len(sub):
                steps2 = np.linspace(sub.max(),2*sub.max(),8.0)
                steps = [ i*pq.pA for i in steps2 ]
            elif len(supra):
                steps2 = np.linspace(-2*(supra.min()),supra.min(),8.0)
                steps = [ i*pq.pA for i in steps2 ]
            #lookup2=list(futures.map(f,steps,repeat(m)))


from itertools import repeat
iter_list=[ (i,j) for i in a for j in b ]

if __name__ == "__main__":
    import time
    beggining=time.time()
    score_matrix=[]
    iter_list=[ (i,j) for i in vr for j in vpeak ]
    from itertools import repeat

    model=model.load_model()
    gen=futures.map(main,iter_list,repeat(model))
    #iter_models= [ m for m in models ]

    for g in gen:
        score_matrix.append((g))
    print(len(score_matrix))
    print(score_matrix)
    end=time.time()
    whole_time=end-beggining
    f=open('brute_force_time','w')
    f.write(str(whole_time))
    f.close()
    import pickle

    with open('score_matrix.pickle', 'wb') as handle:
        pickle.dump(score_matrix, handle)

    #Do the rheobase test. This is a serial bottle neck that must occur before any parallel optomization.
    #Its because the optimization routine must have apriori knowledge of what suprathreshold current injection values are for each model.
    #def main(func2map,iter_list,suite):
    #score_matrix = [func2map(i) for i in iter_arg]

    #Contact GitHub API Training Shop Blog About
