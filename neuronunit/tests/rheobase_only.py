

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
from scoop import futures

import neuronunit.capabilities as cap
#import neuronunit.capabilities.spike_functions as sf
#from neuronunit import neuroelectro
#from .channel import *



vr = np.linspace(-75.0,-50.0,2)
a = np.linspace(0.015,0.045,10)
b = np.linspace(-3.5*10E-9,-0.5*10E-9,1)
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
verbose=False


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
#evaluate once with a current injection at 0pA
past=[]
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
    print('returned')
    return model




#from neuronunit.models import backends
AMPL = 0.0*pq.pA
DELAY = 100.0*pq.ms
DURATION = 1000.0*pq.ms


rheobase=None



def g(m):
    while_true=True
    steps2 = np.linspace(-100,200,8.0)
    steps = [ i*pq.pA for i in steps2 ]

    while(while_true):
        #for m in models:
        for k,v in m.lookup.items():
            ample=float(k)
            n_spikes=v
            if v==1:
                print('hit rheobase')
                while_true=False
                m.rheobase=ample
                return (m.rheobase,m.attrs)
            else:
                #if not len(sub) or len(supra):

                sub = np.array([x for x in m.lookup if m.lookup[x]==0])#*units
                supra = np.array([x for x in m.lookup if m.lookup[x]>0])#*units

                step_size=abs(float(steps[1])-float(steps[0]))


                if len(sub) and len(supra):
                    new_search_middle=(supra.min() + sub.max())/2.0
                    steps2 = np.linspace(new_search_middle-(step_size/2.0),new_search_middle+(step_size/2.0),8.0)
                    steps = [ i*pq.pA for i in steps2 ]

                elif len(sub):
                    new_search_middle=(sub.max())
                    steps2 = np.linspace(new_search_middle,new_search_middle+(step_size),8.0)
                    steps = [ i*pq.pA for i in steps2 ]

                elif len(supra):
                    new_search_middle=(supra.min())
                    steps2 = np.linspace(new_search_middle-(step_size),new_search_middle,6.0)
                    steps = [ i*pq.pA for i in steps2 ]

                model=f(ample,m)

    #return m
def main(iter_arg,model):
    attrs={}
    attrs['//izhikevich2007Cell']={}
    param=['vr','vpeak']
    i,j=iter_arg
    print(i,j)
    #model.name=str(i)+str(j)
    model.name=str(i)+str(j)#+str(k)+str(k)
    assert model.name != 'vanilla'
    attrs['//izhikevich2007Cell']['vr']=i
    attrs['//izhikevich2007Cell']['vpeak']=j#40.0
    model.update_run_params(attrs)
    model.rheobase=None
    model.lookup={}
    import pdb
    from itertools import repeat
    guess = float(get_neab.observation['value'])
    past=[]
    model=f(guess,model)
    model=g(model)

    #models=list(futures.map(steps,g))

    #pdb.set_trace()
    for k,v in model.lookup.items():
        ample=float(k)
        n_spikes=v


    #print(steps)
    from itertools import repeat
    for k,v in model.lookup.items():
        ample=float(k)
        n_spikes=v

    model=f(ample,model)

        #model.lookup.
    #models=list(futures.map(f,repeat(ample),repeat(model)))
    #print(len(models))
    #print('got here')

    #models=list(futures.map(g,repeat(model))

    #iter_models= [ m for m in models ]
    #pdb.set_trace()





if __name__ == "__main__":
    import time
    beggining=time.time()
    score_matrix=[]
    iter_list=[ (i,j) for i in vr for j in vpeak ]
    from itertools import repeat

    from neuronunit.models.reduced import ReducedModel

    model = ReducedModel(get_neab.LEMS_MODEL_PATH,name='vanilla',backend='NEURON')

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


    #Do the rheobase test. This is a serial bottle neck that must occur before any parallel optomization.
    #Its because the optimization routine must have apriori knowledge of what suprathreshold current injection values are for each model.
    #def main(func2map,iter_list,suite):
    #score_matrix = [func2map(i) for i in iter_arg]
