


import mpi4py
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

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


from scoop import futures
#from neuronunit.models import backends
AMPL = 0.0*pq.pA
DELAY = 100.0*pq.ms
DURATION = 1000.0*pq.ms

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
'''
iter_list=[ (i,j,r,l,m,n,o,p,q) for i in a for j in b \
                                 for r in vr for l in k \
                                 for m in C for n in c \
                                 for o in d for p in v0 for q in vt ]
'''
iter_list=[ (i,j,k,q,r) for i in a for j in b for k in vr for q in vt for r in vpeak ]
#list_len=len(iter_list)
#iter_list=iter(iter_list)
import pdb


#class RheobaseTest(VmTest):

#Tests the full widths of APs at their half-maximum
#under current injection.



#required_capabilities = (cap.ReceivesSquareCurrent,
#                         cap.ProducesSpikes)
'''
params = {'injected_square_current':
            {'amplitude':100.0*pq.pA, 'delay':DELAY, 'duration':DURATION}}

name = "Rheobase test"

description = ("A test of the rheobase, i.e. the minimum injected current "
               "needed to evoke at least one spike.")

score_type = scores.RatioScore

def generate_prediction(model):
    #Implementation of sciunit.Test.generate_prediction.
    # Method implementation guaranteed by
    # ProducesActionPotentials capability.
    #from neuronunit.models.reduced import ReducedModel
    #model = ReducedModel(get_neab.LEMS_MODEL_PATH,name='vanilla',backend='NEURON')
    #model=model.load_model()
    prediction = {'value': None}
    model.rerun = True

    import quantities as pq
    units = pq.pA

    #pdb.set_trace()

    #units = observation['value'].units

    #this call is returning none.
    lookup = threshold_FI(model, units)

    sub = np.array([x for x in lookup if lookup[x]==0])*units
    supra = np.array([x for x in lookup if lookup[x]>0])*units
    verbose=True
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

def threshold_FI(model, units, guess=None):
    lookup = {} # A lookup table global to the function below.
    verbose=True
    def f(ampl):
        if float(ampl) not in lookup:
            current = params.copy()['injected_square_current']
            #This does not do what you would expect.
            #due to syntax I don't understand.
            #updating the dictionary keys with new values doesn't work.

            uc={'amplitude':ampl}
            current.update(uc)
            current={'injected_square_current':current}
            model.inject_square_current(current)
            n_spikes = model.get_spike_count()
            if verbose:
                print("Injected %s current and got %d spikes" % \
                        (ampl,n_spikes))
            lookup[float(ampl)] = n_spikes


    max_iters = 10

    #evaluate once with a current injection at 0pA
    def evaluate1(n,guess):
        print(n)
        #print(rc.ids)
        if n==0:
            f(0.0*units)

            return (None,None,lookup)
        if n==1:
            if guess is None:
                try:
                    guess = get_neab.observation['value']
                except KeyError:
                    guess = 100*pq.pA
            high = guess*2
            high = (50.0*pq.pA).rescale(units) if not high else high
            small = (1*pq.pA).rescale(units)
            #f(0.0*units) could happen in parallel with f(high) below
            f(high)
            import pdb

            return (small,high)

    def evaluate2(n,guess,small):
        #sub means below threshold, or no spikes
        #print(n)
        #print(rc.ids)
        sub = np.array([x for x in lookup if lookup[x]==0])*units
        #supra means above threshold, but possibly too high above threshold.
        supra = np.array([x for x in lookup if lookup[x]>0])*units

        if len(sub) and len(supra):
            f((supra.min() + sub.max())/2)

        elif len(sub):
            #the argument of f resolves to the maximum number of two in a list
            f(max(small,sub.max()*2))
        elif len(supra):
        #the argument of f resolves to the minimum number of two in a list
            f(min(-small,supra.min()*2))
        rlist=[small,sub,supra]
        return rlist

    import pdb
    #strategy get to work with serial map first.
    #run it in exhaustive search first such that calls to scoop are not nested yet.
    from itertools import repeat
    from scoop import futures
    returned_list = list(futures.map(evaluate1,[i for i in range(0,2)],repeat(guess)))
    small=returned_list[1]

    rlist = list(futures.map(evaluate2,[i for i in range(1,10)],repeat(guess),repeat(small)))
    small=rlist[0]
    sub=rlist[1]
    supra=rlist[2]
    return lookup

generate_prediction(model)
'''
#print('passes rheobase')
#pdb.set_trace()


def func2map(iter_arg):
    attrs={}
    attrs['//izhikevich2007Cell']={}
    param=['a','b','vr','vt','v_peak']
    #param=['a','b','vr','k','C','c','d','v0','vt']
    i,j,r,q,r=iter_arg

    #i,j,r,l,m,n,o,p,q=iter_list[iter_arg]
    model.name=str(i)+str(j)+str(k)+str(k)
    attrs['//izhikevich2007Cell']['a']=i
    attrs['//izhikevich2007Cell']['b']=j
    attrs['//izhikevich2007Cell']['vr']=r
    attrs['//izhikevich2007Cell']['vt']=q
    #Dangerous change
    #Attention change value back later!
    attrs['//izhikevich2007Cell']['vpeak']=40.0

    '''
    attrs['//izhikevich2007Cell']['k']=l
    attrs['//izhikevich2007Cell']['C']=m
    attrs['//izhikevich2007Cell']['c']=n
    attrs['//izhikevich2007Cell']['d']=o
    attrs['//izhikevich2007Cell']['v0']=p
    '''
    print('failed on attributes: \n')
    print(attrs)

    model.update_run_params(attrs)
    score = get_neab.suite.judge(model)#passing in model, changes model
    model.run_number+=1
    RUN_TIMES='{}{}{}'.format('counting simulation run times on models',model.results['run_number'],model.run_number)
    return score

iter_arg = [ iter_list[i] for i in range(RANK, len(iter_list), SIZE) ]
bg_bf=time.time()
score_matrix = [func2map(i) for i in iter_arg]

COMM.barrier()
score_matrix2 = COMM.gather(score_matrix, root=0)
if RANK == 0:
    score_matrix=[]
    for p in score_matrix:
        score_matrix.extend(p)
    print(score_matrix)
    end_bf=time.time()
    whole_time=end_bf-bg_bf
    with open('score_matrix.pickle', 'wb') as handle:
        pickle.dump(score_matrix, handle)

    f=open('brute_force_time','w')
    f.write(str(whole_time))
    f.close()
else:
   score_matrix=None
