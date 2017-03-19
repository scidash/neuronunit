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
model = ReducedModel(get_neab.LEMS_MODEL_PATH,name='vanilla',backend='NEURON')
model=model.load_model()
from neuronunit import tests as nutests
import copy
from itertools import repeat
import sciunit.scores as scores
import neuronunit.capabilities as cap

def model2map(iter_arg):#This method must be pickle-able for scoop to work.
    vm=VirtualModel()
    attrs={}
    attrs['//izhikevich2007Cell']={}
    param=['a','b']#,'vr','vpeak']
    i,j=iter_arg#,k,l
    model.name=str(i)+str(j)#+str(k)+str(l)
    attrs['//izhikevich2007Cell']['a']=i
    attrs['//izhikevich2007Cell']['b']=j
    #attrs['//izhikevich2007Cell']['vr']=k
    #attrs['//izhikevich2007Cell']['vpeak']=l
    vm.attrs=attrs
    return vm

def func2map(iter_arg,value):#This method must be pickle-able for scoop to work.
    '''
    Inputs an iterable list, a neuron unit test object suite of neuron model
    tests of emperical data reproducibility.
    '''
    assert iter_arg.attrs!=None
    model.update_run_params(iter_arg.attrs)
    import quantities as qt
    import os
    import os.path
    from scoop import utils
    score = None
    sane = False
    #st=SanityTest()
    #vm=st.generate_prediction(model)
    #score=st.compute_score(vm)

    get_neab.suite.tests[4].params['injected_square_current']['amplitude'] = value*pq.pA*1.01
    sane = get_neab.suite.tests[4].sanity_check(rh_value=value*pq.pA*1.01)
    if sane == True:
        #get_neab.suite.tests[0].prediction = {}
        #get_neab.suite.tests[0].prediction['value'] = value*pq.pA
        #model.inject_square_current(get_neab.suite.tests[4].params['injected_square_current'])
        #mp = model.get_membrane_potential()
        score = get_neab.suite.judge(model)#passing in model, changes model
        model.run_number+=1
        for i in score.unstack():
            if type(i.score)!=float:
                i.score=10.0
        error = [ float(i.score) for i in score.unstack() if i.score!=None ]
    elif score == False:
        import sciunit.scores as scores
        error = scores.InsufficientDataScore(None)
    return (error,iter_arg.attrs)

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

param=['a','b']#,'vr','vpeak']
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


def check_fix_range(lookup):
    '''
    Inputs: lookup, A dictionary of previous current injection values
    used to search rheobase
    Outputs: A boolean to indicate if the correct rheobase current was found
    and a dictionary containing the range of values used.
    If rheobase was actually found then rather returning a boolean and a dictionary,
    instead logical True, and the rheobase current is returned.
    given a dictionary of rheobase search values, use that
    dictionary as input for a subsequent search.
    '''
    sub=[]
    supra=[]
    print(lookup)
    for k,v in lookup:
        if v==1:
            return (True,k)
        elif v==0:
            sub.append(k)
        elif v>0:
            supra.append(k)

    sub=np.array(sub)
    supra=np.array(supra)

    if len(sub) and len(supra):


        center = np.linspace(sub.max(),supra.min(),7.0)
        np.delete(center,np.array(lookup))
        #make sure that element 4 in a seven element vector
        #is exactly half way between sub.max() and supra.min()
        center[int(len(center)/2)+1]=(sub.max()+supra.min())/2.0
        steps = [ i*pq.pA for i in center ]

    elif len(sub):
        steps2 = np.linspace(sub.max(),2*sub.max(),7.0)
        np.delete(steps2,np.array(lookup))
        steps = [ i*pq.pA for i in steps2 ]

    elif len(supra):
        steps2 = np.linspace(-2*(supra.min()),supra.min(),7.0)
        np.delete(steps2,np.array(lookup))
        steps = [ i*pq.pA for i in steps2 ]


    return (False,steps)

def f(ampl,vm):
    '''
    Inputs are an amplitude to test and a virtual model
    output is an virtual model with an updated dictionary.
    '''
    #import pdb
    #pdb.set_trace()
    if float(ampl) not in vm.lookup or len(vm.lookup)==0:
        current = params.copy()['injected_square_current']
        uc={'amplitude':ampl}
        current.update(uc)
        current={'injected_square_current':current}
        vm.run_number+=1

        model.inject_square_current(current)
        vm.previous=ampl
        n_spikes = model.get_spike_count()
        if n_spikes==1:
            vm.rheobase=ampl
        verbose=False
        if verbose:
            print("Injected %s current and got %d spikes" % \
                    (ampl,n_spikes))
        vm.lookup[float(ampl)] = n_spikes
        #print(vm.lookup)
        return vm

    if float(ampl) in vm.lookup:
        return vm



def searcher(f,rh_param,vms,guess_value=None):
    '''
    ultimately an attempt to capture the essence a lot of repeatative code below.
    This is not yet used, but it is intended for future use.
    Its intended to replace the less general searcher function
    '''
    #rh_param[0]=False
    if guess_value!=None:
        rh_param=f(guess_value,vms)
    else:
        if rh_param[0]==True:
            print(rh_param)
            return rh_param[1]
        lookuplist=[]
        while rh_param[0]==False:
            if len(vms.lookup)==0:
                returned_list1 = list(futures.map(f,rh_param[1],repeat(vms)))
                #print(returned_list1)
                for vm in returned_list1:
                    for k,v in vm.lookup.items():
                        lookuplist.append((k,v))
            else:
                rh_param=check_fix_range(lookuplist)
                if rh_param[0]==True:
                    break
                    #print(rh_param)
                returned_list2 = list(futures.map(f,rh_param[1],repeat(vms)))
                for r in returned_list2:
                    for k,v in r.lookup.items():
                        lookuplist.append((k,v))

    print(rh_param)
    import pdb
    pdb.set_trace()
    return rh_param[1]


def evaluate(individual, guess_value=None):
    #This method must be pickle-able for scoop to work.
    model=VirtualModel()
    if guess_value != None:
        #individual.lookup={}

        import copy
        model.attrs=copy.copy(individual.attrs)
        #should there already exist a lookup table at this late stage in code?

        #steps = np.linspace(40,80,7.0)
        #steps_current = [ i*pq.pA for i in steps ]
        #model.attrs=vm.attrs
        rh_param=(False,guess_value)
        print(rh_param)
        #lookup=list(futures.map(ff,steps,repeat(mean_vm)))
        rheobase=searcher(f,rh_param,model,guess_value)
        print(type(rheobase))
    return rheobase



if __name__ == "__main__":

    #New file for model parameters.
    vr = np.linspace(-75.0,-50.0,2)
    a = np.linspace(0.015,0.045,2)
    b = np.linspace(-3.5*10E-9,-0.5*10E-9,2)
    k = np.linspace(7.0E-4-+7.0E-5,7.0E-4+70E-5,10)
    C = np.linspace(1.00000005E-4-1.00000005E-5,1.00000005E-4+1.00000005E-5,10)
    c = np.linspace(-55,-60,10)
    d = np.linspace(0.050,0.2,10)
    v0 = np.linspace(-75.0,-45.0,10)
    vt =  np.linspace(-50.0,-30.0,10)
    vpeak =np.linspace(30.0,40.0,10)

    iter_list=[ (i,j) for i in a for j in b  ]

    mean_vm=VirtualModel()
    guess_attrs=[]
    #find the mean parameter sets, and use them to inform the rheobase search.
    guess_attrs.append(np.mean( [ i for i in a ]))
    guess_attrs.append(np.mean( [ i for i in b ]))

    for i, p in enumerate(param):
        value=str(guess_attrs[i])
        model.name=str(model.name)+' '+str(p)+str(value)
        if i==0:
            attrs={'//izhikevich2007Cell':{p:value }}
        else:
            attrs['//izhikevich2007Cell'][p]=value
    mean_vm.attrs=attrs

    #this might look like a big list iteration, but its not.
    #the statement below just finds rheobase on one value, that is the value
    #constituted by mean_vm. This will be used to speed up the rheobase search later.
    #model.attrs=mean_vm.attrs
    #def bulk_process(ff,steps,mean_vm):
    #Attempts to parallelize rheobase search here should be based on those in nsga.py
    #_,rh_value,_=main(model,guess_attrs)

    steps = np.linspace(40,80,7.0)
    steps_current = [ i*pq.pA for i in steps ]
    model.attrs=mean_vm.attrs
    #lookup=list(futures.map(ff,steps,repeat(mean_vm)))
    #vm.rheobase=searcher(ff,steps_current,vm)
    rh_param=(False,steps_current)
    rh_value=searcher(f,rh_param,mean_vm)
    list_of_models=list(futures.map(model2map,iter_list))


    for i in list_of_models:
        if type(i)==None:
            del i
    rhstorage=list(futures.map(evaluate,list_of_models,repeat(rh_value)))
    #rhstorage = [  i.rheobase for i in iterator ]

    #iterator = [x for x in iterator if x.attrs != None]
    #rheobase = [x.rheobase for x in iterator]

    #for i,j in enumerate(iterator):
    #    assert j.attrs!=None
    print(rhstorage)
    score_matrixt=list(futures.map(func2map,iterator,rhstorage))
    score_matrix=[]
    attrs=[]
    score_typev=[]
    #below score is just the floats associated with RatioScore and Z-scores.
    for score,attr in score_matrixt:
    #for score_type,score,attr in score_matrixt:
        if not isinstance(score,scores.InsufficientDataScore):
            score_matrix.append(score)
            attrs.append(attr)
            #score_typev.append(score_type)
    score_matrix=np.array(score_matrix)

    import pickle
    with open('score_matrix.pickle', 'wb') as handle:
        pickle.dump(score_matrixt, handle)
    storagei = [ np.sum(i) for i in score_matrix ]
    storagesmin=np.where(storagei==np.min(storagei))
    storagesmax=np.where(storagei==np.max(storagei))

    #since there are non unique maximum and minimum values, just take the first ones of each.
    tuplepickle=(score_matrix[np.shape(storagesmin)[0]],score_matrix[np.shape(storagesmax)[0]],attrs[np.shape(storagesmax)[0]])
    with open('minumum_and_maximum_values.pickle', 'wb') as handle:
        pickle.dump(tuplepickle,handle)
    with open('minumum_and_maximum_values.pickle', 'rb') as handle:
        opt_values=pickle.load(handle)
        print('minumum value')
        print(opt_values)
