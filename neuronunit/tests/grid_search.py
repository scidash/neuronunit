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
model.load_model()
#model = ReducedModel(get_neab.LEMS_MODEL_PATH,name='vanilla',backend='NEURON')
model.rheobase_memory=None

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


from scoop import futures, shared
from neuronunit import tests as nutests
import copy
from itertools import repeat
import sciunit.scores as scores
import neuronunit.capabilities as cap

def model2map(iter_arg):#This method must be pickle-able for scoop to work.
    vm=VirtualModel()
    attrs={}
    attrs['//izhikevich2007Cell']={}
    param=['a','b','vr','vpeak']#,'vr','vpeak']
    i,j,k,l=iter_arg#,k,l
    model.name=str(i)+str(j)+str(k)+str(l)
    attrs['//izhikevich2007Cell']['a']=i
    attrs['//izhikevich2007Cell']['b']=j
    attrs['//izhikevich2007Cell']['vr']=k
    attrs['//izhikevich2007Cell']['vpeak']=l
    vm.attrs=attrs
    return vm


def pop2map(iter_arg):
    '''
    Just a sanity check an otherwise impotent method
    '''
    vm=VirtualModel()
    attrs={}
    attrs['//izhikevich2007Cell']={}
    param=['a','b','vr','vpeak']#,'vr','vpeak']
    i,j,k,l=iter_arg#,k,l
    model.name=str(i)+str(j)+str(k)+str(l)
    attrs['//izhikevich2007Cell']['a']=i
    attrs['//izhikevich2007Cell']['b']=j
    attrs['//izhikevich2007Cell']['vr']=k
    attrs['//izhikevich2007Cell']['vpeak']=l
    vm.attrs=attrs
    model.load_model()
    model.update_run_params(attrs)
    model.update_run_params(model.attrs)
    assert model.attrs==attrs==vm.attrs
    return (model, vm)

def func2map(iter_):#This method must be pickle-able for scoop to work.
    '''
    Inputs an iterable list, a neuron unit test object suite of neuron model
    tests of emperical data reproducibility.
    '''
    iter_arg,value=iter_
    assert iter_arg.attrs!=None
    model.load_model()
    import pdb
    model.update_run_params(iter_arg.attrs)
    model.update_run_params(model.attrs)
    assert model.attrs==iter_arg.attrs

    import quantities as qt
    import os
    import os.path
    from scoop import utils
    score = None
    sane = False

    sane = get_neab.suite.tests[3].sanity_check(value*pq.pA,model)

    uc = {'amplitude':value}
    current = params.copy()['injected_square_current']
    current.update(uc)
    current = {'injected_square_current':current}
    if len(model.attrs) == 0:
        model.update_run_params(vm.attrs)
    model.inject_square_current(current)
    n_spikes = model.get_spike_count()
    #print(n_spikes)
    assert n_spikes == 1
    if sane == True and n_spikes == 1:
        for i in [4,5,6]:
            get_neab.suite.tests[i].params['injected_square_current']['amplitude']=value*pq.pA
        get_neab.suite.tests[0].prediction={}
        score = get_neab.suite.tests[0].prediction['value']=value*pq.pA
        score = get_neab.suite.judge(model)#passing in model, changes model
        import neuronunit.capabilities as cap
        spikes_numbers=[]
        plt.clf()
        plt.hold(True)
        for k,v in score.related_data.items():
            spikes_numbers.append(cap.spike_functions.get_spike_train(((v.values[0]['vm']))))
            plt.plot(model.results['t'],v.values[0]['vm'])
        plt.savefig(str(model.name)+'.png')
        plt.clf()

        #n_spikes = model.get_spike_count()


        model.run_number+=1
        for i in score.sort_key.values[0]:
            if type(i)==None:
                i=10.0
        error= score.sort_key.values
    elif sane == False:
        import sciunit.scores as scores
        #error = scores.InsufficientDataScore(None)
        error = [ 10.0 for i in range(0,7) ]
    return (error,iter_arg.attrs,value*pq.pA)#,score)#.related_data.to_pickle.to_python())

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
        self.steps=None
        self.name=None
        self.s_html=None
        self.results=None
        self.error=None
def check_fix_range(vms):
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
    vms.rheobase=0.0
    for k,v in vms.lookup.items():
        if v==1:
            #A logical flag is returned to indicate that rheobase was found.
            vms.rheobase=float(k)
            vms.steps=0.0
            return (True,vms)
        elif v==0:
            sub.append(k)
        elif v>0:
            supra.append(k)

    sub=np.array(sub)
    supra=np.array(supra)

    if len(sub)!=0 and len(supra)!=0:
        #this assertion would only be wrong if there was a bug

        assert not sub.max()>supra.min()
            #import pdb; pdb.set_trace()

    if len(sub) and len(supra):
        everything=np.concatenate((sub,supra))

        center = np.linspace(sub.max(),supra.min(),7.0)
        np.delete(center,np.array(everything))
        #make sure that element 4 in a seven element vector
        #is exactly half way between sub.max() and supra.min()
        center[int(len(center)/2)+1]=(sub.max()+supra.min())/2.0
        steps = [ i*pq.pA for i in center ]

    elif len(sub):
        steps2 = np.linspace(sub.max(),2*sub.max(),7.0)
        np.delete(steps2,np.array(sub))
        steps = [ i*pq.pA for i in steps2 ]

    elif len(supra):
        steps2 = np.linspace(-2*(supra.min()),supra.min(),7.0)
        np.delete(steps2,np.array(supra))
        steps = [ i*pq.pA for i in steps2 ]

    vms.steps=steps
    vms.rheobase=None
    return (False,vms)

def check_current(ampl,vm):
    '''
    Inputs are an amplitude to test and a virtual model
    output is an virtual model with an updated dictionary.
    '''
    import copy
    if float(ampl) not in vm.lookup or len(vm.lookup)==0:
        current = params.copy()['injected_square_current']

        uc = {'amplitude':ampl}
        current.update(uc)
        current = {'injected_square_current':current}
        vm.run_number += 1
        model.re_init(vm.attrs)
        model.update_run_params(vm.attrs)
        model.attrs = vm.attrs
        if len(model.attrs) == 0:
            model.update_run_params(vm.attrs)
        model.inject_square_current(current)
        vm.previous=ampl
        n_spikes = model.get_spike_count()
        if n_spikes == 1:
            model.rheobase_memory=copy.copy(float(ampl))
            vm.rheobase=copy.copy(float(ampl))
            assert vm.rheobase != None
            assert model.rheobase_memory != None
        verbose = True
        if verbose:
            print('8 CPUs are testing different values of current injections simultaneously Injected %s current and got %d spikes' % \
                    (ampl,n_spikes))

        vm.lookup[float(ampl)] = n_spikes
        return vm
    if float(ampl) in vm.lookup:
        return vm



def searcher(f,rh_param,vms):
    '''
    inputs f a function to evaluate. rh_param a tuple with element 1 boolean, element 2 float or list
    and a  virtual model object.
    '''
    if rh_param[0]==True:
        return rh_param[1]
    lookuplist=[]
    cnt=0
    boolean=False
    from itertools import repeat
    while boolean==False and cnt<6:

        if len(model.attrs)==0:

            model.attrs = vms.attrs
            model.update_run_params(vms.attrs)

        if type(rh_param[1]) is float:
            #if its a single value educated guess
            #print('educated guess attempted')
            if model.rheobase_memory == None:
                #The educated guess, is the average of all the model parameters
                #with the latest model rheobase that was sampled.
                # using the name model.rheobase rheobase is deceptive, so I have
                # used the more accurate label rheobase_memory. This is not the actual found rheobase
                # that will get bound to vms its just a memory of the last tried value inside the constant variable model.
                # There is no use in using vms.rheobase for this, because this is a local value we are trying to find, not a global variable
                # that was already once found.
                model.rheobase_memory = rh_param[1]
                #model.rheobase_memory = (model.rheobase_memory + rh_param[1])/2.0
                #vms = check_current(rh_param[1],vms)



            vms = check_current(model.rheobase_memory,vms)
            model.re_init(vms.attrs)
            boolean,vms = check_fix_range(vms)
            #vms.rheobase_memory = vms.rheobase_memory

            if boolean:
                return vms
            else:
                #else search returned none type, effectively false
                rh_param = (None,None)

        elif len(vms.lookup)==0 and type(rh_param[1]) is list:
            #If the educated guess failed, or if the first attempt is parallel vector of samples
            returned_list=[]

            returned_list = list(futures.map(check_current,rh_param[1],repeat(vms)))
            #d={}
            assert vms!=None
            for v in returned_list:
                vms.lookup.update(v.lookup)
            boolean,vms=check_fix_range(vms)
            assert vms!=None
            if boolean:
                return vms

        else:
            #Finally if a parallel vector of samples failed zoom into the
            #smallest relevant interval and re-sample at a higher resolution

            returned_list=[]
            if type(vms.steps) is type(None):
                steps = np.linspace(50,150,7.0)
                steps_current = [ i*pq.pA for i in steps ]
                vms.steps = steps_current
                assert type(vms.steps) is not type(None)

            #rh_param=(False,steps_current)
            returned_list = list(futures.map(check_current,vms.steps,repeat(vms)))
            for v in returned_list:
                vms.lookup.update(v.lookup)
            boolean,vms=check_fix_range(vms)
            if boolean:
                return vms
        cnt+=1
    return vms

def evaluate(individual, guess_value=None):
    #This method must be pickle-able for scoop to work.
    #print(individual.attrs)
    vm = VirtualModel()
    import copy
    vm.attrs = copy.copy(individual.attrs)
    rh_param = (False,guess_value)
    vm = searcher(check_current,rh_param,vm)
    return vm
