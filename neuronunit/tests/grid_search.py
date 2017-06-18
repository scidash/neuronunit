import numpy as np
import time
#import inspect
#from types import MethodType
import quantities as pq
#from quantities.quantity import Quantity
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
#model.rheobase_memory=None

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
    vm.attrs={}

    vm.attrs['a']=iter_arg[0]
    vm.attrs['b']=iter_arg[1]
    vm.attrs['vr']=iter_arg[2]
    vm.attrs['vpeak']=iter_arg[3]

    #attrs['//izhikevich2007Cell']['b']=j
    #attrs['//izhikevich2007Cell']['vr']=k
    #attrs['//izhikevich2007Cell']['vpeak']=l

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
    #i=iter_arg
    model.name=str(i)+str(j)+str(k)+str(l)
    attrs['//izhikevich2007Cell']['a']=i
    attrs['//izhikevich2007Cell']['b']=j
    attrs['//izhikevich2007Cell']['vr']=k
    attrs['//izhikevich2007Cell']['vpeak']=l
    vm.attrs=attrs
    model.load_model()
    model.update_run_params(vm.attrs)
    #model.update_run_params(model.attrs)
    print(model.params,attrs,vm.attrs)
    return (model, vm)

def func2map(iter_):#This method must be pickle-able for scoop to work.
    '''
    Inputs an iterable list, a neuron unit test object suite of neuron model
    tests of emperical data reproducibility.
    '''
    iter_arg,value = iter_

    assert iter_arg.attrs is not type(None)
    return_list=[]
    # model.load_model()
    model.update_run_params(iter_arg.attrs)
    import quantities as qt
    import copy
    import os
    import os.path
    import pdb
    score = None
    sane = False
    #if value<0:
        #break

    if type(value) is not type(None) and value > 0:
        assert value > 0
        sane = get_neab.suite.tests[3].sanity_check(value*pq.pA,model)
        uc = {'amplitude':value}
        current = params.copy()['injected_square_current']
        current.update(uc)
        current = {'injected_square_current':current}
        import copy
        model.inject_square_current(current)
        init_vm = copy.copy(model.results['vm'])
        n_spikes = model.get_spike_count()
        assert n_spikes == 1 or n_spikes == 0
        get_neab.suite.tests[0].prediction={}
        get_neab.suite.tests[0].prediction['value'] = value * pq.pA
        #print()
        return_list = []
        error = []# re-declare error this way to stop it growing.
        if sane == True and n_spikes == 1:
            #we are not using the rheobase test from the test suite, we are using a custom parallel rheobase test
            #instead.
            #del get_neab.suite.tests[0]
            for i in [3,4,5]:

                get_neab.suite.tests[i].params['injected_square_current']['amplitude']=value*pq.pA

            print(get_neab.suite.tests[0].prediction['value'],value)

            import os
            import scoop
            import pickle
            score = get_neab.suite.judge(model)#passing in model, changes model

            import neuronunit.capabilities as cap
            spikes_numbers=[]
            plt.clf()
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
            error= [np.abs(i) for i in score.sort_key.values[0]]
            #pdb.set_trace()

            return_list.append(np.sum(error))
            return_list.append(iter_arg.attrs)
            return_list.append(value*pq.pA)
            return_list.append(model.results['t'])
            return_list.append(model.results['vm'])
            return_list.append(error)
            return_list.append(init_vm)

            print(len(error))
        elif sane == False:
            import sciunit.scores as scores
            error = [ 10.0 for i in range(0,7) ]
            import copy
            print(len(error))
            print(score.sort_key.values[0])
            #pdb.set_trace()

            return_list.append(np.sum(error))
            return_list.append(iter_arg.attrs)

            return_list.append(value*pq.pA)
            return_list.append(model.results['t'])
            return_list.append(model.results['vm'])
            return_list.append(error)
            return_list.append(init_vm)

            print(len(returned_list[0]))
            print(len(returned_list[1]))
            print(len(returned_list[2]))
            print(len(returned_list[3]))

    return return_list
    #return list(zip(np.sum(error), iter_arg.attrs, value*pq.pA, model.results['t'], model.results['vm']))

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
        print(str(bool(sub.max()>supra.min())))
        assert not sub.max()>supra.min()
            #import pdb; pdb.set_trace()

    if len(sub) and len(supra):
        everything=np.concatenate((sub,supra))

        center = np.linspace(sub.max(),supra.min(),7.0)
        centerl = list(center)
        for i,j in enumerate(centerl):
            if i in list(everything):
                np.delete(center,i)
                del centerl[i]
                '{}'.format(i)
        #print(i,j,'stuck in a loop')
        #delete the index
        #np.delete(center,np.where(everything is in center))
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
    import scoop
    import pickle
    #from scoop import _debug
    print(scoop.utils.getWorkerQte(scoop.utils.getHosts()))
    ##dv = _debug.getDebugIdentifier()
    #print(dv)

    if float(ampl) not in vm.lookup or len(vm.lookup)==0:

        '''
        filename = '{}{}'.format(str(scoop.utils.getWorkerQte(scoop.utils.getHosts())),'test_current_failed_attrs.pickle')
        with open(filename, 'wb') as handle:
            #scoop.utils.getWorkerQte(scoop.utils.getHosts())
            failed_attrs=(ampl,vm.attrs,scoop.utils.socket.gethostname(),scoop.utils.getWorkerQte(scoop.utils.getHosts()))
            pickle.dump(failed_attrs, handle)
        '''
        current = params.copy()['injected_square_current']

        uc = {'amplitude':ampl}
        current.update(uc)
        current = {'injected_square_current':current}
        vm.run_number += 1
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
            print('8 CPUs are testing different values of current injections simultaneously Injected %s current and got %d spikes on model %s' % \
                    (ampl,n_spikes,vm.attrs))

        vm.lookup[float(ampl)] = n_spikes
        return vm
    if float(ampl) in vm.lookup:
        return vm



def searcher(rh_param,vms):
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
    while boolean == False and cnt < 6:
        if len(model.params)==0:
            model.update_run_params(vms.attrs)
        if type(rh_param[1]) is float:
            #if its a single value educated guess
            if model.rheobase_memory == None:
                model.rheobase_memory = rh_param[1]
            vms = check_current(model.rheobase_memory,vms)
            model.update_run_params(vms.attrs)
            boolean,vms = check_fix_range(vms)
            if boolean:
                return vms
            else:
                #else search returned none type, effectively false
                rh_param = (None,None)

        elif len(vms.lookup)==0 and type(rh_param[1]) is list:
            #If the educated guess failed, or if the first attempt is parallel vector of samples
            returned_list = list(futures.map(check_current,rh_param[1],repeat(vms)))
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
    vm = VirtualModel()
    import copy
    vm.attrs = copy.copy(individual.attrs)
    rh_param = (False,guess_value)
    import scoop
    import pickle
    with open('pre_failed_attrs.pickle', 'wb') as handle:
        scoop.utils.getWorkerQte(scoop.utils.getHosts())
        failed_attrs=(vm.attrs,scoop.utils.socket.gethostname(),scoop.utils.getWorkerQte(scoop.utils.getHosts()))
        pickle.dump(failed_attrs, handle)

    vm = searcher(rh_param,vm)


    return vm
