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
#from neuronunit.models import LEMSModel

from neuronunit.models.reduced import ReducedModel
model = ReducedModel(get_neab.LEMS_MODEL_PATH,name='vanilla',backend='NEURON')

model=model.load_model()



from neuronunit import tests as nutests
#pdb.set_trace()


import sciunit.scores as scores
import neuronunit.capabilities as cap
class SanityTest():
#class RheobaseTest(TestPulseTest):
    def __init__(self):
        self=self
    """Tests the input resistance of a cell."""

    name = "Sanity test"

    description = ("Test for if injecting current results in not a numbers (NAN).")

    score_type = scores.ZScore


    required_capabilities = (cap.ReceivesSquareCurrent,
                             cap.ProducesSpikes)


    def generate_prediction(self,model):
        """
        Use inherited code
        Implementation of sciunit.Test.generate_prediction.
        mp and vm are different because they are outputs from different current injections.
        However they probably both should be the same current found through rheobase current.
        """
        model.inject_square_current(get_neab.suite.tests[4].params['injected_square_current'])
        mp=model.get_membrane_potential()

        i,vm = nutests.TestPulseTest.generate_prediction(nutests.TestPulseTest,model)
        median = model.get_median_vm() # Use median for robustness.
        std = model.get_std_vm()
        return vm, mp, median, std

    def compute_score(self,prediction):
        """Implementation of sciunit.Test.score_prediction."""
        import math
        (vm, mp, median, std) = prediction
        print(vm,mp,median,std)
        for j in vm:
            if math.isnan(j):
                return False
        for j in mp:
            if math.isnan(j):
                return False
        from neuronunit.capabilities import spike_functions
        spike_waveforms=spike_functions.get_spike_waveforms(vm)
        n_spikes = len(spike_waveforms)
        thresholds = []
        for i,s in enumerate(spike_waveforms):
            s = np.array(s)
            dvdt = np.diff(s)
            import math
            for j in dvdt:
                if math.isnan(j):
                    return False
        return True

def build_single(rh_value):
    #This method is only used to check singlular sets of hard coded parameters.]
    #This medthod is probably only useful for diagnostic purposes.
    import sciunit.scores as scores
    import quantities as qt
    attrs={}
    attrs['//izhikevich2007Cell']={}
    attrs['//izhikevich2007Cell']['a']=0.045
    attrs['//izhikevich2007Cell']['b']=-5e-09
    model.update_run_params(attrs)
    st=SanityTest()
    vm=st.generate_prediction(model)
    score=st.compute_score(vm)
    if score == True:
        get_neab.suite.tests[0].prediction={}
        get_neab.suite.tests[0].prediction['value']=rh_value*qt.pA
        score = get_neab.suite.judge(model)#passing in model, changes model
        return model
    else:
        return 10.0



def model2map(iter_arg):#This method must be pickle-able for scoop to work.
    vm=VirtualModel()
    attrs={}
    attrs['//izhikevich2007Cell']={}
    param=['a','b','vr','vpeak']
    i,j,k=iter_arg
    model.name=str(i)+str(j)#+str(k)+str(k)
    attrs['//izhikevich2007Cell']['a']=i
    attrs['//izhikevich2007Cell']['b']=j
    attrs['//izhikevich2007Cell']['vpeak']=k
    vm.attrs=attrs
    return vm

def func2map(iter_arg,suite):#This method must be pickle-able for scoop to work.
    '''
    Inputs an iterable list, a neuron unit test object suite of neuron model
    tests of emperical data reproducibility.
    '''
    print(iter_arg,suite)
    import pdb
    assert iter_arg.attrs!=None:
    model.update_run_params(iter_arg.attrs)
    import quantities as qt
    import os
    import os.path
    from scoop import utils
    score=None
    st=SanityTest()
    vm=st.generate_prediction(model)
    score=st.compute_score(vm)

    if score == True:
        get_neab.suite.tests[0].prediction={}
        get_neab.suite.tests[0].prediction['value']=suite*qt.pA
        model.inject_square_current(get_neab.suite.tests[4].params['injected_square_current'])
        mp=model.get_membrane_potential()
        import math
        for i in mp:
            if math.isnan(i):
                print(mp)
                error = scores.InsufficientDataScore(None)
                #pdb.set_trace()
                error = 10.0
                return (error,iter_arg.attrs)
        score = get_neab.suite.judge(model)#passing in model, changes model
        model.run_number+=1
        for i in score.unstack():
            if type(i.score)!=float:
                i.score=10.0
        error = [ float(i.score) for i in score.unstack() if i.score!=None ]


    elif score == False:
        import sciunit.scores as scores
        error = scores.InsufficientDataScore(None)
        error = 10.0
        #score = scores.ErrorScore(None)

        #error = sciunit.ErrorScore(None)
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
#verbose=True




def ff(ampl,vm):
    '''
    Inputs are an amplitude to test and a virtual model
    output is an virtual model with an updated dictionary.
    '''
    if float(ampl) not in vm.lookup:
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
        return vm

    if float(ampl) in vm.lookup:
        return vm



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
        assert(type(m))!=None

        sub=[]
        supra=[]
        import pdb
        assert(type(m.lookup))!=None
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
        import pdb
        assert(type(vm))!=None
        assert(type(vm.lookup))!=None
        #pdb.set_trace()
        for k,v in vm.lookup.items():
            if v==1:
                individual.rheobase=k
                return individual
            if v!=1:
                guess_value = None#more trial and error.
    if guess_value == None:
        (run_number,k,attrs)=main2(individual)
    individual.rheobase=k
    model.rheobase=k
    return model



if __name__ == "__main__":
    vr = np.linspace(-75.0,-50.0,3)
    a = np.linspace(0.015,0.045,3)
    b = np.linspace(-3.5*10E-9,-0.5*10E-9,3)
    k = np.linspace(7.0E-4-+7.0E-5,7.0E-4+70E-5,10)
    C = np.linspace(1.00000005E-4-1.00000005E-5,1.00000005E-4+1.00000005E-5,10)
    c = np.linspace(-55,-60,10)
    d = np.linspace(0.050,0.2,10)
    v0 = np.linspace(-75.0,-45.0,10)
    vt =  np.linspace(-50.0,-30.0,10)
    vpeak =np.linspace(30.0,40.0,10)
    #container=[]
    iter_list=[ (i,j,k) for i in a for j in b for k in vr ]#for l in vpeak]
    guess_attrs=[]

    guess_attrs.append(0.045)
    guess_attrs.append(-5e-09)
    steps2 = np.linspace(50,190,4.0)
    steps = [ i*pq.pA for i in steps2 ]

    run_number,rh_value,attrs=main2(model,guess_attrs)
    model=build_single(rh_value)

    #for x,y in enumerate(param):
    #    guess_attrs.append(np.mean( [ i[x] for i in pop ]))

    from itertools import repeat
    mean_vm=VirtualModel()

    guess_attrs=[]
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
    import copy



    steps2 = np.linspace(40,80,7.0)
    steps = [ i*pq.pA for i in steps2 ]



    #this might look like a big list iteration, but its not.
    #the statement below just finds rheobase on one value, that is the value
    #constituted by mean_vm. This will be used to speed up the rheobase search later.
    #model.attrs=mean_vm.attrs
    #def bulk_process(ff,steps,mean_vm):


    run_number,rh_value,attrs=main2(model,guess_attrs)

    from itertools import repeat
    list_of_models=list(futures.map(model2map,iter_list))
    for i in list_of_models:
        if type(i)==None:
            del i
        assert(type(i))!=None
    iterator=list(futures.map(evaluate2,list_of_models,repeat(rh_value)))
    iterator = [x for x in iterator if x.attrs != None]


    for i,j in enumerate(iterator):
        assert j.attrs!=None

    rhstorage = [  i.rheobase for i in iterator ]
    score_matrixt=list(futures.map(func2map,iterator,rhstorage))
    score_matrix=[]
    attrs=[]
    for i,j in score_matrixt:
        if InsufficientDataScore not in i:
            score_matrix.append(i)
            attrs.append(j)

    score_matrix=np.array(score_matrix)
    import pickle
    with open('score_matrix.pickle', 'wb') as handle:
        pickle.dump(score_matrixt, handle)

    storagei = [ np.sum(i) for i in score_matrix ]
    print(storagei)

    storagesmin=np.where(storagei==np.min(storagei))
    storagesmax=np.where(storagei==np.max(storagei))

    print(np.shape(storagesmin)[0])
    print(np.shape(storagesmax)[0])
    tuplepickle=(score_matrix[np.shape(storagesmin)[0]],score_matrix[np.shape(storagesmax)[0]],attrs[np.shape(storagesmax)[0]])
    with open('minumum_and_maximum_values.pickle', 'wb') as handle:
        pickle.dump(tuplepickle,handle)

    with open('minumum_and_maximum_values.pickle', 'rb') as handle:
        opt_values=pickle.load(handle)
        print('minumum and maximum')
        print(opt_values)
