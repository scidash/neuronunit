
# coding: utf-8

# In[17]:
#Retain jupyter notebook cell #s and comments, such that the Notebook can quickly be remade with a conversion.
#get_ipython().magic('load_ext autoreload')
#get_ipython().magic('autoreload 2')
#get_ipython().magic('matplotlib notebook')
import os,sys
import numpy as np
import matplotlib.pyplot as plt
import quantities as pq


import sciunit


#Over ride any neuron units in the PYTHON_PATH with this one.
#only appropriate for development.
thisnu = str(os.getcwd())+'/../..'
sys.path.insert(0,thisnu)
print(sys.path)
    
import neuronunit
from neuronunit import aibs
from neuronunit.models.reduced import ReducedModel
import pdb
import pickle
# In[16]:

# This example is from https://github.com/OpenSourceBrain/IzhikevichModel.
#model_path = os.getcwd()+str('/neuronunit/software_tests/NeuroML2') # Replace this the path to your 
                                                                       # working copy of 
                                                                       # github.com/OpenSourceBrain/IzhikevichModel.  
#file_path=model_path+str('/LEMS_2007One.xml')

IZHIKEVICH_PATH = os.getcwd()+str('/NeuroML2') # Replace this the path to your
LEMS_MODEL_PATH = IZHIKEVICH_PATH+str('/LEMS_2007One.xml')


# In[8]:

import quantities as pq
from neuronunit import tests as nu_tests, neuroelectro
neuron = {'nlex_id': 'nifext_50'} # Layer V pyramidal cell
tests = []

dataset_id = 354190013  # Internal ID that AIBS uses for a particular Scnn1a-Tg2-Cre 
                        # Primary visual area, layer 5 neuron.
observation = aibs.get_observation(dataset_id,'rheobase')


if os.path.exists(str(os.getcwd())+"/neuroelectro.pickle"):
    print('attempting to recover from pickled file')
    with open('neuroelectro.pickle', 'rb') as handle:
        tests = pickle.load(handle)

else:
    print('checked path:')
    print(str(os.getcwd())+"/neuroelectro.pickle")
    print('no pickled file found. Commencing time intensive Download')
    
    #(nu_tests.TimeConstantTest,None),                           (nu_tests.InjectedCurrentAPAmplitudeTest,None),
    tests += [nu_tests.RheobaseTest(observation=observation)]
    test_class_params = [(nu_tests.InputResistanceTest,None),
                         (nu_tests.RestingPotentialTest,None),
                         (nu_tests.InjectedCurrentAPWidthTest,None),
                         (nu_tests.InjectedCurrentAPThresholdTest,None)]
                         


    for cls,params in test_class_params:
        #use of the variable 'neuron' in this conext conflicts with the module name 'neuron'
        #at the moment it doesn't seem to matter as neuron is encapsulated in a class, but this could cause problems in the future.
        
        
        observation = cls.neuroelectro_summary_observation(neuron)
        tests += [cls(observation,params=params)]

    with open('neuroelectro.pickle', 'wb') as handle:
        pickle.dump(tests, handle) 
    
def update_amplitude(test,tests,score):
    rheobase = score.prediction['value']#first find a value for rheobase
    #then proceed with other optimizing other parameters.
    

    print(len(tests))
    #pdb.set_trace()
    for i in [3,4,5]:
        # Set current injection to just suprathreshold
        tests[i].params['injected_square_current']['amplitude'] = rheobase*1.01 


#Do the rheobase test. This is a serial bottle neck that must occur before any parallel optomization.
#Its because the optimization routine must have apriori knowledge of what suprathreshold current injection values are for each model.


hooks = {tests[0]:{'f':update_amplitude}} #This is a trick to dynamically insert the method
#update amplitude at the location in sciunit thats its passed to, without any loss of generality.
suite = sciunit.TestSuite("vm_suite",tests,hooks=hooks)


# In[4]:

# In[5]:

#sciunit opt as a repository is now obsolete.


# In[6]:

from neuronunit.models import backends
from neuronunit.models.reduced import ReducedModel

model = ReducedModel(LEMS_MODEL_PATH,name='vanilla',backend='NEURON')
model=model.load_model()



from types import MethodType
def optimize(self,model,rov,param):
    best_params = None
    best_score = None
    from neuronunit.deapcontainer.deap_container import DeapContainer
    dc=DeapContainer()
    pop_size=12
    ngen=5
                
    #commited the change: pass in the model to deap, don't recreate it in every iteration just mutate the one existing model.
    #arguably recreating it would lead to less bugs however so maybe change back later.                      
    #check performance both ways to check for significant speed up without recreating the model object every iteration.
    pop = dc.sciunit_optimize(suite,model,pop_size,ngen,rov, param,
                                                         NDIM=2,OBJ_SIZE=2,seed_in=1)
                                                         
      
    return pop#(best_params, best_score, model)

my_test = tests[0]
my_test.verbose = True
my_test.optimize = MethodType(optimize, my_test) # Bind to the score.


# In[7]:
param='vr'
rov = np.linspace(-67,-50,1000)
pop = my_test.optimize(model,rov,param)

# In[13]:

print("%.2f mV" % np.mean([p[0] for p in pop]))


NeuronObject=backends.NEURONBackend(LEMS_MODEL_PATH)
NeuronObject.load_model()#Only needs to occur once
#NeuronObject.update_nrn_param(param_dict)
#NeuronObject.update_inject_current(stim_dict)
'''
TODO: change such that it execs simulations.
Note this is not actually running any simulations. 
Its just initialising them.
brute force optimization:
for comparison
#print(dir(NeuronObject))
for vr in iter(np.linspace(-75,-50,6)):
    for i,a in iter(enumerate(np.linspace(0.015,0.045,7))):
        for j,b in iter(enumerate(np.linspace(-3.5,-0.5,7))):
            for k in iter(np.linspace(100,200,4)):
                param_dict={}#Very important to redeclare dictionary or badness.
                param_dict['vr']=vr

                param_dict['a']=str(a) 
                param_dict['b']=str(b)               
                param_dict['C']=str(150)
                param_dict['k']=str(0.70) 
                param_dict['vpeak']=str(45)                      
                             
                NeuronObject.update_nrn_param(param_dict)
                stim_dict={}
                stim_dict['delay']=200
                stim_dict['duration']=500
                stim_dict['amplitude']=k#*100+150

                NeuronObject.update_inject_current(stim_dict)
                NeuronObject.local_run()
                vm,im,time=NeuronObject.out_to_neo()
                print('\n')
                print('\n')
                print(vm.trace)
                print(time.trace)
                print(im.trace)
                print('\n')
                print('\n')
'''
