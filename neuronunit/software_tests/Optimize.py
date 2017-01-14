
# coding: utf-8

import os,sys
import numpy as np
import matplotlib.pyplot as plt
import quantities as pq
import sciunit



import mpi4py
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
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

IZHIKEVICH_PATH = os.getcwd()+str('/NeuroML2') # Replace this the path to your
LEMS_MODEL_PATH = IZHIKEVICH_PATH+str('/LEMS_2007One.xml')


import time

from pyneuroml import pynml
'''
recorded_times=[]
before_pynml=time.time()
f=pynml.run_lems_with_jneuroml_neuron
DEFAULTS={}
DEFAULTS['v']=True
results=pynml.run_lems_with_jneuroml_neuron(LEMS_MODEL_PATH,
                  skip_run=False,
                  nogui=True,
                  load_saved_data=True,
                  only_generate_scripts = False,
                  plot=False,
                  show_plot_already=False,
                  exec_in_dir = ".",
                  verbose=DEFAULTS['v'],
exit_on_fail = True)
after_pynml=time.time()
delta_pynml=after_pynml-before_pynml
print('jneuroml time: ',after_pynml-before_pynml)
'''
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
    for i in [2,3,4]:
        # Set current injection to just suprathreshold
        tests[i].params['injected_square_current']['amplitude'] = rheobase*1.01


#Do the rheobase test. This is a serial bottle neck that must occur before any parallel optomization.
#Its because the optimization routine must have apriori knowledge of what suprathreshold current injection values are for each model.


hooks = {tests[0]:{'f':update_amplitude}} #This is a trick to dynamically insert the method
#update amplitude at the location in sciunit thats its passed to, without any loss of generality.
suite = sciunit.TestSuite("vm_suite",tests,hooks=hooks)

from neuronunit.models import backends
from neuronunit.models.reduced import ReducedModel
model = ReducedModel(LEMS_MODEL_PATH,name='vanilla',backend='NEURON')
model=model.load_model()
model.local_run()
'''
before_nrn=time.time()

#Its because Reduced model is the base class that calling super on SingleCellModel does not work.
model = ReducedModel(LEMS_MODEL_PATH,name='vanilla',backend='NEURON')
model=model.load_model()
model.local_run()
after_nrn=time.time()
times_nrn=after_nrn-before_nrn
print(times_nrn)
delta_pynml
delta_nrn_pynml=np.abs(delta_pynml-times_nrn)
print('the time difference is: \n')
print(delta_nrn_pynml)
print('press c to continue')
pdb.set_trace()
'''
from types import MethodType
def optimize(self,model,rov,param):
    best_params = None
    best_score = None
    from neuronunit.deapcontainer.deap_container2 import DeapContainer
    dc=DeapContainer()
    pop_size=12
    ngen=5

    #commited the change: pass in the model to deap, don't recreate it in every iteration just mutate the one existing model.
    #arguably recreating it would lead to less bugs however so maybe change back later.
    #check performance both ways to check for significant speed up without recreating the model object every iteration.
    pop = dc.sciunit_optimize_nsga(suite,model,pop_size,ngen,rov, param,
                                                         NDIM=3,OBJ_SIZE=6,seed_in=1)

    '''
    #NDIM is the number of parameters that are varied (dimensions of individual in deap). This is 1 (vr)
    #OBJ_SIZE is the number of error functions or objective functions that are explored this is 6. The elements of the objective functions are:
    RheobaseTest, InputResistanceTest, RestingPotentialTest,  InjectedCurrentAPWidthTest, InjectedCurrentAPAmplitudeTest, InjectedCurrentAPThresholdTest
    '''


    return pop
#toolbox = base.Toolbox()
#from scoop import futures
if __name__ == '__main__':
#toolbox.register("map", futures.map)
    my_test = tests[0]
    my_test.verbose = True
    my_test.optimize = MethodType(optimize, my_test) # Bind to the score.


    param=['vr','a','b']
    rov=[]
    rov0 = np.linspace(-65,-55,1000)
    rov1 = np.linspace(0.015,0.045,7)
    rov2 = np.linspace(-0.0010,-0.0035,7)
    rov.append(rov0)
    rov.append(rov1)
    rov.append(rov2)
    before_ga=time.time()
    pop = my_test.optimize(model,rov,param)
    after_ga=time.time()
    print('the time was:')
    delta=after_ga-before_ga
    print(delta)


    #This needs to act on error.
    #pdb.set_trace()
    if RANK==0:
        print("%.2f mV" % np.mean([p[0] for p in pop]))


        import matplotlib as matplotlib
        matplotlib.use('agg')
        import matplotlib.pyplot as plt
        plt.hold(True)
        for i in xrange(0,9):
            plt.plot(pop[i].time_trace,pop[i].voltage_trace)
        plt.savefig('best 10')
#NeuronObject=backends.NEURONBackend(LEMS_MODEL_PATH)
#NeuronObject.load_model()#Only needs to occur once
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
