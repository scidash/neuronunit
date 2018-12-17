import os
import sciunit
import neuronunit
from neuronunit import aibs
import pickle
from neuronunit import tests as _, neuroelectro
#from neuronunit import tests as nu_tests, neuroelectro
from neuronunit.tests import passive, waveform, fi
from neuronunit.tests.fi import RheobaseTestP


import urllib.request, json

#print(obs_frame)
from neuronunit import neuroelectro

def neuroelectro_summary_observation(neuron,ontology):
    ephysprop_name = ''
    verbose = False

    reference_data = neuroelectro.NeuroElectroSummary(
        neuron = neuron, # Neuron type lookup using the NeuroLex ID.
        ephysprop = {'name': ontology['name']}, # Ephys property name in
        # NeuroElectro ontology.
    )
    reference_data.get_values() # Get and verify summary data
                                # from neuroelectro.org.
    return reference_data
import urllib.request, json

def get_obs(pipe):
    with urllib.request.urlopen("https://neuroelectro.org/api/1/e/") as url:
        ontologies = json.loads(url.read().decode())
        #print(ontologies)
    #with urllib.request.urlopen("https://neuroelectro.org/api/1/e/") as url:
    #    ontologies = json.loads(url.read().decode())
    obs = []
    for p in pipe:
        for l in ontologies['objects']:
            obs.append(neuroelectro_summary_observation(p,l))
    return obs

def update_amplitude(test,tests,score):
    rheobase = score.prediction['value']
    for i in [4,5,6]:
        tests[i].params['injected_square_current']['amplitude'] = rheobase*1.01 # I feel that 1.01 may lead to more than one spike
    return

def substitute_parallel_for_serial(electro_tests):
    for test,obs in electro_tests:
        test[0] = RheobaseTestP(obs['Rheobase'])

    return electro_tests

def substitute_criteria(observations_donar,observations_acceptor):
    # Inputs an observation donar
    # and an observation acceptor
    # Many neuroelectro data sources have std 0 values
    for index,oa in observations_acceptor.items():
        for k,v in oa.items():
            if k == 'std' and v == 0.0:
                oa[k] = observations_donar[index][k]
    return observations_acceptor

def substitute_parallel_for_serial(electro_tests):
    for test,obs in electro_tests:
        test[0] = RheobaseTestP(obs['Rheobase'])

    return electro_tests

def replace_zero_std(electro_tests):
    for test,obs in electro_tests:
        test[0] = RheobaseTestP(obs['Rheobase'])
        for k,v in obs.items():
            if v['std'] == 0:
                #print(electro_tests[1][1],obs)
                obs = substitute_criteria(electro_tests[1][1],obs)
                #print(obs)
    return electro_tests

def get_neuron_criteria(cell_id,file_name = None):#,observation = None):
    # Use neuroelectro experimental obsevations to find test
    # criterion that will be used to inform scientific unit testing.
    # some times observations are not sourced from neuroelectro,
    # but they are imputated or borrowed from other TestSuite
    # if that happens make test objections using observations external
    # to this method, and provided as a method argument.
    tests = []
    observations = None
    test_classes = None
    test_classes = [fi.RheobaseTest,
                     passive.InputResistanceTest,
                     passive.TimeConstantTest,
                     passive.CapacitanceTest,
                     passive.RestingPotentialTest,
                     waveform.InjectedCurrentAPWidthTest,
                     waveform.InjectedCurrentAPAmplitudeTest,
                     waveform.InjectedCurrentAPThresholdTest]#,
    observations = {}
    for index, t in enumerate(test_classes):
        #import pdb; pdb.set_trace()
        obs = t.neuroelectro_summary_observation(cell_id)

        if obs is not None:
            if 'mean' in obs.keys():
                tests.append(t(obs))
                observations[t.ephysprop_name] = obs

    #hooks = {tests[0]:{'f':update_amplitude}} #This is a trick to dynamically insert the method
    #update amplitude at the location in sciunit thats its passed to, without any loss of generality.
    suite = sciunit.TestSuite(tests,name="vm_suite")

    if file_name is not None:
        file_name = file_name +'.p'
        with open(file_name, 'wb') as f:
            pickle.dump(tests, f)

    return tests,observations

import copy
import sciunit
def get_tests():
    # get neuronunit tests
    # and select out Rheobase test and input resistance test
    # and less about electrophysiology of the membrane.
    # We are more interested in phenomonogical properties.
    electro_path = str(os.getcwd())+'/pipe_tests.p'
    assert os.path.isfile(electro_path) == True
    with open(electro_path,'rb') as f:
        electro_tests = pickle.load(f)
    electro_tests = replace_zero_std(electro_tests)
    electro_tests = substitute_parallel_for_serial(electro_tests)
    test, observation = electro_tests[0]
    tests = copy.copy(electro_tests[0][0])
    suite = sciunit.TestSuite(tests)
    #tests_ = tests[0:2]
    return tests, test, observation, suite
