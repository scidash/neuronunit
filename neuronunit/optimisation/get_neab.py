import os
import sciunit
import neuronunit
#from neuronunit import aibs
import pickle
from neuronunit import tests as _, neuroelectro
from neuronunit.tests import passive, waveform, fi
from neuronunit.tests.fi import RheobaseTestP
from neuronunit.tests import passive, waveform, druckman2013
from neuronunit.tests import druckman2013 as dm

import copy
import sciunit


import urllib.request, json

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

def get_obs(pipe):
    with urllib.request.urlopen("https://neuroelectro.org/api/1/e/") as url:
        ontologies = json.loads(url.read().decode())
        #print(ontologies)
    #with urllib.request.urlopen("https://neuroelectro.org/api/1/e/") as url:
    #    ontologies = json.loads(url.read().decode())
    obs = []
    for p in pipe:
        for l in ontologies['objects']:
            print(p,l)
            try:
                print('obs.append(neuroelectro_summary_observation(p,l))')
                obs.append(neuroelectro_summary_observation(p,l))
                print('worked')
            except:
                print('did not work')
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

def executable_druckman_tests(cell_id,file_name = None):
    # Use neuroelectro experimental obsevations to find test
    # criterion that will be used to inform scientific unit testing.
    # some times observations are not sourced from neuroelectro,
    # but they are imputated or borrowed from other TestSuite
    # if that happens make test objections using observations external
    # to this method, and provided as a method argument.
    tests = []
    observations = None
    test_classes = None

    dm.AP1AP1AHPDepthTest.ephysprop_name = None
    dm.AP1AP1AHPDepthTest.ephysprop_name = 'AHP amplitude'
    dm.AP2AP1AHPDepthTest.ephysprop_name = None
    dm.AP2AP1AHPDepthTest.ephysprop_name = 'AHP amplitude'
    dm.AP1AP1WidthHalfHeightTest.ephysprop_name = None
    dm.AP1AP1WidthHalfHeightTest.ephysprop_name = 'spike half-width'
    dm.AP1AP1WidthPeakToTroughTest.ephysprop_name = None
    dm.AP1AP1WidthPeakToTroughTest.ephysprop_name = 'spike width'
    #dm.IinitialAccomodationMeanTest.ephysprop_name = None
    #dm.IinitialAccomodationMeanTest.ephysprop_name = 'adaptation_percent'

    test_classes = [fi.RheobaseTest, \
    dm.AP1AP1AHPDepthTest, \
    dm.AP2AP1AHPDepthTest,\
    dm.AP1AP1WidthHalfHeightTest,\
    dm.AP1AP1WidthPeakToTroughTest,\
    ]
    observations = {}
    for index, t in enumerate(test_classes):
        obs = t.neuroelectro_summary_observation(cell_id)

        if obs is not None:
            if 'mean' in obs.keys():
                tests.append(t(obs))
                observations[t.ephysprop_name] = obs

    suite = sciunit.TestSuite(tests,name="vm_suite")

    if file_name is not None:
        file_name = file_name +'.p'
        with open(file_name, 'wb') as f:
            pickle.dump(tests, f)

    return tests,observations

def executable_tests(cell_id,file_name = None):#,observation = None):
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

def get_common_criteria():
    purkinje ={"id": 18, "name": "Cerebellum Purkinje cell", "neuron_db_id": 271, "nlex_id": "sao471801888"}
    #fi_basket = {"id": 65, "name": "Dentate gyrus basket cell", "neuron_db_id": None, "nlex_id": "nlx_cell_100201"}
    pvis_cortex = {"id": 111, "name": "Neocortex pyramidal cell layer 5-6", "neuron_db_id": 265, "nlex_id": "nifext_50"}
    #This olfactory mitral cell does not have datum about rheobase, current injection values.
    olf_mitral = {"id": 129, "name": "Olfactory bulb (main) mitral cell", "neuron_db_id": 267, "nlex_id": "nlx_anat_100201"}
    ca1_pyr = {"id": 85, "name": "Hippocampus CA1 pyramidal cell", "neuron_db_id": 258, "nlex_id": "sao830368389"}
    pipe = [ olf_mitral, ca1_pyr, purkinje,  pvis_cortex]
    electro_tests = []
    obs_frame = {}
    test_frame = {}
    import neuronunit
    anchor = neuronunit.__file__
    anchor = os.path.dirname(anchor)
    mypath = os.path.join(os.sep,anchor,'optimisation/all_tests.p')

    try:

        electro_path = str(os.getcwd())+'all_tests.p'

        assert os.path.isfile(electro_path) == True
        with open(electro_path,'rb') as f:
            (obs_frame,test_frame) = pickle.load(f)

    except:
        for p in pipe:
            print(p)
            p_tests, p_observations = get_obs(p)

            obs_frame[p["name"]] = p_observations#, p_tests))
            test_frame[p["name"]] = p_tests#, p_tests))
        electro_path = str(os.getcwd())+'all_tests.p'
    return (obs_frame,test_frame)
        #with open(electro_path,'wb') as f:
        #    pickle.dump((obs_frame,test_frame),f)




def get_tests(backend=str("RAW")):
    import neuronunit
    anchor = neuronunit.__file__
    anchor = os.path.dirname(anchor)
    mypath = os.path.join(os.sep,anchor,'unit_test/pipe_tests.p')

    # get neuronunit tests
    # and select out Rheobase test and input resistance test
    # and less about electrophysiology of the membrane.
    # We are more interested in phenomonogical properties.
    electro_path = mypath
    #str(os.getcwd())+'/pipe_tests.p'
    assert os.path.isfile(electro_path) == True
    with open(electro_path,'rb') as f:
        electro_tests = pickle.load(f)
    obs_frame,electro_tests = get_common_criteria()
    electro_tests = replace_zero_std(electro_tests)
    if backend in str('ADEXP'):
        electro_tests = substitute_parallel_for_serial(electro_tests)
    
    test, observation = electro_tests[0]
    tests = copy.copy(electro_tests[0][0])
    suite = sciunit.TestSuite(tests)
    #tests_ = tests[0:2]
    return tests, test, observation, suite
