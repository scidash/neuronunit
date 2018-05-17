import os
import sciunit
import neuronunit
from neuronunit import aibs
import pickle
from neuronunit import tests as _, neuroelectro
#from neuronunit import tests as nu_tests, neuroelectro
from neuronunit.tests import passive, waveform, fi
from neuronunit.tests.fi import RheobaseTestP

def replace_zero_std(electro_tests):
    for test,obs in electro_tests:
        #test[0] = RheobaseTestP(obs['Rheobase'])
        for k,v in obs.items():
            if v['std'] == 0:
                print(electro_tests[1][1],obs)
                obs = get_neab.substitute_criteria(electro_tests[1][1],obs)
                print(obs)
    return electro_tests

def update_amplitude(test,tests,score):
    rheobase = score.prediction['value']
    for i in [4,5,6]:
        tests[i].params['injected_square_current']['amplitude'] = rheobase*1.01 # I feel that 1.01 may lead to more than one spike
    return

def substitute_criteria(observations_donar,observations_acceptor):
    # Inputs an observation donar
    # and an observation acceptor
    # Many neuroelectro data sources have std 0 values
    for index,oa in observations_acceptor.items():
        for k,v in oa.items():
            if k == 'std' and v == 0.0:
                oa[k] = observations_donar[index][k]
    return observations_acceptor

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
        obs = t.neuroelectro_summary_observation(cell_id)
        tests.append(t(obs))
        observations[t.ephysprop_name] = obs

    #hooks = {tests[0]:{'f':update_amplitude}} #This is a trick to dynamically insert the method
    #update amplitude at the location in sciunit thats its passed to, without any loss of generality.
    #suite = sciunit.TestSuite("vm_suite",tests)

    if file_name is not None:
        file_name = file_name +'.p'
        with open(file_name, 'wb') as f:
            pickle.dump(tests, f)

    return tests,observations
