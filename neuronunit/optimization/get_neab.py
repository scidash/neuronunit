import os
import sciunit
import neuronunit
from neuronunit import aibs
import pickle
from neuronunit import tests as _, neuroelectro
#from neuronunit import tests as nu_tests, neuroelectro
from neuronunit.tests import passive, waveform, fi

def update_amplitude(test,tests,score):
    rheobase = score.prediction['value']
    for i in [4,5,6]:
        tests[i].params['injected_square_current']['amplitude'] = rheobase*1.01 # I feel that 1.01 may lead to more than one spike
    return

def impute_criteria(observations_donar,observations_acceptor):
    #
    #

    for index,oa in observations_acceptor.items():
        for k,v in oa.items():
            if k == 'std' and v == 0.0:
                oa[k] = observations_donar[index][k]
    return observations_acceptor


def get_neuron_criteria(cell_id,file_name = None,observation = None):
    # Use neuroelectro experimental obsevations to find test
    # criterion that will be used to inform scientific unit testing.
    # some times observations are not sourced from neuroelectro,
    # but they are imputated or borrowed from other TestSuite
    # if that happens make test objections using observations external
    # to this method, and provided as a method argument.
    tests = []
    observations = {}
    test_classes = [fi.RheobaseTestP,
                     passive.InputResistanceTest,
                     passive.TimeConstantTest,
                     passive.CapacitanceTest,
                     passive.RestingPotentialTest,
                     waveform.InjectedCurrentAPWidthTest,
                     waveform.InjectedCurrentAPAmplitudeTest,
                     waveform.InjectedCurrentAPThresholdTest]#,
    if observation is not None:
       for index, t in enumerate(test_classes):
           obs = observation[t.ephysprop_name]
           tests.append(t(obs))
           observations[t.ephysprop_name] = obs
    else:
        for index, t in enumerate(test_classes):
            obs = t.neuroelectro_summary_observation(cell_id)
            tests.append(t(obs))
            observations[t.ephysprop_name] = obs

    hooks = {tests[0]:{'f':update_amplitude}} #This is a trick to dynamically insert the method
    #update amplitude at the location in sciunit thats its passed to, without any loss of generality.
    #suite = sciunit.TestSuite("vm_suite",tests)

    if file_name is not None:
        file_name = file_name +'.p'
        with open(file_name, 'wb') as f:
            pickle.dump(tests, f)

    return tests,observations
