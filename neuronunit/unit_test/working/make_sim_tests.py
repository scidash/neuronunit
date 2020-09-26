
# coding: utf-8

# # Set up the environment
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import matplotlib
from neuronunit.optimisation.optimization_management import inject_and_plot_model, inject_and_plot_passive_model
import copy
import pickle
from neuronunit.optimisation.optimization_management import check_match_front, jrt
from scipy.stats import linregress
import unittest
import numpy as np
from neuronunit.optimisation.optimization_management import check_binary_match
from neuronunit.optimisation.optimization_management import which_key, TSD
from neuronunit.plottools import plot_score_history1
from neuronunit.optimisation.model_parameters import MODEL_PARAMS

def setUp(model_type):
    backend = model_type
    with open('processed_multicellular_constraints.p','rb') as f: test_frame = pickle.load(f)
    stds = {}
    for k,v in TSD(test_frame['Neocortex pyramidal cell layer 5-6']).items():
        temp = TSD(test_frame['Neocortex pyramidal cell layer 5-6'])[k]
        stds[k] = temp.observation['std']
    cloned_tests = copy.copy(test_frame['Neocortex pyramidal cell layer 5-6'])
    cloned_tests = TSD(cloned_tests)
    #{'RestingPotentialTest':cloned_tests['RestingPotentialTest']}
    OM = jrt(cloned_tests,backend,protocol='elephant')
    return OM
def test_all_objective_test(free_parameters,model_type="RAW"):
    results = {}
    tests = {}
    OM = setUp(model_type)
    simulated_data_tests, OM, target = OM.make_sim_data_tests(   
        model_type,
        free_parameters=free_parameters, 
        test_key=["RheobaseTest","TimeConstantTest","RestingPotentialTest","InputResistanceTest","CapacitanceTest","InjectedCurrentAPWidthTest","InjectedCurrentAPAmplitudeTest","InjectedCurrentAPThresholdTest"])  
    stds = {}
    for k,v in simulated_data_tests.items():
        keyed = which_key(simulated_data_tests[k].observation)
        if k == str('RheobaseTest'):
            mean = simulated_data_tests[k].observation[keyed]
            std = simulated_data_tests[k].observation['std']
            x = np.abs(std/mean)
        if k == str('TimeConstantTest') or k == str('CapacitanceTest') or k == str('InjectedCurrentAPWidthTest'):
            # or k == str('InjectedCurrentAPWidthTest'):
            mean = simulated_data_tests[k].observation[keyed]
            simulated_data_tests[k].observation['std'] = np.abs(mean)*2.0
        elif k == str('InjectedCurrentAPThresholdTest') or k == str('InjectedCurrentAPAmplitudeTest'):
            mean = simulated_data_tests[k].observation[keyed]
            simulated_data_tests[k].observation['std'] = np.abs(mean)*2.0
        stds[k] = (x,mean,std)
    target.tests = simulated_data_tests
    model = target.dtc_to_model()
    for t in simulated_data_tests.values(): 
        score0 = t.judge(target.dtc_to_model())
        score1 = target.tests[t.name].judge(target.dtc_to_model())
        assert float(score0.score)==0.0
        assert float(score1.score)==0.0
    tests = TSD(copy.copy(simulated_data_tests))
    check_tests = copy.copy(tests)
    return tests, OM, target
