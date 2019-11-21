"""Tests of NeuronUnit test classes"""
import unittest
import os
import sys
import dask
from dask import bag
import matplotlib
matplotlib.use('Agg')

from itertools import repeat
import quantities as pq

import copy
import unittest
import pickle

import numpy as np
import pickle
import dask.bag as db
import os


from neuronunit.optimisation import get_neab
from neuronunit.optimisation.data_transport_container import DataTC

from neuronunit.optimisation.optimization_management import dtc_to_rheo, mint_generic_model, TSD
from neuronunit.optimisation.optimization_management import OptMan

from neuronunit import tests as nu_tests, neuroelectro
from neuronunit.tests import passive, waveform, fi
from neuronunit.optimisation import exhaustive_search
from neuronunit.models.reduced import ReducedModel
from neuronunit.optimisation.model_parameters import MODEL_PARAMS
from neuronunit.tests import dynamics
from neuronunit.models.reduced import ReducedModel


from neuronunit.optimisation import data_transport_container

from neuronunit.models.reduced import ReducedModel

from neuronunit.tests.fi import RheobaseTest, RheobaseTestP
from neuronunit.models.reduced import ReducedModel
from neuronunit import aibs


def test_all_tests_pop(dtcpop, tests):

    rheobase_test = [tests[0]['Hippocampus CA1 pyramidal cell']['RheobaseTest']]
    all_tests = list(tests[0]['Hippocampus CA1 pyramidal cell'].values())

    for d in dtcpop:
        d.tests = rheobase_test
        d.backend = str('RAW')
        assert len(list(d.attrs.values())) > 0

    dtcpop = list(map(dtc_to_rheo,dtcpop))
    OM = OptMan(all_tests)

    format_test = OM.format_test
    elephant_evaluation = OM.elephant_evaluation

    b0 = db.from_sequence(dtcpop, npartitions=8)
    dtcpop = list(db.map(format_test,b0).compute())

    b0 = db.from_sequence(dtcpop, npartitions=8)
    dtcpop = list(db.map(elephant_evaluation,b0).compute())
    return dtcpop

class testHighLevelOptimisation(unittest.TestCase):

    def setUp(self):
        electro_path = str(os.getcwd())+'/../../tests/russell_tests.p'

        assert os.path.isfile(electro_path) == True
        with open(electro_path,'rb') as f:
            (self.test_frame,self.obs_frame) = pickle.load(f)
        self.filtered_tests = {key:val for key,val in self.test_frame.items() if len(val) ==8}

        self.predictions = None
        self.predictionp = None
        self.score_p = None
        self.score_s = None
         #self.grid_points

        #electro_path = 'pipe_tests.p'
        assert os.path.isfile(electro_path) == True
        with open(electro_path,'rb') as f:
            self.electro_tests = pickle.load(f)
        #self.electro_tests = get_neab.replace_zero_std(self.electro_tests)

        #self.test_rheobase_dtc = test_rheobase_dtc
        #self.dtcpop = test_rheobase_dtc(self.dtcpop,self.electro_tests)
        self.standard_model = self.model = mint_generic_model('RAW')
        self.MODEL_PARAMS = MODEL_PARAMS
        self.MODEL_PARAMS.pop(str('NEURON'),None)

        self.heavy_backends = [
                    str('NEURONBackend'),
                    str('jNeuroMLBackend')
                ]
        self.light_backends = [
                    str('RAWBackend'),
                    str('ADEXPBackend')
                ]
        self.medium_backends = [
                    str('GLIFBackend')
                ]

    def test_data_driven_ae(self):
        '''
        forward euler, and adaptive exponential
        '''
        use_test1 = self.filtered_tests['Hippocampus CA1 pyramidal cell']
        use_tests = list(self.test_frame['Hippocampus CA1 pyramidal cell'].values())
        from neuronunit.optimisation.optimisations import run_ga
        import pdb
        from neuronunit.optimisation import model_parameters
        results = {}
        results['RAW'] = {}
        results['ADEXP'] = {}

        for key, use_test in self.test_frame.items():
            use_test = TSD(use_test)
            use_test.use_rheobase_score = True
            backend = str('ADEXP')
            NGEN = MU = 2
            OM = OptMan(model_parameters.MODEL_PARAMS[backend], NGEN, use_tests, MU=MU, protocol={'allen': False, 'elephant': True})
            ga_out = OM.optimize()
            #ga_out = run_ga(model_parameters.MODEL_PARAMS[backend], NGEN, use_tests, free_params=model_parameters.MODEL_PARAMS[backend].keys(), \
#                            backend=backend, MU=MU, protocol={'allen': False, 'elephant': True})
            results[backend][key] = copy.copy(ga_out)
            backend = str('RAW')
            ga_out = run_ga(model_parameters.MODEL_PARAMS[backend], NGEN, use_tests, free_params=model_parameters.MODEL_PARAMS[backend].keys(), \
                            backend=backend, MU=MU, protocol={'allen': False, 'elephant': True})
            results[backend][key] = copy.copy(ga_out)
            break
        return results
a = testHighLevelOptimisation()
a.setUp()
resultsae = a.test_data_driven_ae()
new_dic ={}
with open('contentsae.p','wb') as f:
    pickle.dump(restultsae,f)
B
