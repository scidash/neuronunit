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

from neuronunit.optimisation.optimization_management import dtc_to_rheo
from neuronunit.optimisation.optimization_management import nunit_evaluation
from neuronunit.optimisation.optimization_management import format_test, mint_generic_model

from neuronunit import tests as nu_tests, neuroelectro
from neuronunit.tests import passive, waveform, fi
#from neuronunit.optimisation import get_neab
from neuronunit.optimisation import exhaustive_search
from neuronunit.models.reduced import ReducedModel
#from neuronunit.optimisation import get_neab
from neuronunit.optimisation.model_parameters import MODEL_PARAMS
from neuronunit.tests import dynamics
from neuronunit.models.reduced import ReducedModel

from neuronunit.optimisation.optimization_management import format_test, inject_and_plot
from neuronunit.optimisation import data_transport_container

from neuronunit.models.reduced import ReducedModel
#from neuronunit.optimisation import get_neab

from neuronunit.tests.fi import RheobaseTest, RheobaseTestP
#from neuronunit.optimisation import get_neab
from neuronunit.models.reduced import ReducedModel
from neuronunit import aibs
from neuronunit.optimisation.optimization_management import round_trip_test

def test_all_tests_pop(dtcpop, tests):

    rheobase_test = [tests[0]['Hippocampus CA1 pyramidal cell']['RheobaseTest']]
    all_tests = list(tests[0]['Hippocampus CA1 pyramidal cell'].values())

    for d in dtcpop:
        d.tests = rheobase_test
        d.backend = str('RAW')
        assert len(list(d.attrs.values())) > 0

    dtcpop = list(map(dtc_to_rheo,dtcpop))


    b0 = db.from_sequence(dtcpop, npartitions=8)
    dtcpop = list(db.map(format_test,b0).compute())

    b0 = db.from_sequence(dtcpop, npartitions=8)
    dtcpop = list(db.map(nunit_evaluation,b0).compute())
    return dtcpop

def grid_points():
    npoints = 2
    nparams = 10
    free_params = MODEL_PARAMS[str('RAW')]
    USE_CACHED_GS = False
    grid_points = exhaustive_search.create_grid(npoints = npoints,free_params=free_params)
    b0 = db.from_sequence(list(grid_points)[0:2], npartitions=8)
    es = exhaustive_search.update_dtc_grid
    dtcpop = list(b0.map(es).compute())
    assert dtcpop is not None
    return dtcpop

class testHighLevelOptimisation(unittest.TestCase):

    def setUp(self):
        electro_path = str(os.getcwd())+'/../tests/russell_tests.p'

        assert os.path.isfile(electro_path) == True
        with open(electro_path,'rb') as f:
            (self.test_frame,self.obs_frame) = pickle.load(f)

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


    def test_solution_quality1(self):

        #Select random points in parameter space,
        #pretend these points are from experimental observations, by coding them in
        #as NeuroElectro observations.
        #This effectively redefines the sampled point as a the global minimum of error.
        #Show that the optimiser can find this point, only using information obtained by
        #sparesely learning the error surface.

        #MBEs = list(self.MODEL_PARAMS.keys())
        MBEs = [str('RAW'),str('BADEXP')]
        for key, use_test in self.test_frame.items():
            for b in MBEs:
                use_test['protocol'] = str('elephant')

                tuples_ = round_trip_test(use_test,b)
                (boolean,self.dtcpop) = tuples_
                print('done one')
                print(boolean,self.dtcpop)
                self.assertTrue(boolean)
        return

    def test_solution_quality0(self):

        from neuronunit.tests.allen_tests import pre_obs#, test_collection
        NGEN = 10
        local_tests = pre_obs[2][1]
        pre_obs[2][1]['spikes'][0]

        local_tests.update(pre_obs[2][1]['spikes'][0])
        local_tests['current_test'] = pre_obs[1][0]
        local_tests['spk_count'] = len(pre_obs[2][1]['spikes'])
        local_tests['protocol'] = str('allen')
        tuples_ = round_trip_test(local_tests,str('GLIF'))
        (boolean,self.dtcpop) = tuples_
        print('done one')
        print(boolean,self.dtcpop)
        self.assertTrue(boolean)
        return

    def test_solution_quality3(self):

        from neuronunit.tests.allen_tests import pre_obs#, test_collection
        NGEN = 10
        local_tests = pre_obs[2][1]
        pre_obs[2][1]['spikes'][0]

        local_tests.update(pre_obs[2][1]['spikes'][0])
        local_tests['current_test'] = pre_obs[1][0]
        local_tests['spk_count'] = len(pre_obs[2][1]['spikes'])
        local_tests['protocol'] = str('allen')
        tuples_ = round_trip_test(local_tests,str('RAW'))
        (boolean,self.dtcpop) = tuples_
        print('done one')
        print(boolean,self.dtcpop)
        self.assertTrue(boolean)
        return

    '''
    move to low level tests
    def test_rotate_backends2(self):
        self.dtcpop = grid_points()

        self.dtcpop = test_all_tests_pop(self.dtcpop,self.electro_tests)
        self.dtc = self.dtcpop[0]
        self.rheobase = self.dtc.rheobase

        broken_backends = [ str('NEURON'),str('jNeuroML') ]

        all_backends = [

            str('RAW'),
            str('ADEXP'),
            str('GLIF')

        ]

        for b in all_backends:
            if b in str('GLIF'):
                print(b)

            model = mint_generic_model(b)
            self.assertTrue(model is not None)
            from neuronunit.optimisation.data_transport_container import DataTC

            dtc = DataTC()
            dtc.backend = b
            dtc.attrs = model.attrs
            print(b,model.attrs)
            dtc = dtc_to_rheo(dtc)
            inject_and_plot(dtc)
            self.assertTrue(dtc is not None)

            #assert dtc is not None

        #MBEs = list(self.MODEL_PARAMS.keys())
        for b in all_backends:
            model = mint_generic_model(b)
            #assert model is not None
            self.assertTrue(model is not None)
            dtc = DataTC()
            dtc.backend = b
            dtc.attrs = model.attrs
            inject_and_plot(dtc)
            self.assertTrue(dtc is not None)

        return

    '''




if __name__ == '__main__':
    unittest.main()
