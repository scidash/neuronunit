"""Tests of NeuronUnit test classes"""
import unittest
import os
import sys
import dask
from dask import bag


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
#
from neuronunit.optimisation.optimisation_management import dtc_to_rheo
from neuronunit.optimisation.optimisation_management import nunit_evaluation
from neuronunit.optimisation.optimisation_management import format_test, mint_generic_model

from neuronunit import tests as nu_tests, neuroelectro
from neuronunit.tests import passive, waveform, fi
#from neuronunit.optimisation import get_neab
from neuronunit.optimisation import exhaustive_search
from neuronunit.models.reduced import ReducedModel
#from neuronunit.optimisation import get_neab
from neuronunit.optimisation.model_parameters import MODEL_PARAMS
from neuronunit.tests import dynamics
from neuronunit.models.reduced import ReducedModel

from neuronunit.optimisation.optimisation_management import format_test
from neuronunit.optimisation import data_transport_container

from neuronunit.models.reduced import ReducedModel
#from neuronunit.optimisation import get_neab

from neuronunit.tests.fi import RheobaseTest, RheobaseTestP
#from neuronunit.optimisation import get_neab
from neuronunit.models.reduced import ReducedModel
from neuronunit import aibs

def test_all_tests_pop(dtcpop, tests):
    rheobase_test = tests[0][0][0]
    all_tests = tests[0][0]
    for d in dtcpop:
        d.tests = rheobase_test
        d.backend = str('RAW')
        assert len(list(d.attrs.values())) > 0

    dtcpop = list(map(dtc_to_rheo,dtcpop))

    for d in dtcpop:
        d.tests = all_tests
        d.backend = str('RAW')
        assert len(list(d.attrs.values())) > 0

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

        try:
            electro_path = str(os.getcwd())+'all_tests.p'
            assert os.path.isfile(electro_path) == True
            with open(electro_path,'rb') as f:
                (self.obs_frame,self.test_frame) = pickle.load(f)

        except:
            for p in pipe:
                p_tests, p_observations = get_neab.get_neuron_criteria(p)
                self.obs_frame[p["name"]] = p_observations#, p_tests))
                self.test_frame[p["name"]] = p_tests#, p_tests))
            electro_path = str(os.getcwd())+'all_tests.p'
            with open(electro_path,'wb') as f:
                pickle.dump((obs_frame,test_frame),f)

        self.predictions = None
        self.predictionp = None
        self.score_p = None
        self.score_s = None
        self.dtcpop = grid_points()
         #self.grid_points

        electro_path = 'pipe_tests.p'
        assert os.path.isfile(electro_path) == True
        with open(electro_path,'rb') as f:
            self.electro_tests = pickle.load(f)
        #self.electro_tests = get_neab.replace_zero_std(self.electro_tests)

        #self.test_rheobase_dtc = test_rheobase_dtc
        #self.dtcpop = test_rheobase_dtc(self.dtcpop,self.electro_tests)
        self.dtcpop = test_all_tests_pop(self.dtcpop,self.electro_tests)
        self.dtc = self.dtcpop[0]
        self.rheobase = self.dtc.rheobase
        self.standard_model = self.model = mint_generic_model('RAW')
        self.MODEL_PARAMS = MODEL_PARAMS
        self.MODEL_PARAMS.pop(str('NEURON'),None)

        self.heavy_backends = [
                    str('NEURONBackend'),
                    str('jNeuroMLBackend')
                ]
        self.light_backends = [
                    str('RAWBackend'),
                    str('HHBackend')
                ]
        self.medium_backends = [
                    str('GLIFBackend')
                ]
        #self.standard_model = ReducedModel(get_neab.LEMS_MODEL_PATH, backend='RAW')
        #self.model = ReducedModel(get_neab.LEMS_MODEL_PATH, backend='RAW')


    def test_rotate_backends0(self):
        broken_backends = [ str('NEURON') ]

        all_backends = [
            str('jNeuroML'),
            str('RAW'),
            str('HH'),
            str('GLIF')

        ]

        for b in all_backends:
            if b in str('GLIF'):
                print(b)

            model = mint_generic_model(b)
            self.assertTrue(model is not None)
            dtc = DataTC()
            dtc.backend = b
            dtc.attrs = model.attrs
            dtc = dtc_to_rheo(dtc)
            inject_and_plot(dtc)
            self.assertTrue(dtc is not None)

            #assert dtc is not None

        MBEs = list(self.MODEL_PARAMS.keys())
        for b in MBEs:
            model = mint_generic_model(b)
            #assert model is not None
            self.assertTrue(model is not None)
            dtc = DataTC()
            dtc.backend = b
            dtc.attrs = model.attrs
            inject_and_plot(dtc)
            self.assertTrue(dtc is not None)

        return

    def test_get_druckmann1(self):
        '''
        test the extraction of Druckmann property Ephys measurements.
        '''
        from neuronunit.optimisation.optimisation_management import add_dm_properties_to_cells
        (self.dtcpop,dm_properties) = add_dm_properties_to_cells(self.dtcpop)

        for d in dm_properties:
            self.assertTrue(d is not None)
        return

    def test_executable_druckmann_science_unit_tests(self):
        '''
        test the extraction of Druckmann property Ephys measurements.
        '''
        from neuronunit.optimisation.optimisation_management import add_dm_properties_to_cells

        tests,observations = get_neab.executable_druckman_tests(p)
        (self.dtcpop,dm_properties) = add_dm_properties_to_cells(self.dtcpop)

    def test_solution_quality0(self):
        '''
        Select random points in parameter space,
        pretend these points are from experimental observations, by coding them in
        as NeuroElectro observations.
        This effectively redefines the sampled point as a the global minimum of error.
        Show that the optimiser can find this point, only using information obtained by
        sparesely learning the error surface.

        '''
        MBEs = list(self.MODEL_PARAMS.keys())
        from neuronunit.optimisation.optimisation_management import round_trip_test
        for key, use_test in self.test_frame.items():
            for b in MBEs:
                tuple = round_trip_test(use_test,b)
                (boolean,self.dtcpop) = tuple

                self.assertTrue(boolean)
        return



if __name__ == '__main__':
    unittest.main()