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
from neuronunit.optimisation.optimization_management import OptMan as OM, OptMan
elephant_evaluation = OM.elephant_evaluation
format_test = OM.format_test
round_trip_test = OM.format_test

#mint_generic_model = OM.mint_generic_model

from neuronunit.optimisation.optimization_management import mint_generic_model
from neuronunit.optimisation import mint_tests
from neuronunit import tests as nu_tests, neuroelectro
from neuronunit.tests import passive, waveform, fi
#from neuronunit.optimisation import get_neab
from neuronunit.optimisation import exhaustive_search
from neuronunit.models.reduced import ReducedModel, VeryReducedModel
#from neuronunit.optimisation import get_neab
from neuronunit.optimisation.model_parameters import MODEL_PARAMS
from neuronunit.tests import dynamics
from neuronunit.optimisation.optimization_management import TSD

#from neuronunit.models.reduced import

#from neuronunit.optimisation.optimization_management import format_test, inject_and_plot
from neuronunit.optimisation import data_transport_container

from neuronunit.models.reduced import ReducedModel
#from neuronunit.optimisation import get_neab

from neuronunit.tests.fi import RheobaseTest, RheobaseTestP
#from neuronunit.optimisation import get_neab
from neuronunit.models.reduced import ReducedModel
from neuronunit import aibs
import pandas as pd
from neuronunit.optimisation import get_neab

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
        self.test_frame = get_neab.process_all_cells()

        self.test_frame = {k:tf for k,tf in self.test_frame.items() if len(tf.tests)>0 }

        for testsuite in self.test_frame.values():
            for t in testsuite.tests:
                if float(t.observation['std']) == 0.0:
                    t.observation['std'] = t.observation['mean']

        self.predictions = None
        self.predictionp = None
        self.score_p = None
        self.score_s = None
        dtc = DataTC()
        dtc.backend = 'RAW'
        try:
            self.standard_model = self.model = dtc.dtc_to_model()
        except:
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
            import sciunit
            if type(use_test) is type(sciunit.suites.TestSuite):
                print('capture')
                use_test = {t.name:t for t in use_test}
                import pdb
                pdb.set_trace()
            use_test = {k.name:k for k in use_test.tests }
            use_test['protocol'] = str('elephant')
            use_test = TSD(use_test)
            use_test.use_rheobase_score = True

            for b in MBEs:
                edges = self.MODEL_PARAMS[b]

                OM = OptMan(use_test,\
                    backend=b,\
                    boundary_dict=edges,\
                    protocol={'allen': False, 'elephant': True})
                out = OM.round_trip_test(use_test,b,edges,NGEN = 10, MU = 10)
                boolean = out[4]<0.5
                print('done one')
                #print(boolean,self.dtcpop)
                self.assertTrue(boolean)
        return

    def test_solution_quality0(self):

        from neuronunit.tests.allen_tests import pre_obs#, test_collection
        NGEN = 10
        local_tests = pre_obs[2][1]
        pre_obs[2][1]['spikes'][0]
        b = str('GLIF')

        local_tests.update(pre_obs[2][1]['spikes'][0])
        local_tests['current_test'] = pre_obs[1][0]
        local_tests['spk_count'] = len(pre_obs[2][1]['spikes'])
        local_tests['protocol'] = str('allen')

        edges = self.MODEL_PARAMS[b]

        OM = OptMan(local_tests,\
            backend=b,\
            boundary_dict=edges,\
            protocol={'allen': False, 'elephant': True})
        out = OM.round_trip_test(local_tests,b,edges,NGEN = 10, MU = 10)
        boolean = out[4]<0.5
        print('done one')
        #print(boolean,self.dtcpop)
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
