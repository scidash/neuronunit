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

from neuronunit.optimisation.optimization_management import dtc_to_rheo, mint_generic_model
from neuronunit.optimisation.optimization_management import OptMan

from neuronunit import tests as nu_tests, neuroelectro
from neuronunit.tests import passive, waveform, fi
#from neuronunit.optimisation import get_neab
from neuronunit.optimisation import exhaustive_search
from neuronunit.models.reduced import ReducedModel
#from neuronunit.optimisation import get_neab
from neuronunit.optimisation.model_parameters import MODEL_PARAMS
from neuronunit.tests import dynamics
from neuronunit.models.reduced import ReducedModel

#from neuronunit.optimisation.optimization_management import format_test, inject_and_plot
from neuronunit.optimisation import data_transport_container

from neuronunit.models.reduced import ReducedModel
#from neuronunit.optimisation import get_neab

from neuronunit.tests.fi import RheobaseTest, RheobaseTestP
#from neuronunit.optimisation import get_neab
from neuronunit.models.reduced import ReducedModel
from neuronunit import aibs
#from neuronunit.optimisation.optimization_management import round_trip_test, round_trip_test_rheob

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


    def test_solution_quality0(self):
        #for key, use_test in self.filtered_tests.items():
        #    use_test['protocol'] = str('elephant')
        #    break
        use_test = self.filtered_tests['Hippocampus CA1 pyramidal cell']#['RheobaseTest']]
        OM = OptMan(use_test,protocol={'elephant':True,'allen':False,'dm':False})
        results = OM.round_trip_test(use_test,str('RAW'),MU=8,NGEN=8,mini_tests=True)
        pdb.set_trace()
        model = results['pf'][0].dtc.dtc_to_model()

        return



a = testHighLevelOptimisation()
a.setUp()
a.test_solution_quality0()
#if __name__ == '__main__':
#    unittest.main()
