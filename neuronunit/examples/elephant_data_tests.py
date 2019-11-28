"""Tests of NeuronUnit test classes"""
import unittest
import os
import sys
from sciunit.utils import NotebookTools#,import_all_modules
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

from neuronunit.optimisation.optimization_management import TSD
#from neuronunit.optimisation.optimization_management import TSD
from neuronunit.optimisation import get_neab
from neuronunit.optimisation.data_transport_container import DataTC
from neuronunit.optimisation.optimization_management import dtc_to_rheo#, mint_generic_model
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
from neuronunit.optimisation.optimisations import run_ga
import pdb
from neuronunit.optimisation import model_parameters


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
        electro_path = str(os.getcwd())+'/../tests/russell_tests.p'

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

        self.MODEL_PARAMS = MODEL_PARAMS
        self.MODEL_PARAMS.pop(str('NEURON'),None)


        self.light_backends = [
                    str('RAWBackend'),
                    str('ADEXPBackend')
                ]

        # unused backends:
        self.heavy_backends = [
                    str('NEURONBackend'),
                    str('jNeuroMLBackend')
                ]

        self.medium_backends = [
                    str('GLIFBackend')
                ]


    def get_cells(self,backend,model_parameters,NGEN,MU,score_type=None,short_test=None):
        if score_type is not None:
            from sciunit import scores
            for v in self.test_frame.values():
                for _,values in v.items():
                    if score_type in str('ratio'):
                        values.score_type = scores.RatioScore
                    else:
                        values.score_type = scores.ZScore

        tests= self.test_frame['Hippocampus CA1 pyramidal cell']
        tests['name'] = 'Hippocampus CA1 pyramidal cell'
        ca1 = TSD(tests = tests,use_rheobase_score=True)
        ca1_out = ca1.optimize(model_parameters.MODEL_PARAMS[backend], NGEN=NGEN, \
                               backend=backend, MU=MU, protocol={'allen': False, 'elephant': True})
        dtcpop2 = [p for p in ca1_out[0]['pf'] ]

        tests = self.test_frame['Cerebellum Purkinje cell']
        tests['name'] = 'Cerebellum Purkinje cell'
        cpc = TSD(tests= tests,use_rheobase_score=False)
        cpc_out = cpc.optimize(model_parameters.MODEL_PARAMS[backend], NGEN=NGEN, \
                                backend=backend, MU=MU, protocol={'allen': False, 'elephant': True})
        dtcpop1 = [p for p in cpc_out[0]['pf'] ]
        tests = self.test_frame['Olfactory bulb (main) mitral cell']
        tests['name'] = 'Olfactory bulb (main) mitral cell'

        omc = TSD(tests=tests,use_rheobase_score=False)
        om_out = omc.optimize(model_parameters.MODEL_PARAMS[backend], NGEN=NGEN, \
                                backend=backend, MU=MU, protocol={'allen': False, 'elephant': True})
        dtcpop0 = [p for p in om_out[0]['pf'] ]

        tests = self.test_frame['Hippocampus CA1 basket cell']
        tests['name'] = 'Hippocampus CA1 basket cell'

        basket = TSD(tests=tests,use_rheobase_score=True)#, NGEN=10, \
        basket_out = basket.optimize(model_parameters.MODEL_PARAMS[backend], NGEN=NGEN, \
                                backend=backend, MU=8, protocol={'allen': False, 'elephant': True})
        dtcpop3 = [p for p in basket_out[0]['pf'] ]
        tests = self.test_frame['Neocortex pyramidal cell layer 5-6']
        tests['name'] = 'Neocortex pyramidal cell layer 5-6'

        neo = TSD(tests=tests,use_rheobase_score=True)#, NGEN=10, \
        neo_out = neo.optimize(model_parameters.MODEL_PARAMS[backend], NGEN=NGEN, \
                                backend=backend, MU=MU, protocol={'allen': False, 'elephant': True})
        dtcpop4 = [p for p in neo_out[0]['pf'] ]

        pdic = {str(backend):{'olf':dtcpop0,'purkine':dtcpop1,'ca1pyr':dtcpop2,'ca1basket':dtcpop3,'neo':dtcpop4}}

        pickle.dump(pdic,open(str(backend)+str('all_data_tests.p'),'wb'))
        return pdic

    def get_short_round_trip(self,backend,model_parameters,score_type=None,short_test=None):
        if score_type is not None:
            from sciunit import scores
            for v in self.test_frame.values():
                for _,values in v.items():
                    if score_type in str('ratio'):
                        values.score_type = scores.RatioScore
                    else:
                        values.score_type = scores.ZScore


        NGEN = 8
        MU = 4
        tests= self.test_frame['Hippocampus CA1 pyramidal cell']
        tests['name'] = 'Hippocampus CA1 pyramidal cell'
        ca1 = TSD(tests = tests,use_rheobase_score=True)
        OM = OptMan(ca1,protocol={'elephant':True,'allen':False})
        out = OM.round_trip_test(ca1,backend,model_parameters.MODEL_PARAMS[backend], NGEN=NGEN, MU=MU)
        return out
    def test_not_data_driven_rt_ae(self):
        '''
        forward euler, and adaptive exponential
        '''
        backend = str('RAW')
        out = self.get_short_round_trip(backend,model_parameters)
        return out 
    def test_data_driven_ae(self):
        '''
        forward euler, and adaptive exponential
        '''
        NGEN = 4
        MU = 4
        #pdb.set_trace()
        backend = str('RAW')
        out = self.get_cells(backend,model_parameters,NGEN,MU)

        backend = str('ADEXP')
        out = self.get_cells(backend,model_parameters,NGEN,MU)

        backend = str('BHH')
        out = self.get_cells(backend,model_parameters,NGEN,MU)
        #import pdb
        return out
