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

from neuronunit.optimisation.optimization_management import dtc_to_rheo, mint_generic_model
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
        electro_path = str(os.getcwd())+'/..//tests/russell_tests.p'

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
        

    def get_cells(self,backend,model_parameters):
        #import pdb; pdb.set_trace()
        cpc = TSD(tests= self.test_frame['Cerebellum Purkinje cell'],use_rheobase_score=False)
        cpc_out = cpc.optimize(model_parameters.MODEL_PARAMS[backend], NGEN=9, \
                                backend=backend, MU=9, protocol={'allen': False, 'elephant': True})

        omc = TSD(tests= self.test_frame['Olfactory bulb (main) mitral cell'],use_rheobase_score=False)
        om_out = omc.optimize(model_parameters.MODEL_PARAMS[backend], NGEN=9, \
                                backend=backend, MU=9, protocol={'allen': False, 'elephant': True})
        ca1 = TSD(tests= self.test_frame['Hippocampus CA1 pyramidal cell'],use_rheobase_score=False)
        ca1_out = ca1.optimize(model_parameters.MODEL_PARAMS[backend], NGEN=9, backend=backend, MU=9, protocol={'allen': False, 'elephant': True})

        basket = TSD(tests= self.test_frame['Hippocampus CA1 basket cell'],use_rheobase_score=False)#, NGEN=10, \
        basket_out = basket.optimize(model_parameters.MODEL_PARAMS[backend], NGEN=9, \
                                backend=backend, MU=9, protocol={'allen': False, 'elephant': True})

        neo = TSD(tests= self.test_frame['Neocortex pyramidal cell layer 5-6'],use_rheobase_score=False)#, NGEN=10, \
        neo_out = neo.optimize(model_parameters.MODEL_PARAMS[backend], NGEN=8, \
                                backend=backend, MU=8, protocol={'allen': False, 'elephant': True})
        
        dtcpop0 = [p for p in om_out[0]['pf'] ]
        dtcpop1 = [p for p in cpc_out[0]['pf'] ]
        dtcpop2 = [p for p in ca1_out[0]['pf'] ]
        dtcpop3 = [p for p in basket_out[0]['pf'] ]
        dtcpop4 = [p for p in neo_out[0]['pf'] ]

        pdic = {str(backend):{'olf':dtcpop0,'purkine':dtcpop1,'ca1pyr':dtcpop2,'ca1basket':dtcpop3,'neo':dtcpop4}}

        pickle.dump(pdic,open(str(backend)+str('all_data_tests.p'),'wb'))

        return (dtcpop0,dtcpop1,dtcpop2,dtcpop3,dtcpop4,cpc_out,ca1_out,om_out,basket_out,neo_out)

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
        prev = 0
        use_test_old = 0.0
        tests_changed=0.0
        backend = str('ADEXP')
        (dtcpop0,dtcpop1,dtcpop2,dtcpop3,dtcpop4,cpc_out,ca1_out,om_out,basket_out,neo_out) = self.get_cells(backend,model_parameters)
        return (dtcpop0,dtcpop1,dtcpop2,dtcpop3,dtcpop4,cpc_out,ca1_out,om_out,basket_out,neo_out)


a = testHighLevelOptimisation()
a.setUp()
(dtcpop0,dtcpop1,dtcpop2,dtcpop3,dtcpop4, cpc_out,ca1_out,om_out,basket_out,neo_out) = a.test_data_driven_ae()
from neuronunit.optimisation.optimization_management import inject_and_plot
aaa = inject_and_plot(dtcpop0,second_pop=dtcpop1,third_pop=dtcpop2,figname='not_a_problemsfdsfs.png',snippets=False)
aaa = inject_and_plot(dtcpop0,second_pop=dtcpop1,third_pop=dtcpop3,figname='not_a_problemsffs.png',snippets=False)

import pdb
pdb.set_trace()
#new_dic ={}

#pdic = {key:{k:value[0]['pf']} for k,v in resultsae.items() for key,value in v.items() }
#with open('contentsae.p','wb') as f:
#    pickle.dump(pdic,f)
#B
