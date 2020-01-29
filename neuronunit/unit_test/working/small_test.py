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

from neuronunit.optimisation.optimization_management import dtc_to_rheo#, mint_generic_model
from neuronunit.optimisation.optimization_management import OptMan

from neuronunit import tests as nu_tests, neuroelectro
from neuronunit.tests import passive, waveform, fi
from neuronunit.optimisation import exhaustive_search
from neuronunit.optimisation.model_parameters import MODEL_PARAMS
from neuronunit.tests import dynamics
from neuronunit.optimisation import data_transport_container

from neuronunit.tests.fi import RheobaseTest, RheobaseTestP
from neuronunit import aibs
from neuronunit.optimisation.optimisations import run_ga
from neuronunit.optimisation import model_parameters
from neuronunit.optimisation import mint_tests

def test_all_tests_pop(dtcpop, tests):

    rheobase_test = [tests[0]['Hippocampus CA1 pyramidal cell']['RheobaseTest']]
    all_tests = list(tests[0]['Hippocampus CA1 pyramidal cell'].values())

    for d in dtcpop:
        d.tests = rheobase_test
        d.backend = str('RAW')
        assert len(list(d.attrs.values())) > 0

    dtcpop = list(map(dtc_to_rheo,dtcpop))
    OM = OptMan(all_tests,simulated_obs=False)
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
        import neuronunit
        anchor = neuronunit.__file__
        anchor = os.path.dirname(anchor)
        electro_path = os.path.join(os.sep,anchor,'tests/multicellular_constraints.p')
        os.path.isfile(electro_path)
        #electro_path = str(os.getcwd())+'/../../tests/russell_tests.p'
        """
        if os.path.isfile(electro_path):
            try:
                assert os.path.isfile(electro_path) == True
            except:
                print('Exception')
            with open(electro_path,'rb') as f:
                try:
                    self.test_frame = pickle.load(f)
                except:
                    (self.test_frame,self.obs_frame) = pickle.load(f)
        else:
            self.test_frame = mint_tests.get_cell_constraints()
            df = pd.DataFrame(rts)
            import pdb
            pdb.set_trace()
        """
        from neuronunit.optimisation import get_neab
        self.test_frame = get_neab.process_all_cells()
        self.test_frame = {k:tf for k,tf in self.test_frame.items() if len(tf.tests)>0 }
        for testsuite in self.test_frame.values():
            for t in testsuite.tests:
                if float(t.observation['std']) == 0.0:
                    t.observation['std'] = t.observation['mean']
         #[tf for tf in self.test_frame if len(tf.tests)>0 ]

        self.filtered_tests = {key:val for key,val in self.test_frame.items() }# if len(val) ==8}
        print(self.filtered_tests)
        self.predictions = None
        self.predictionp = None
        self.score_p = None
        self.score_s = None
         #self.grid_points

        #electro_path = 'pipe_tests.p'
        with open(electro_path,'rb') as f:
            self.electro_tests = pickle.load(f)
        #self.electro_tests = get_neab.replace_zero_std(self.electro_tests)

        #self.test_rheobase_dtc = test_rheobase_dtc
        #self.dtcpop = test_rheobase_dtc(self.dtcpop,self.electro_tests)
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
        #use_test = self.filtered_tests['Hippocampus CA1 pyramidal cell']#['RheobaseTest']]
        from neuronunit.optimisation.optimization_management import TSD
        try:
            use_test = TSD(self.filtered_tests['Neocortex pyramidal cell layer 5-6'])
        except:
            use_test = TSD(self.filtered_tests[list(self.filtered_tests.keys())[0]])
        #easy_standards = {ut.name:ut.observation['std'] for ut in use_test.values()}

        use_test.use_rheobase_score = True
        [(value.name,value.observation) for value in use_test.values()]
        print(use_test)
        edges = model_parameters.MODEL_PARAMS["RAW"]
        OM = OptMan(use_test,protocol={'elephant':True,'allen':False,'dm':False})
        results,converged,target,simulated_tests = OM.round_trip_test(use_test,str('RAW'),MU=2,NGEN=2)#,stds = easy_standards)
        print(converged,target)
        temp = [results,converged,target,simulated_tests]

        with open('jd.p','wb') as f:
            pickle.dump(temp,f)
        param_edges = model_parameters.MODEL_PARAMS['HH']
        #try:
        #    with open('jda.p','rb') as f:
        #        adconv = pickle.load(f)[0]
        #except:
        ga_out = run_ga(param_edges, 2, simulated_tests, free_params=param_edges.keys(), \
                backend=str('HH'), MU = 4,  protocol={'allen': False, 'elephant': True})

        ga_out = use_test.optimize(edges,backend="RAW",protocol={'allen': False, 'elephant': True},\
           MU=5,NGEN=5,free_params=None,seed_pop=None,hold_constant=None)

        try:
            adconv = [ p.dtc for p in ga_out[0]['pf'] ]
            for dtc in adconv:
                dtc.tests = None

            for dtc in adconv:
                for attr, value in dtc.__dict__.items():
                    print(attr);
                    pickle.dump([attr,value],open('temp.p','wb'))

            with open('jda.p','wb') as f:
                temp = adconv
                pickle.dump(temp,f)
        except:
            import pdb
            pdb.set_trace()
        return
        """
        import copy
        from neuronunit.optimisation import optimization_management as om
        om.inject_and_plot(copy.copy(converged),second_pop=copy.copy(target),third_pop=copy.copy(adconv),figname='snippets_false.png',snippets=False)
        om.inject_and_plot(copy.copy(converged),second_pop=copy.copy(target),third_pop=copy.copy(adconv),figname='snippets_true.png',snippets=True)
        om.inject_and_plot(copy.copy(adconv),second_pop=copy.copy(adconv),third_pop=copy.copy(adconv),figname='adexp_only_true.png',snippets=True)
        om.inject_and_plot(copy.copy(adconv),second_pop=copy.copy(adconv),third_pop=copy.copy(adconv),figname='adexp_only_false.png',snippets=False)

        #mpa = adconv[0].iap()
        #cpm = converged[0].iap()
        return
        """

if __name__ == '__main__':
    unittest.main()
