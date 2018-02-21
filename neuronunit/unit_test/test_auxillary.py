"""Tests of NeuronUnit test classes"""
import unittest
import os
import sys
from sciunit.utils import NotebookTools#,import_all_modules
import dask
from dask import bag
from base import *

class TestBackend(NotebookTools,unittest.TestCase):

    def setUp(self):
        self.predictions = None
        self.predictionp = None
        self.score_p = None
        self.score_s = None
        from neuronunit.models.reduced import ReducedModel
        from neuronunit.optimization import get_neab
        self.model = ReducedModel(get_neab.LEMS_MODEL_PATH, backend='NEURON')

    def get_observation(self, cls):
        print(cls.__name__)
        neuron = {'nlex_id': 'nifext_50'} # Layer V pyramidal cell
        return cls.neuroelectro_summary_observation(neuron)

    def run_test(self, cls, pred =None):
        observation = self.get_observation(cls)
        test = cls(observation=observation)
        score = test.judge(self.model,stop_on_error = True, deep_error = True)
        return score




    def test_1rheobase_serial(self):
        from neuronunit.optimization import data_transport_container
        from neuronunit.tests.fi import RheobaseTest as T
        score = self.run_test(T)
        #print(score)


    def test_2inputresistance(self):
        from neuronunit.tests.passive import InputResistanceTest as T
        score = self.run_test(T)
        self.assertTrue(-0.6 < score < -0.5)

    def test_3restingpotential(self):
        from neuronunit.tests.passive import RestingPotentialTest as T
        score = self.run_test(T)
        self.assertTrue(1.2 < score < 1.3)

    def test_4capacitance(self):
        from neuronunit.tests.passive import CapacitanceTest as T
        score = self.run_test(T)
        self.assertTrue(-0.15 < score < -0.05)

    def test_5timeconstant(self):
        from neuronunit.tests.passive import TimeConstantTest as T
        score = self.run_test(T)
        self.assertTrue(-1.45 < score < -1.35)


    def test_19ap_width(self):
        from neuronunit.tests.waveform import InjectedCurrentAPWidthTest as T
        self.update_amplitude(T)
        score = self.run_test(T)
        self.assertTrue(-0.6 < score < -0.5)

    def test_20ap_amplitude(self):
        from neuronunit.tests.waveform import InjectedCurrentAPAmplitudeTest as T
        self.update_amplitude(T)
        score = self.run_test(T)
        self.assertTrue(-1.7 < score < -1.6)

    def test_21ap_threshold(self):
        from neuronunit.tests.waveform import InjectedCurrentAPThresholdTest as T
        self.update_amplitude(T)
        score = self.run_test(T)

    def sub_test_backend(self,dtc):
        import copy
        import unittest
        dtc = copy.copy(dtc)
        dtc.scores = {}
        size = len(list(dtc.attrs.values()))
        assertGreat(size,0)

        model = ReducedModel(get_neab.LEMS_MODEL_PATH, name= str('vanilla'), backend=('NEURON', {'DTC':dtc}))
        model.set_attrs(dtc.attrs)
        AssertGreat(len(list(self.model.attrs.values())),0)

        rbt = get_neab.tests[0]
        scoreN = rbt.judge(model,stop_on_error = False, deep_error = True)
        import copy
        dtc.scores[str(rbt)] = copy.copy(scoreN.sort_key)
        assertTrue(scoreN.sort_key is not None)
        dtc.rheobase = copy.copy(scoreN.prediction)
        #AssertN
        return dtc


    def test_0_score_df_movement(self):
        npoints = 2
        nparams = 10
        from neuronunit.optimization.model_parameters import model_params
        provided_keys = list(model_params.keys())
        USE_CACHED_GS = False
        from neuronunit.optimization import exhaustive_search
        self.grid_points = exhaustive_search.create_grid(npoints = npoints,nparams = nparams)
        import dask.bag as db
        b0 = db.from_sequence(grid_points[0:2], npartitions=8)
        self.dlist = list(db.map(exhaustive_search.update_dtc_grid,b0).compute())
        #self.dtcpop = dlist[0:2]#int(len(dlist)/N)]
        from neuronunit.optimization import get_neab
        from neuronunit.optimization.optimization_management import dtc_to_rheo
        from neuronunit.optimization.optimization_management import nunit_evaluation
        from neuronunit.optimization.optimization_management import format_test

        def compute_chunk(dtclist):
            dtclist = list(map(dtc_to_rheo,dtclist))
            import dask.bag as db
            b0 = db.from_sequence(dtclist, npartitions=8)
            dtclist = list(db.map(format_test,b0).compute())

            b0 = db.from_sequence(dtclist, npartitions=8)
            dtclist = list(db.map(nunit_evaluation,b0).compute())
            return dtclist
        processed_dtclist = compute_chunk(self.dlist)

    """Testing  notebooks"""

    def test_backend(self):
        grid_points = self.grid_points
        second_point = grid_points[int(len(grid_points)/2)]
        three_points = [grid_points[0],second_point,grid_points[-1]]
        assertEqual(len(three_points),3)
        dtcpop = list(map(exhaustive_search.update_dtc_grid,three_points))
        for i, dtc in enumerate(dtcpop):
            dtcpop[i] = self.sub_test_backend(dtc)
            #import pdb; pdb.set_trace()
        self.dtcpop = dtcpop
        return dtcpop

    def test_neuron_set_attrs(self):
        from neuronunit.models.reduced import ReducedModel
        from neuronunit.optimization import get_neab
        dtc = self.dtcpop[0]
        self.model = ReducedModel(get_neab.LEMS_MODEL_PATH, backend=('NEURON',{'DTC':dtc}))
        AssertGreat(len(list(self.model.attrs.values())),0)


    def test_1rheobase(self):
        from neuronunit.tests.fi import RheobaseTest, RheobaseTestP
        from neuronunit.optimization import get_neab
        from neuronunit.models.reduced import ReducedModel
        from neuronunit import aibs
        import os
        dataset_id = 354190013  # Internal ID that AIBS uses for a particular Scnn1a-Tg2-Cre
                                # Primary visual area, layer 5 neuron.
        observation = aibs.get_observation(dataset_id,'rheobase')
        rt = RheobaseTest(observation = observation)
        rtp = RheobaseTestP(observation = observation)
        model = ReducedModel(get_neab.LEMS_MODEL_PATH,name=str('vanilla'),backend='NEURON')
        self.score_p = rtp.judge(model,stop_on_error = False, deep_error = True)
        self.predictionp = self.score_p.prediction
        self.score_p = self.score_p.sort_key
        model = ReducedModel(get_neab.LEMS_MODEL_PATH,name=str('vanilla'),backend='NEURON')

        #self.score_neuron = rtp.judge(model,stop_on_error = False, deep_error = True)
        #self.prediction_neuron = self.score_p.prediction
        #self.score_neuron = self.score_p.sort_key
        #assertEqual(self.prediction_neuron,self.predictionp)

        serial_model = ReducedModel(get_neab.LEMS_MODEL_PATH,name=str('vanilla'),backend='NEURON')
        self.score_s = rt.judge(serial_model,stop_on_error = False, deep_error = True)
        self.predictions = self.score_s.prediction
        self.score_s = self.score_s.sort_key
        import numpy as np
        check_less_thresh = float(np.abs(self.predictionp['value'] - self.predictions['value']))
        #self.assertEqual(int(self.score_s*1000), int(self.score_p*1000))
        self.assertLessEqual(check_less_thresh, 3.0)
        #self.assertLess(check_less_thresh)
    # def test_attrs_diff_backends(self):
    #     attrs_ = {x:attrs[x] for x in ['a','b','c','d']}

    def test_backend_inheritance(self):
        from neuronunit.models.reduced import ReducedModel
        from neuronunit.optimization import get_neab
        print(get_neab.LEMS_MODEL_PATH)
        model = ReducedModel(get_neab.LEMS_MODEL_PATH, backend='NEURON')
        ma = list(dir(model))
        if 'get_spike_train' in ma and 'rheobase' in ma:
            return True
        else:
            return False

    def test_parallelnb_15(self):
        '''
        Lastly test the notebook
        '''
        self.do_notebook('test_ga_versus_grid')


if __name__ == '__main__':
    unittest.main()
