"""Tests of NeuronUnit test classes"""
import unittest
import os
import sys
#from sciunit.utils import NotebookTools#,import_all_modules
import dask
from dask import bag
from base import *

def grid_points():
    npoints = 2
    nparams = 10
    from neuronunit.optimization.model_parameters import model_params
    provided_keys = list(model_params.keys())
    USE_CACHED_GS = False
    electro_path = 'pipe_tests.p'
    import pickle
    assert os.path.isfile(electro_path) == True
    with open(electro_path,'rb') as f:
        electro_tests = pickle.load(f)
    from neuronunit.optimization import exhaustive_search
    grid_points = exhaustive_search.create_grid(npoints = npoints,nparams = nparams)
    import dask.bag as db
    b0 = db.from_sequence(grid_points[0:2], npartitions=8)
    dtcpop = list(db.map(exhaustive_search.update_dtc_grid,b0).compute())
    assert dtcpop is not None
    return dtcpop

def test_01a_compute_score(dtcpop, tests):
    from neuronunit.optimization import get_neab
    from neuronunit.optimization.optimization_management import dtc_to_rheo
    from neuronunit.optimization.optimization_management import nunit_evaluation
    from neuronunit.optimization.optimization_management import format_test
    from itertools import repeat
    #dtcpop = grid_points()
    rheobase_test = tests[0][0][0]

    xargs = list(zip(dtcpop,repeat(rheobase_test),repeat('NEURON')))
    dtclist = list(map(dtc_to_rheo,xargs))

    #dtclist = list(map(dtc_to_rheo,dtcpop))
    for d in dtclist:
        assert len(list(d.attrs.values())) > 0
    import dask.bag as db
    b0 = db.from_sequence(dtclist, npartitions=8)
    dtclist = list(db.map(format_test,b0).compute())

    b0 = db.from_sequence(dtclist, npartitions=8)
    dtclist = list(db.map(nunit_evaluation,b0).compute())
    return dtclist

class testOptimizationBackend(NotebookTools,unittest.TestCase):

    def setUp(self):
        self.predictions = None
        self.predictionp = None
        self.score_p = None
        self.score_s = None
        self.grid_points = grid_points()
        dtcpop = self.grid_points
        import pickle
        electro_path = 'pipe_tests.p'
        assert os.path.isfile(electro_path) == True
        with open(electro_path,'rb') as f:
            self.electro_tests = pickle.load(f)
        #self.electro_tests = get_neab.replace_zero_std(self.electro_tests)

        self.test_01a_compute_score = test_01a_compute_score
        self.dtcpop = test_01a_compute_score(dtcpop,self.electro_tests)
        self.dtc = self.dtcpop[0]
        self.rheobase = self.dtc.rheobase
        from neuronunit.models.reduced import ReducedModel
        from neuronunit.optimization import get_neab
        self.standard_model = ReducedModel(get_neab.LEMS_MODEL_PATH, backend='NEURON')
        self.model = ReducedModel(get_neab.LEMS_MODEL_PATH, backend='NEURON')

    def get_observation(self, cls):
        print(cls.__name__)
        neuron = {'nlex_id': 'nifext_50'} # Layer V pyramidal cell
        return cls.neuroelectro_summary_observation(neuron)

    def run_test(self, cls, pred =None):
        observation = self.get_observation(cls)
        test = cls(observation=observation)
        score = test.judge(self.standard_model, stop_on_error = True, deep_error = True)
        return score

    # Get experimental electro physology bservations for a dentate gyrus baskett cell
    # An inhibitory neuron
    @unittest.skip("Not fully developed yet")
    def test_get_rate_CV(self):
        # Dictionary of observations, in this case two ephys properties from one paper
        from neuronunit.tests import dynamics
        import quantities as pq
        doi = 'doi:10.1113/jphysiol.2010.200683'
        observations={doi:{'ap_amplitude':{'mean':45.1*pq.mV,
                                           'sem':0.7*pq.mV,
                                           'n':25},
                           'ap_width':{'mean':19.7*pq.ms,
                                       'sem':1.0*pq.ms,
                                       'n':25}}}

        # Instantiate two tests based on these properties
        ap_width_test = APWidthTest(observation=observations[doi]['ap_width'])
        ap_amplitude_test = APAmplitudeTest(observation=observations[doi]['ap_amplitude'])
        from neuronunit import tests as nu_tests, neuroelectro
        from neuronunit.tests import passive, waveform, fi
        cholinergic = {'neuron':'115'}
        observation = {}
        observation[doi] = {}
        observation[doi]['isi'] = 598.0*pq.ms
        observation[doi]['mean'] = 598.0*pq.ms
        observation[doi]['std'] = 37.0*pq.ms
        isi_test = dynamics.ISITest(observation=observation[doi])
        observation = {}
        observation[doi] = {}
        observation[doi]['isi'] = 16.1
        observation[doi]['mean'] = 16.1*pq.ms
        observation[doi]['std'] = 2.1*pq.ms

    #@unittest.skip("Not fully developed yet")
    def test_get_inhibitory_neuron(self):
        from neuronunit import tests as nu_tests, neuroelectro
        from neuronunit.tests import passive, waveform, fi
        fi_basket = {'nlex_id':'NLXCELL:100201'}
        #observation =  cls.neuroelectro_summary_observation(fi_basket)
        test_class_params = [(fi.RheobaseTest,None),
                         (passive.InputResistanceTest,None),
                         (passive.TimeConstantTest,None),
                         (passive.CapacitanceTest,None),
                         (passive.RestingPotentialTest,None),
                         (waveform.InjectedCurrentAPWidthTest,None),
                         (waveform.InjectedCurrentAPAmplitudeTest,None),
                         (waveform.InjectedCurrentAPThresholdTest,None)]#,
        inh_observations = []
        for cls,params in test_class_params:
            inh_observations.append(cls.neuroelectro_summary_observation(fi_basket))
        self.inh_observations = inh_observations
        return inh_observations

    def test_rheobase(self):
        from neuronunit.models.reduced import ReducedModel
        from neuronunit.optimization import get_neab
        import copy
        import unittest
        dtc = copy.copy(self.dtc)
        self.assertNotEqual(self.dtc,None)
        dtc.scores = {}
        size = len([ v for v in dtc.attrs.values()])
        assert size > 0
        self.assertGreater(size,0)
        model = ReducedModel(get_neab.LEMS_MODEL_PATH, name= str('vanilla'),
                             backend=('NEURON', {'DTC':dtc}))
        temp = [v for v in model.attrs.values()]
        assert len(temp) > 0
        self.assertGreater(len(temp), 0)
        rbt = get_neab.tests[0]
        scoreN = rbt.judge(model, stop_on_error = False, deep_error = True)
        import copy
        dtc.scores[str(rbt)] = copy.copy(scoreN.sort_key)
        assert scoreN.sort_key is not None
        self.assertTrue(scoreN.sort_key is not None)
        dtc.rheobase = copy.copy(scoreN.prediction)
        return dtc

    def test_rheobase_on_list(self):
        from neuronunit.optimization import exhaustive_search
        grid_points = self.grid_points
        second_point = grid_points[int(len(grid_points)/2)]
        three_points = [grid_points[0], second_point, grid_points[-1]]
        self.assertEqual(len(three_points),3)
        dtcpop = list(map(exhaustive_search.update_dtc_grid, three_points))
        for d in self.dtcpop:
            assert len(list(d.attrs.values())) > 0
        dtcpop = self.test_01a_compute_score(self.dtcpop)
        self.dtcpop = dtcpop
        return dtcpop

    def test_neuron_set_attrs(self):
        from neuronunit.models.reduced import ReducedModel
        from neuronunit.optimization import get_neab
        self.assertNotEqual(self.dtcpop,None)
        dtc = self.dtcpop[0]
        self.model = ReducedModel(get_neab.LEMS_MODEL_PATH,
                                  backend=('NEURON',{'DTC':dtc}))
        temp = [ v for v in self.model.attrs.values() ]
        assert len(temp) > 0
        self.AssertGreater(temp,0)
        old_ = self.model.attrs.items()
        assert self.model.attrs.keys() in old_
        assert self.model.attrs.values() in old_

    def test_rheobase_serial(self):
        from neuronunit.optimization import data_transport_container
        from neuronunit.tests.fi import RheobaseTest as T
        score = self.run_test(T)
        self.rheobase = score.prediction
        self.assertNotEqual(self.rheobase,None)
        self.dtc.attrs = self.model.attrs


    def test_inputresistance(self):
        from neuronunit.tests.passive import InputResistanceTest as T
        score = self.run_test(T)
        print(score)
        print(score.sort_key)
        self.assertTrue(-0.6 < float(score.sort_key) < -0.5)

    def test_restingpotential(self):
        from neuronunit.tests.passive import RestingPotentialTest as T
        score = self.run_test(T)
        self.assertTrue(1.2 < score < 1.3)

    def test_capacitance(self):
        from neuronunit.tests.passive import CapacitanceTest as T
        score = self.run_test(T)
        self.assertTrue(-0.15 < score < -0.05)

    def test_timeconstant(self):
        from neuronunit.tests.passive import TimeConstantTest as T
        score = self.run_test(T)
        self.assertTrue(-1.45 < score < -1.35)



    def test_ap_width(self):
        from neuronunit.models.reduced import ReducedModel
        from neuronunit.optimization import get_neab
        from neuronunit.tests.waveform import InjectedCurrentAPWidthTest as T

        #self.update_amplitude(T)
        score = self.run_test(T,pred=self.rheobase)
        self.assertTrue(-0.6 < score < -0.5)

    def test_ap_amplitude(self):
        from neuronunit.models.reduced import ReducedModel
        from neuronunit.optimization import get_neab
        from neuronunit.tests.waveform import InjectedCurrentAPAmplitudeTest as T
        #from neuronunit.optimization.optimization_management import format_test
        #from neuronunit.optimization import data_transport_container
        #dtc = data_transport_container.DataTC()
        #dtc.rheobase = self.rheobase
        #def run_test(self, cls, pred =None):
        #dtc = format_test(dtc)
        #self.model = ReducedModel(get_neab.LEMS_MODEL_PATH, backend=('NEURON',{'DTC':dtc}))

        score = self.run_test(T,pred=self.rheobase)
        self.assertTrue(-1.7 < score < -1.6)

    def test_ap_threshold(self):
        from neuronunit.models.reduced import ReducedModel
        from neuronunit.optimization import get_neab
        from neuronunit.tests.waveform import InjectedCurrentAPThresholdTest as T
        from neuronunit.optimization.optimization_management import format_test
        from neuronunit.optimization import data_transport_container
        dtc = data_transport_container.DataTC()
        dtc.rheobase = self.rheobase
        dtc = format_test(dtc)
        self.model = ReducedModel(get_neab.LEMS_MODEL_PATH, backend=('NEURON',{'DTC':dtc}))
        #score = self.run_test(T)
        score = self.run_test(T,pred=self.rheobase)




    def test_rheobase_single_value_parallel_and_serial_comparison(self):
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
        model = ReducedModel(get_neab.LEMS_MODEL_PATH, backend=('NEURON'))

        #model = ReducedModel(get_neab.LEMS_MODEL_PATH,name=str('vanilla'),backend='NEURON')
        self.score_p = rtp.judge(model,stop_on_error = False, deep_error = True)
        self.predictionp = self.score_p.prediction
        self.score_p = self.score_p.sort_key
        #model = ReducedModel(get_neab.LEMS_MODEL_PATH,name=str('vanilla'),backend='NEURON')

        serial_model = ReducedModel(get_neab.LEMS_MODEL_PATH,name=str('vanilla'),backend='NEURON')
        self.score_s = rt.judge(serial_model,stop_on_error = False, deep_error = True)
        self.predictions = self.score_s.prediction
        self.score_s = self.score_s.sort_key
        import numpy as np
        check_less_thresh = float(np.abs(self.predictionp['value'] - self.predictions['value']))
        self.assertLessEqual(check_less_thresh, 2.0)

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

    """Testing  notebooks"""
    @unittest.skip("takes too long")
    def test_parallelnb_15(self):
        '''
        Lastly test the notebook
        '''
        self.do_notebook('test_ga_versus_grid')

    @unittest.skip("Not implemented")
    def test_subset(self):
        from neuronunit.optimization import create_subset
        create_subset(5)

    @unittest.skip("Not implemented")
    def test_update_deap_pop(self):
        from neuronunit.optimization import update_deap_pop

    @unittest.skip("Not implemented")
    def test_dtc_to_rheo(self):
        from neuronunit.optimization import dtc_to_rheo
        dtc_to_rheo(dtc)

    @unittest.skip("Not implemented")
    def test_evaluate(self,dtc):
        from neuronunit.optimization_management import evaluate
        assert dtc.scores is not None
        evauate(dtc)


if __name__ == '__main__':
    unittest.main()
