"""Tests of NeuronUnit test classes"""
import unittest
import os
import sys
from sciunit.utils import NotebookTools
import dask
from dask import bag
import dask.bag as db
import matplotlib as mpl
mpl.use('Agg') # Avoid any problems with Macs or headless displays.

from sciunit.utils import NotebookTools,import_all_modules
from neuronunit import neuroelectro,bbp,aibs

from base import *


def grid_points():
    from neuronunit.optimization.optimization_management import map_wrapper

    npoints = 2
    nparams = 10
    from neuronunit.optimization.model_parameters import model_params
    provided_keys = list(model_params.keys())
    USE_CACHED_GS = False
    from neuronunit.optimization import exhaustive_search
    from neuronunit.optimization.optimization_management import map_wrapper
    ## exhaustive_search

    grid_points = exhaustive_search.create_grid(npoints = npoints,nparams = nparams)
    b0 = db.from_sequence(grid_points[0:2], npartitions=8)
    dtcpop = list(db.map(exhaustive_search.update_dtc_grid,b0).compute())
    assert dtcpop is not None
    return dtcpop

def test_compute_score(dtcpop):
    from neuronunit.optimization.optimization_management import map_wrapper
    from neuronunit.optimization import get_neab
    from neuronunit.optimization.optimization_management import dtc_to_rheo
    from neuronunit.optimization.optimization_management import nunit_evaluation
    from neuronunit.optimization.optimization_management import format_test
    dtclist = list(map(dtc_to_rheo,dtcpop))
    for d in dtclist:
        assert len(list(d.attrs.values())) > 0
    dtclist = map_wrapper(format_test, dtclist)
    dtclist = map_wrapper(nunit_evaluation, dtclist)
    return dtclist

class testOptimizationBackend(NotebookTools,unittest.TestCase):

    def setUp(self):
        self.predictions = None
        self.predictionp = None
        self.score_p = None
        self.score_s = None
        self.grid_points = grid_points()
        dtcpop = self.grid_points
        self.test_compute_score = test_compute_score
        self.dtcpop = test_compute_score(dtcpop)
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


    def test_map_wrapper(self):
        npoints = 2
        nparams = 3
        from neuronunit.optimization.model_parameters import model_params
        provided_keys = list(model_params.keys())
        USE_CACHED_GS = False
        from neuronunit.optimization import exhaustive_search
        from neuronunit.optimization.optimization_management import map_wrapper
        grid_points = exhaustive_search.create_grid(npoints = npoints,nparams = nparams)
        b0 = db.from_sequence(grid_points[0:2], npartitions=8)
        dtcpop = list(db.map(exhaustive_search.update_dtc_grid,b0).compute())
        assert dtcpop is not None
        dtcpop_compare = map_wrapper(exhaustive_search.update_dtc_grid,grid_points[0:2])
        for i,j in enumerate(dtcpop):
            for k,v in dtcpop_compare[i].attrs.items():
                print(k,v,i,j)
                self.assertEqual(j.attrs[k],v)
        return True

    def test_grid_dimensions(self):
        from neuronunit.optimization.model_parameters import model_params
        provided_keys = list(model_params.keys())
        USE_CACHED_GS = False
        from neuronunit.optimization import exhaustive_search
        from neuronunit.optimization.optimization_management import map_wrapper
        import dask.bag as db
        npoints = 2
        nparams = 3
        for i in range(1,10):
            for j in range(1,10):
                grid_points = exhaustive_search.create_grid(npoints = i, nparams = j)
                b0 = db.from_sequence(grid_points[0:2], npartitions=8)
                dtcpop = list(db.map(exhaustive_search.update_dtc_grid,b0).compute())
                self.assertEqual(i*j,len(dtcpop))
                self.assertNotEqual(dtcpop,None)
                dtcpop_compare = map_wrapper(exhaustive_search.update_dtc_grid,grid_points[0:2])
                self.assertNotEqual(dtcpop_compare,None)
                self.assertEqual(len(dtcpop_compare),len(dtcpop))
                for i,j in enumerate(dtcpop):
                    for k,v in dtcpop_compare[i].attrs.items():
                        print(k,v,i,j)
                        self.assertEqual(j.attrs[k],v)

        return True


    def test_get_rate_CV(self):
        from neuronunit.tests import dynamics
        import quantities as pq
        doi = 'doi:10.1113/jphysiol.2010.200683'
        observation = {}
        observation[doi] = {}
        observation[doi]['isi'] = 598.0*pq.ms
        observation[doi]['mean'] = 598.0*pq.ms
        observation[doi]['std'] = 37.0*pq.ms
        isi_test = dynamics.ISITest(observation = observation[doi])
        # Put them together in a test suite

        observation = {}
        observation[doi] = {}
        observation[doi]['cv'] = 0.210526*pq.dimensionless
        observation[doi]['mean'] = 0.210526*pq.dimensionless
        observation[doi]['std'] = 0.105263*pq.dimensionless
        #si_test = dynamics.ISITest(observation = observation[doi])

        cv_test = dynamics.CVTest(observation = observation[doi])
        ap_tests = sciunit.TestSuite([isi_test,cv_test], name="AP CV and ISI Tests")

        from neuronunit.models import static_model
        true_model = self.model
        params = {}
        params['injected_square_current'] = {}
        params['injected_square_current']['amplitude'] = self.rheobase*2.01
        model.inject_square_current(params)
        vm = model.get_membrane_potential()
        proxy_model = static_model.StaticModel(vm = vm, st = None)
        score_matrix = ap_tests.judge(proxy_model)
        for score in score_matrix:
            self.assertNotEqual(score,None)


        # Put them together in a test suite
        #ap_tests = sciunit.TestSuite([ap_width_test,ap_amplitude_test], name="AP Tests")


    #@unittest.skip("Not fully developed yet")
    def test_get_inhibitory_neuron(self):
        from neuronunit import tests as nu_tests, neuroelectro
        from neuronunit.tests import passive, waveform, fi
        fi_basket = {'nlex_id':'NLXCELL:100201'}
        observation =  fi.neuroelectro_summary_observation(fi_basket)
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
        model = ReducedModel(get_neab.LEMS_MODEL_PATH, name= str('vanilla'), backend=('NEURON', {'DTC':dtc}))
        temp = [ v for v in model.attrs.values() ]
        assert len(temp) > 0
        self.assertGreater(len(temp),0)
        rbt = get_neab.tests[0]
        scoreN = rbt.judge(model,stop_on_error = False, deep_error = True)
        import copy
        dtc.scores[str(rbt)] = copy.copy(scoreN.norm_score)
        assert scoreN.norm_score is not None
        self.assertTrue(scoreN.norm_score is not None)
        dtc.rheobase = copy.copy(scoreN.prediction)
        return dtc

    def test_rheobase_on_list(self):
        from neuronunit.optimization import exhaustive_search
        grid_points = self.grid_points
        second_point = grid_points[int(len(grid_points)/2)]
        three_points = [grid_points[0],second_point,grid_points[-1]]
        self.assertEqual(len(three_points),3)
        dtcpop = list(map(exhaustive_search.update_dtc_grid,three_points))
        for d in self.dtcpop:
            assert len(list(d.attrs.values())) > 0
        dtcpop = self.test_compute_score(self.dtcpop)
        self.dtcpop = dtcpop
        return dtcpop

    def test_neuron_set_attrs(self):
        from neuronunit.models.reduced import ReducedModel
        from neuronunit.optimization import get_neab
        self.assertNotEqual(self.dtcpop,None)
        dtc = self.dtcpop[0]
        self.model = ReducedModel(get_neab.LEMS_MODEL_PATH, backend=('NEURON',{'DTC':dtc}))
        temp = [ v for v in self.model.attrs.values() ]
        assert len(temp) > 0
        self.assertGreater(len(temp),0)

    def test_rheobase_serial(self):
        from neuronunit.optimization import data_transport_container
        from neuronunit.tests.fi import RheobaseTest as T
        score = self.run_test(T)
        self.rheobase = score.prediction
        self.assertNotEqual(self.rheobase,None)


    def test_inputresistance(self):
        from neuronunit.tests.passive import InputResistanceTest as T
        score = self.run_test(T)
        print(score)
        print(score.norm_score)
        assert score.norm_score is not None
        self.assertTrue(score.norm_score is not None)

        self.assertTrue(-0.6 < score < -0.5)

    def test_restingpotential(self):
        from neuronunit.tests.passive import RestingPotentialTest as T
        score = self.run_test(T)
        print(score)
        print(score.norm_score)
        assert score.norm_score is not None
        self.assertTrue(score.norm_score is not None)
        self.assertTrue(1.2 < score < 1.3)


    def test_capacitance(self):
        from neuronunit.tests.passive import CapacitanceTest as T
        score = self.run_test(T)
        print(score)
        print(score.norm_score)
        assert score.norm_score is not None
        self.assertTrue(score.norm_score is not None)

        self.assertTrue(-0.15 < score < -0.05)

    def test_timeconstant(self):
        from neuronunit.tests.passive import TimeConstantTest as T
        score = self.run_test(T)
        print(score)
        print(score.norm_score)
        print(score)
        assert score.norm_score is not None
        self.assertTrue(score.norm_score is not None)

        self.assertTrue(-1.45 < score < -1.35)



    def test_ap_width(self):
        from neuronunit.models.reduced import ReducedModel
        from neuronunit.optimization import get_neab
        from neuronunit.tests.waveform import InjectedCurrentAPWidthTest as T

        #self.update_amplitude(T)
        score = self.run_test(T,pred=self.rheobase)
        assert score.norm_score is not None
        self.assertTrue(score.norm_score is not None)
        self.assertTrue(-0.6 < score < -0.5)

    def test_ap_amplitude(self):
        from neuronunit.models.reduced import ReducedModel
        from neuronunit.optimization import get_neab
        from neuronunit.tests.waveform import InjectedCurrentAPAmplitudeTest as T
        score = self.run_test(T,pred=self.rheobase)
        assert score.norm_score is not None
        self.assertTrue(score.norm_score is not None)
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
        assert score.norm_score is not None
        self.assertTrue(score.norm_score is not None)




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
        self.score_p = self.score_p.norm_score
        #model = ReducedModel(get_neab.LEMS_MODEL_PATH,name=str('vanilla'),backend='NEURON')

        serial_model = ReducedModel(get_neab.LEMS_MODEL_PATH,name=str('vanilla'),backend='NEURON')
        self.score_s = rt.judge(serial_model,stop_on_error = False, deep_error = True)
        self.predictions = self.score_s.prediction
        self.score_s = self.score_s.norm_score
        import numpy as np
        check_less_thresh = float(np.abs(self.predictionp['value'] - self.predictions['value']))
        self.assertLessEqual(check_less_thresh, 2.0)

    def test_set_model_attrs(self):
        from neuronunit.optimization.model_parameters import model_params
        provided_keys = list(model_params.keys())
        from bluepyopt.deapext.optimisations import DEAPOptimisation
        DO = DEAPOptimisation()
        for i in range(1,10):
            for j in range(1,10):
                provided_keys = list(model_params.keys())[j]
                DO.setnparams(nparams = i, provided_keys = provided_keys)

    def test_opt_set_GA_params(self):
        '''
        Test to check if making the population size bigger, or increasing the run_number
        of generations the GA consumes actually increases the fitness.

        Note only test for better fitness every 3rd count, because GAs are stochastic,
        and by chance occassionally fitness may not increase, just because the GA has
        more genes to sample with.

        A fairer test of the GA would probably test if the mean of mean fitness is improving
        or use a sliding window.
        '''
        from neuronunit.optimization.model_parameters import model_params
        provided_keys = list(model_params.keys())
        from bluepyopt.deapext.optimisations import DEAPOptimisation
        DO = DEAPOptimisation()
        nparams = 5
        cnt = 0
        provided_keys = list(model_params.keys())[nparams]
        DO.setnparams(nparams = nparams, provided_keys = provided_keys)
        list_check_increase = []
        for NGEN in range(1,30):
            for MU in range(5,40):
                pop, hof, log, history, td, gen_vs_hof = DO.run(offspring_size = MU, max_ngen = NGEN, cp_frequency=4,cp_filename='checkpointedGA.p')
                avgf = np.mean([ h.fitness for h in hof ])
                list_check_increase.append(avgf)

                if cnt % 3 == 0:
                    avgf = np.mean([ h.fitness for h in hof ])
                    old_avg = np.mean(list_check_increase)
                    self.assertGreater(np.mean(list_check_increase),old_avg)
                    self.assertGreater(avgf,old)

                    old = avgf
                elif cnt == 0:
                    avgf = np.mean([ h.fitness for h in hof ])
                    old = avgf
                    old_avg = np.mean(list_check_increase)



                cnt += 1
                print(old,cnt)


    def test_frp(self):
        from neuronunit.models.reduced import ReducedModel
        from neuronunit.optimization import get_neab
        model = ReducedModel(get_neab.LEMS_MODEL_PATH,name=str('vanilla'),backend=('NEURON'))
        attrs = {'a':0.02, 'b':0.2, 'c':-65+15*0.5, 'd':8-0.5**2 }
        from neuronunit.optimization.data_transport_container import DataTC
        dtc = DataTC()
        from neuronunit.tests import fi
        model.set_attrs(attrs)
        from neuronunit.optimization import get_neab
        rtp = get_neab.tests[0]
        rheobase = rtp.generate_prediction(model)
        self.assertTrue(float(rheobase['value']))


    def test_parallelnb(self):
        '''
        Lastly test the notebook
        '''
        self.do_notebook('test_ga_versus_grid')


    """Testing  notebooks"""
    @unittest.skip("takes too long")

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
