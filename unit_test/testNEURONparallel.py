"""Tests of NeuronUnit test classes"""


#from .base import *
import unittest
#import os
#os.system('ipcluster start -n 8 --profile=default & sleep 5;')
import ipyparallel as ipp
rc = ipp.Client(profile='default')
rc[:].use_cloudpickle()
dview = rc[:]

class TestBackend(unittest.TestCase):


    def setUp(self):
        self.predictions = None
        self.predictionp = None
        self.score_p = None
        self.score_s = None


    def get_observation(self, cls):
        print(cls.__name__)
        neuron = {'nlex_id': 'nifext_50'} # Layer V pyramidal cell
        return cls.neuroelectro_summary_observation(neuron)

    def test_0import(self):
        import ipyparallel as ipp
        return True

    def check_parallel_path_consistency(self):
        '''
        import paths and test for consistency
        '''
        from neuronunit import models
        return models.__file__

    def test_1_check_paths(self):
        path_serial = self.check_parallel_path_consistency()
        paths_parallel = dview.apply_async(self.check_parallel_path_consistency).get_dict()
        self.assertEqual(path_serial, paths_parallel[0])

    def backend_inheritance(self):
        from neuronunit.models.reduced import ReducedModel
        from neuronunit.optimization import get_neab
        print(get_neab.LEMS_MODEL_PATH)
        model = ReducedModel(get_neab.LEMS_MODEL_PATH, backend='NEURON')
        method_methods_avail = list(dir(model))
        self.assertTrue('get_spike_train' in method_methods_avail)
        if bool('get_spike_train' in method_methods_avail) == True:
            return True
        else:
            return False


    def test_3_backend_inheritance(self):
        boolean = self.backend_inheritance()
        self.assertTrue(boolean)

    def test_4_backend_inheritance_parallel(self):
        booleans = self.backend_inheritance()

        booleanp = dview.apply_sync(self.backend_inheritance)#.get()#.get_dict()
        #print(len(booleans))
        self.assertEqual(booleans, booleanp[0])


    def test_5ngsa(self):
        from neuronunit.optimization import nsga_object

        #N = nsga_object.NSGA(10,10,10)
        #invalid_dtc, pop, logbook, fitnesses = N.main(4,2)
        import numpy as np
        N = nsga_object.NSGA()
        NGEN = 3
        MU = 4

        invalid_dtc, pop, logbook, fitnesses = N.main(MU,NGEN)
        pf = np.mean(fitnesses)
        NGEN = 3
        MU = len(pop)

        for gen in range(1, NGEN):
            final_dtc, pop, final_logbook, final_fitnesses = N.evolve(pop,MU,gen)
        final_pop = pop
        ff = np.mean(final_fitnesses)
        import pdb; pdb.set_trace()

        self.assertNotEqual(ff,pf)
        self.assertGreater(ff,pf)


    def test_6_data_transport_containers_on_bulk(self):
        MU = 10000
        import deap
        import numpy as np
        import copy
        from neuronunit.optimization import evaluate_as_module
        #from neuronunit.optimization import model_parameters
        from neuronunit.optimization import model_parameters as modelp

        #import pdb; pdb.set_trace()
        scores = []
        from neuronunit.optimization import nsga_parallel
        subset = nsga_parallel.create_subset(nparams=10)
        numb_err_f = 8
        toolbox, tools, history, creator, base = evaluate_as_module.import_list(ipp,subset,numb_err_f)
        dview.push({'Individual':evaluate_as_module.Individual})
        dview.apply_sync(evaluate_as_module.import_list,ipp,subset,numb_err_f)
        get_trans_dict = evaluate_as_module.get_trans_dict
        td = get_trans_dict(subset)
        dview.push({'td':td })
        pop = toolbox.population(n = MU)
        pop = [ toolbox.clone(i) for i in pop ]
        dview.scatter('Individual',pop)
        update_dtc_pop = evaluate_as_module.update_dtc_pop
        pre_format = evaluate_as_module.pre_format
        dtcpop = update_dtc_pop(pop, td)
        self.dtcpop = dtcpop
        self.pop = pop
        for index, dtc in enumerate(dtcpop):
            if index!=0:
                self.assertNotEqual(old_attrs, dtc.attrs)
                print(old_attrs,dtc.attrs)
            old_attrs = dtc.attrs

        for index, dtc in enumerate(dtcpop):
            value = dtc.attrs['b']
            boolean = np.isnan(float(value))
            self.assertEqual(boolean, False)
            print(boolean)

            #import pdb; pdb.set_trace()
        # use set to verify the set of attrs is unique.
        self.assertEqual(len(dtcpop), len(pop))
        return dtcpop, pop

    def serial_7rheobase(self):
        from neuronunit.tests.fi import RheobaseTest, RheobaseTestP
        from neuronunit.optimization import get_neab
        from neuronunit.models.reduced import ReducedModel
        from neuronunit import aibs
        import os
        dataset_id = 354190013  # Internal ID that AIBS uses for a particular Scnn1a-Tg2-Cre
                                # Primary visual area, layer 5 neuron.
        observation = aibs.get_observation(dataset_id,'rheobase')
        rt = RheobaseTest(observation = observation)
        model = ReducedModel(get_neab.LEMS_MODEL_PATH,name=str('vanilla'),backend='NEURON')
        self.score_s = rt.judge(model,stop_on_error = True, deep_error = True)
        self.predictions = self.score_s.prediction
        self.score_s = self.score_s.sort_key
        #print(' serial score {0} parallel score {1}'.format(self.score_s,self.score_p))

        self.assertNotEqual(type(self.scores_s),type(None))
        #self.assertEqual(int(self.score_s*1000), int(self.score_p*1000))
        #self.assertEqual(int(self.predictionp['value']), int(self.predictions['value']))


    def test_8rheobase_setup(self):
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
        self.score_s = rt.judge(model,stop_on_error = True, deep_error = True)
        self.predictions = self.score_s.prediction
        self.score_s = self.score_s.sort_key
        print(' serial score {0} parallel score {1}'.format(self.score_s,self.score_p))
        self.assertEqual(int(self.score_s*1000), int(self.score_p*1000))
        self.assertEqual(int(self.predictionp['value']), int(self.predictions['value']))

    def test_9ngsa_setup(self):
        dtcpop, pop = self.test_6_data_transport_containers_on_bulk()
        dtcpop = dtcpop[0:9]#.extend(dtcpop[-5:-1])
        pop = pop[0:9]#.extend(pop[-5:-1])
        #dtcpop =
        #pop = pop[-5:-1]
        #dtcpop = nsga_parallel.dtc_to_rheo(dtcpop)
        from neuronunit.optimization import model_parameters as modelp
        from neuronunit.optimization import evaluate_as_module
        from neuronunit.optimization import nsga_parallel
        update_dtc_pop = evaluate_as_module.update_dtc_pop
        pre_format = evaluate_as_module.pre_format
        dtc_to_rheo = nsga_parallel.dtc_to_rheo
        map_wrapper = nsga_parallel.map_wrapper
        pre_size = len(dtcpop)
        #dtcpop = update_dtc_pop(pop, td)
        dtcpop = list(map(dtc_to_rheo,dtcpop))
        post_size = len(dtcpop)
        self.assertEqual(pre_size,post_size)

        dtcpop = list(map(pre_format,dtcpop))
        post_size = len(dtcpop)
        self.assertEqual(pre_size,post_size)

        dtcpop = list(dview.map_sync(map_wrapper,dtcpop))
        post_size = len(dtcpop)
        self.assertEqual(pre_size,post_size)

        return pop,dtcpop


    def test_10ngsa(self):
        pop,dtcpop = self.test_8ngsa_setup()

    def test_11(self):
        from neuronunit.optimization import nsga_parallel
        from neuronunit.optimization import evaluate_as_module

        for i in range(0,10):
            subset = nsga_parallel.create_subset(nparams=i)
            numb_err_f = 8
            toolbox, tools, history, creator, base = evaluate_as_module.import_list(ipp,subset,numb_err_f)
            ind = toolbox.population(n = 1)
            self.assertEqual(len(ind)+1,i)
            #pop = [ toolbox.clone(i) for i in pop ]





if __name__ == '__main__':
    unittest.main()
    '''
    #import pickle
    #with open('opt_run_data','rb') as handle:
    #    valued = pickle.load(handle)
    #dtcpop, _, _ = valued
    #self.dtcpop = dtcpop
    def run_test(self, cls):
        observation = self.get_observation(cls)
        test = cls(observation=observation)
        #for d in dtcpop:
        #    print(d,d.attrs)
        #    self.model.set_attrs(d.attrs)
        score = test.judge(self.model,stop_on_error = True, deep_error = True)

        score.summarize()
        return score.score
    '''


'''
class TestsPassiveTestCase(TestsTestCase, unittest.TestCase):
    """Test passive validation tests"""

    def test_inputresistance(self):
        from neuronunit.tests.passive import InputResistanceTest as T
        score = self.run_test(T)
        #self.assertTrue(-0.6 < score < -0.5)

    def test_restingpotential(self):
        from neuronunit.tests.passive import RestingPotentialTest as T
        score = self.run_test(T)
        #self.assertTrue(1.2 < score < 1.3)

    def test_capacitance(self):
        from neuronunit.tests.passive import CapacitanceTest as T
        score = self.run_test(T)
        #self.assertTrue(-0.15 < score < -0.05)

    def test_timeconstant(self):
        from neuronunit.tests.passive import TimeConstantTest as T
        score = self.run_test(T)
        #self.assertTrue(-1.45 < score < -1.35)


class TestsWaveformTestCase(TestsTestCase, unittest.TestCase):
    """Test passive validation tests"""

    def test_ap_width(self):
        from neuronunit.tests.waveform import InjectedCurrentAPWidthTest as T
        score = self.run_test(T)
        #self.assertTrue(-0.6 < score < -0.5)

    def test_ap_amplitude(self):
        from neuronunit.tests.waveform import InjectedCurrentAPAmplitudeTest as T
        score = self.run_test(T)
        #self.assertTrue(-1.7 < score < -1.6)

    def test_ap_threshold(self):
        from neuronunit.tests.waveform import InjectedCurrentAPThresholdTest as T
        score = self.run_test(T)
        #self.assertTrue(2.25 < score < 2.35)


class TestsFITestCase(TestsTestCase, unittest.TestCase):
    """Test F/I validation tests"""

    #@unittest.skip("This test takes a long time")
    def test_rheobase_serial(self):
        from neuronunit.tests.fi import T
        score = self.run_test(T)
        #self.assertTrue(0.2 < score < 0.3)

    #@unittest.skip("This test takes a long time")
    def test_rheobase_parallel(self):
        from neuronunit.tests.fi import T
        score = self.run_test(T)
        #self.assertTrue(0.2 < score < 0.3)


class TestsDynamicsTestCase(TestsTestCase, unittest.TestCase):
    """Tests dynamical systems properties tests"""

    @unittest.skip("This test is not yet implemented")
    def test_threshold_firing(self):
        from neuronunit.tests.dynamics import TFRTypeTest as T
        #score = self.run_test(T)
        #self.assertTrue(0.2 < score < 0.3)

    @unittest.skip("This test is not yet implemented")
    def test_rheobase_parallel(self):
        from neuronunit.tests.dynamics import BurstinessTest as T
        #score = self.run_test(T)
        #self.assertTrue(0.2 < score < 0.3)


class TestsChannelTestCase(unittest.TestCase):
    @unittest.skip("This test is not yet implemented")
    def test_iv_curve_ss(self):
        from neuronunit.tests.channel import IVCurveSSTest as T
        #score = self.run_test(T)
        #self.assertTrue(0.2 < score < 0.3)

    @unittest.skip("This test is not yet implemented")
    def test_iv_curve_peak(self):
        from neuronunit.tests.channel import IVCurvePeakTest as T
        #score = self.run_test(T)
        #self.assertTrue(0.2 < score < 0.3)
'''