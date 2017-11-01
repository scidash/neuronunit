"""Tests of NeuronUnit test classes"""
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
        ma = list(dir(model))

        #self.assertTrue('get_spike_train' in method_methods_avail)
        if 'get_spike_train' in ma and 'rheobase' in ma:
            return True
        else:
            return False


    def test_3_backend_inheritance(self):
        boolean = self.backend_inheritance()
        self.assertTrue(boolean)

    def test_4_backend_inheritance_parallel(self):
        booleans = self.backend_inheritance()
        booleanp = dview.apply_sync(self.backend_inheritance)
        self.assertEqual(booleans, booleanp[0])

    def test_5_agreement(self):
        from neuronunit.optimization import nsga_object
        from neuronunit.optimization import nsga_parallel
        from neuronunit.optimization import evaluate_as_module
        import numpy as np
        disagreement = []
        for i in range(1,10):
        #i = 1 #later this will be a loop as comment above.
            subset = nsga_parallel.create_subset(nparams=i)
            numb_err_f = 8
            toolbox, tools, history, creator, base = evaluate_as_module.import_list(ipp,subset,numb_err_f)
            ind = toolbox.population(n = 1)
            print(len(ind),i)

            N = nsga_object.NSGA(nparams=i)
            #self.assertEqual(N.nparams,i)
            N.setnparams(nparams=i)
            #self.assertEqual(N.nparams,i)


            from neuronunit.optimization import exhaustive_search as es
            npoints = 2
            nparams = i
            scores_exh, dtcpop = es.run_grid(npoints,nparams)
            #import pdb; pdb.set_trace()

            minima_attr = dtcpop[np.where[ np.min(scores_exh) == scores_exh ][0]]
            NGEN = 2
            MU = 4
            invalid_dtc, pop, logbook, fitnesses = N.main(MU,NGEN)
            keys = invalid_dtc[0].keys()
            dis = []
            for k in keys:
                dis.append(invalid_dtc[0].attrs[k] - minima_attr.attrs[k])
            disagreement.append(np.mean(dis))
            import pdb; pdb.set_trace()
        return disagreement, dis

    def test_6agreement(self):
        disagreement, dis = self.test5_agreement()


    def test_7ngsa(self):
        from neuronunit.optimization import nsga_object

        import numpy as np
        number = 2
        N = nsga_object.NSGA(nparams=number)
        #self.assertEqual(N.nparams,number)
        number = 2
        N.setnparams(nparams=number)
        #self.assertEqual(N.nparams,number)

        NGEN = 2
        MU = 4

        invalid_dtc, pop, logbook, fitnesses = N.main(MU,NGEN)

        pf = np.mean(fitnesses)
        NGEN = 4
        MU = len(pop)

        for gen in range(1, NGEN):
            final_dtc, pop, final_logbook, final_fitnesses = N.evolve(pop,MU,gen)
        final_pop = pop
        ff = np.mean(final_fitnesses)
        import pdb; pdb.set_trace()

        self.assertNotEqual(ff,pf)
        self.assertGreater(ff,pf)





    def test_8_data_transport_containers_on_bulk(self):
        MU = 10000
        import deap
        import numpy as np
        import copy
        from neuronunit.optimization import evaluate_as_module
        from neuronunit.optimization import model_parameters as modelp
        #scores = []
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

        lists = len(list(pop))
        self.assertEqual(lists,MU)
        # test if the models correspond to unique parameters.
        striped_data_type = []
        for ind in pop:
            disp = []
            for i in ind:
                disp.append(float(i))
            striped_data_type.append(tuple(disp))

        lists = len(list(pop))
        sets = len(set(striped_data_type))
        #self.assertEqual(sets,MU)


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

        # use set to verify the set of attrs is unique.
        self.assertEqual(len(dtcpop), len(pop))
        return dtcpop, pop

    def serial_9rheobase(self):
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
    def test_10ngsa_setup(self):
        dtcpop, pop = self.test_8_data_transport_containers_on_bulk()
        dtcpop = dtcpop[0:9]#.extend(dtcpop[-5:-1])
        pop = pop[0:9]#.extend(pop[-5:-1])

        from neuronunit.optimization import model_parameters as modelp
        from neuronunit.optimization import evaluate_as_module
        from neuronunit.optimization import nsga_parallel
        update_dtc_pop = evaluate_as_module.update_dtc_pop
        pre_format = evaluate_as_module.pre_format
        dtc_to_rheo = nsga_parallel.dtc_to_rheo
        bind_score_to_dtc= nsga_parallel.bind_score_to_dtc
        pre_size = len(dtcpop)
        #dtcpop = update_dtc_pop(pop, td)
        dtcpop = list(map(dtc_to_rheo,dtcpop))
        post_size = len(dtcpop)
        self.assertEqual(pre_size,post_size)

        dtcpop = list(map(pre_format,dtcpop))
        post_size = len(dtcpop)
        self.assertEqual(pre_size,post_size)


        from neuronunit.optimization import get_neab

        for d in dtcpop:
            assert len(d.vtest) == len(get_neab.tests)

        import pdb; pdb.set_trace()

        dtcpop = list(dview.map_sync(bind_score_to_dtc,dtcpop))
        post_size = len(dtcpop)
        self.assertEqual(pre_size,post_size)

        return pop,dtcpop

    def test_10rheobase_setup(self):
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




    def test_12ngsa(self):
        pop,dtcpop = self.test_10ngsa_setup()



    def try_hard_coded0(self):
        params0 = {'C': '0.000107322241995',
        'a': '0.177922330376',
        'b': '-5e-09',
        'c': '-59.5280130394',
        'd': '0.153178745992',
        'k': '0.000879131572692',
        'v0': '-73.3255584633',
        'vpeak': '34.5214177196',
        'vr': '-71.0211905343',
        'vt': '-46.6016774842'}
        #rheobase = {'value': array(131.34765625) * pA}
        return params0



    def try_hard_coded1(self):
        params1 = {'C': '0.000106983591242',
        'a': '0.480856799107',
        'b': '-5e-09',
        'c': '-57.4022276619',
        'd': '0.0818117582621',
        'k': '0.00114004749537',
        'v0': '-58.4899756601',
        'vpeak': '36.6769758895',
        'vr': '-63.4080852004',
        'vt': '-44.1074682812'}
        #rheobase = {'value': array(106.4453125) * pA}131.34765625
        return params1




    def difference(self,observation,prediction): # v is a tesst
        import quantities as pq
        import numpy as np

        # The trick is.
        # prediction always has value. but observation 7 out of 8 times has mean.

        if 'value' in prediction.keys():
            unit_predictions = prediction['value']
            if 'mean' in observation.keys():
                unit_observations = observation['mean']
            elif 'value' in observation.keys():
                unit_observations = observation['value']

        if 'mean' in prediction.keys():
            unit_predictions = prediction['mean']
            if 'mean' in observation.keys():
                unit_observations = observation['mean']
            elif 'value' in observation.keys():
                unit_observations = observation['value']


        to_r_s = unit_observations.units
        unit_predictions = unit_predictions.rescale(to_r_s)
        #unit_observations = unit_observations.rescale(to_r_s)
        unit_delta = np.abs( np.abs(unit_observations)-np.abs(unit_predictions) )

        ##
        # Repurposed from from sciunit/sciunit/scores.py
        # line 156
        ##
        assert type(observation) in [dict,float,int,pq.Quantity]
        assert type(prediction) in [dict,float,int,pq.Quantity]
        ratio = unit_predictions / unit_observations
        unit_delta = np.abs( np.abs(unit_observations)-np.abs(unit_predictions) )
        return unit_delta, ratio

    def run_test(self, cls, pred =None):
        observation = self.get_observation(cls)
        test = cls(observation=observation)
        params0 = self.try_hard_coded0()
        #params1 = self.try_hard_coded1()
        #params = [params0,params1]
        #self.model.prediction =
        self.model.set_attrs(params0)

        score0 = test.judge(self.model,stop_on_error = True, deep_error = True)
        #df, html = self.bar_char_out(score,str(test),params0)
        #self.model.set_attrs(params1)
        #score1 = test.judge(self.model,stop_on_error = True, deep_error = True)
        #score.summarize()
        return [score0,score1]
    def test_13inputresistance(self):
        #from neuronunit.optimization import data_transport_container

        from neuronunit.tests.passive import InputResistanceTest as T
        score = self.run_test(T)
        #self.assertTrue(-0.6 < score < -0.5)

    def test_14restingpotential(self):
        #from neuronunit.optimization import data_transport_container

        from neuronunit.tests.passive import RestingPotentialTest as T
        score = self.run_test(T)
        #self.assertTrue(1.2 < score < 1.3)

    def test_15capacitance(self):
        #from neuronunit.optimization import data_transport_container

        from neuronunit.tests.passive import CapacitanceTest as T
        score = self.run_test(T)
        #self.assertTrue(-0.15 < score < -0.05)
    '''
    @skip_run #known to fail.
    def test_16timeconstant(self):
        #from neuronunit.optimization import data_transport_container

        from neuronunit.tests.passive import TimeConstantTest as T
        score = self.run_test(T)
        #self.assertTrue(-1.45 < score < -1.35)
    '''

    def test_17rheobase_parallel(self):
        #from neuronunit.optimization import data_transport_container

        from neuronunit.tests.fi import RheobaseTest as T
        #super(TestsWaveformTestCase,self).prediction = score.prediction
        self.model.prediction = score.prediction
        #pred =
        score = self.run_test(T)
        #self.prediction = score.prediction
        print(self.model.prediction, 'is prediction being updated properly?')

        self.assertTrue( score.prediction['value'] == 106.4453125 or score.prediction['value'] ==131.34765625)

    def test_18rheobase_serial(self):
        from neuronunit.optimization import data_transport_container

        from neuronunit.tests.fi import RheobaseTest as T
        score = self.run_test(T)
        super(TestsWaveformTestCase,self).prediction = score.prediction
        #self.prediction = score.prediction
        self.assertTrue( int(score.prediction['value']) == int(106) or int(score.prediction['value']) == int(131))


    def update_amplitude(self,test):
        rheobase = self.prediction['value']#first find a value for rheobase
        test.params['injected_square_current']['amplitude'] = rheobase * 1.01



    def test_19ap_width(self):
        #from neuronunit.optimization import data_transport_container

        from neuronunit.tests.waveform import InjectedCurrentAPWidthTest as T
        self.update_amplitude(T)
        score = self.run_test(T)
        #self.assertTrue(-0.6 < score < -0.5)

    def test_20ap_amplitude(self):
        #from neuronunit.optimization import data_transport_container
        from neuronunit.tests.waveform import InjectedCurrentAPAmplitudeTest as T

        self.update_amplitude(T)

        score = self.run_test(T)
        #self.assertTrue(-1.7 < score < -1.6)

    def test_21ap_threshold(self):
        #from neuronunit.optimization import data_transport_container

        from neuronunit.tests.waveform import InjectedCurrentAPThresholdTest as T

        self.update_amplitude(T)

        score = self.run_test(T)
        #self.assertTrue(2.25 < score < 2.35)


if __name__ == '__main__':
    unittest.main()
