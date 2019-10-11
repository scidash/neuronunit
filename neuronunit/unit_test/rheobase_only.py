"""Tests of NeuronUnit test classes"""
import unittest
import os
import sys
from sciunit.utils import NotebookTools#,import_all_modules
import dask
from dask import bag
import matplotlib
try:
    matplotlib.use('Agg')
except:
    pass

import quantities as pq
import copy
import unittest
from neuronunit.optimisation.optimization_management import dtc_to_rheo, mint_generic_model
from neuronunit.optimisation.optimization_management import OptMan
from neuronunit.tests import passive, waveform, fi
from neuronunit.optimisation import exhaustive_search
from neuronunit.optimisation.model_parameters import MODEL_PARAMS
from neuronunit.tests import dynamics
import pickle
import dask.bag as db
from neuronunit.optimisation import data_transport_container
import numpy as np
from neuronunit.tests.fi import RheobaseTest, RheobaseTestP
from neuronunit.optimisation import get_neab
from neuronunit.models.reduced import ReducedModel, VeryReducedModel
from neuronunit import aibs
import os
from sciunit.models.runnable import RunnableModel

def grid_points():
    npoints = 10
    nparams = 10
    free_params = MODEL_PARAMS[str('RAW')]
    USE_CACHED_GS = False
    grid_points = exhaustive_search.create_grid(npoints = npoints,free_params=free_params)
    dtcpop = []
    for g in list(grid_points):
        dtc = data_transport_container.DataTC()
        dtc.attrs = g
        dtcpop.append(dtc)
        #print(dtc.attrs)
    return dtcpop
   

def test_rheobase_dtc(dtcpop, tests):
    rheobase_test = tests[0][0][0]

    for d in dtcpop:
        d.tests = rheobase_test
        d.backend = str('RAW')

    dtcpop = list(map(dtc_to_rheo,dtcpop[0:2]))
    return dtcpop

class testLowLevelOptimisation(unittest.TestCase):

    def setUp(self):

        self.predictions = None
        self.predictionp = None
        self.score_p = None
        self.score_s = None
        self.grid_points = grid_points()
        dtcpop = [self.grid_points[0],self.grid_points[1],self.grid_points[-1]]

        electro_path = 'pipe_tests.p'
        assert os.path.isfile(electro_path) == True
        with open(electro_path,'rb') as f:
            self.electro_tests = pickle.load(f)
        #self.electro_tests = get_neab.replace_zero_std(self.electro_tests)

        self.OM = OptMan([self.electro_tests[0][0][0]])

        self.format_test = self.OM.format_test
        self.elephant_evaluation = self.OM.elephant_evaluation
        self.test_rheobase_dtc = test_rheobase_dtc
        self.dtcpop = test_rheobase_dtc(dtcpop,self.electro_tests)
        self.dtc = self.dtcpop[0]
        self.rheobase = self.dtc.rheobase
        self.standard_model = self.model = mint_generic_model('RAW')
        self.MODEL_PARAMS = MODEL_PARAMS
        #
        # NEURON backend broken this branch (barcelona), should work when
        # merged
        self.MODEL_PARAMS.pop(str('NEURON'),None)

        self.heavy_backends = [
                    str('PYNNBackend'),
                    str('NEURONBackend'),
                    str('jNeuroMLBackend')
                ]
        self.light_backends = [
                    str('RAWBackend'),
                    str('HHBackend'),
                    str('ADEXPBackend')
                ]
        self.medium_backends = [
                    str('GLIFBackend')
                ]
        #self.standard_model = ReducedModel(get_neab.LEMS_MODEL_PATH, backend='RAW')
        #self.model = ReducedModel(get_neab.LEMS_MODEL_PATH, backend='RAW')


        purkinje ={"id": 18, "name": "Cerebellum Purkinje cell", "neuron_db_id": 271, "nlex_id": "sao471801888"}
        fi_basket = {"id": 65, "name": "Dentate gyrus basket cell", "neuron_db_id": None, "nlex_id": "nlx_cell_100201"}
        pvis_cortex = {"id": 111, "name": "Neocortex pyramidal cell layer 5-6", "neuron_db_id": 265, "nlex_id": "nifext_50"}
        #does not have rheobase
        olf_mitral = {"id": 129, "name": "Olfactory bulb (main) mitral cell", "neuron_db_id": 267, "nlex_id": "nlx_anat_100201"}
        ca1_pyr = {"id": 85, "name": "Hippocampus CA1 pyramidal cell", "neuron_db_id": 258, "nlex_id": "sao830368389"}
        pipe = [ fi_basket, ca1_pyr, purkinje,  pvis_cortex,olf_mitral]
        self.pipe = pipe


    def check_dif(pipe_old,pipe_new):
        bool = False
        for key, value in pipe_results.items():
            if value != pipe_new[key]:
                bool = True
            print(value,pipe_new[key])

        return bool

    def test_backend_inheritance(self):
        ma = mint_generic_model('RAW')
        if 'get_spike_train' in ma:
            self.asserTrue(True)
        else:
            self.asserTrue(False)
        return

    def data_driven_druckmann(self):
        '''
        can we use neuro electro data to instance druckmann
        on a handle ful of applicable druckmann?
        '''
        for p in pipe:
            tests,observations = get_neab.executable_druckman_tests(p)


    def test_all_tests_pop(self,dtcpop, tests):
        rheobase_test = tests[0][0][0]
        all_tests = tests[0][0]
        for d in dtcpop:
            d.tests = rheobase_test
            d.backend = str('RAW')
            assert len(list(d.attrs.values())) > 0

        dtcpop = list(map(self.dtc_to_rheo,dtcpop))


        for d in dtcpop:
            d.tests = all_tests
            d.backend = str('RAW')
            assert len(list(d.attrs.values())) > 0
                    
        b0 = db.from_sequence(dtcpop, npartitions=8)
        dtcpop = list(db.map(self.format_test,b0).compute())
        
        b0 = db.from_sequence(dtcpop, npartitions=8)
        dtcpop = list(db.map(self.elephant_evaluation,b0).compute())
        return dtcpop

        

    def test_neuron_set_attrs(self):
        self.assertNotEqual(self.dtcpop,None)
        dtc = self.dtcpop[0]
        self.model = RunnableModel(str(dtc.backend),backend=('RAWBackend', {'DTC':dtc}))

        #self.model = ReducedModel(get_neab.LEMS_MODEL_PATH, backend=('RAWBackend', {'DTC':dtc})) #backend=('RAW'{'DTC':dtc}))
        temp = [ v for v in self.model.attrs.values() ]
        assert len(temp) > 0
        self.AssertGreater(temp,0)
        old_ = self.model.attrs.items()
        assert self.model.attrs.keys() in old_
        assert self.model.attrs.values() in old_

    def test_rotate_backends(self):

        all_backends = [
            str('RAWBackend'),
            str('HHBackend'),
            str('GLIFBackend'),
            str('ADEXPBackend')
        ]

        for b in all_backends:
            model = mint_generic_model(b)
            #assert model is not None
            self.assertTrue(model is not None)

            td = model.attrs
            dtc = update_dtc_pop(pop, td)
            inject_and_plot(dtc)
            self.assertTrue(dtc is not None)

            #assert dtc is not None

        MBEs = list(self.MODEL_PARAMS.keys())
        for b in MBEs:
            model = mint_generic_model(b)
            #assert model is not None
            self.assertTrue(model is not None)

            td = model.attrs
            dtc = update_dtc_pop(pop, td)
            inject_and_plot(dtc)
            #assert dtc is not None
            self.assertTrue(dtc is not None)

        return
    @unittest.skip("done elsewhere")
    def test_solution_quality(self):
        '''
        Select random points in parameter space,
        pretend these points are from experimental observations, by coding them in
        as NeuroElectro observations.
        This effectively redefines the sampled point as a the global minimum of error.
        Show that the optimiser can find this point, only using information obtained by
        sparesely learning the error surface.

        '''
        MBEs = list(self.MODEL_PARAMS.keys())
        with open(self.electro_path,'rb') as f:
            (obs_frame,test_frame) = pickle.load(f)
        for key, use_test in test_frame.items():
            for b in MBEs:
                boolean = round_trip_test(use_test,b)
                self.assertTrue(boolean)
        return
    

    def test_get_druckmann(self):
        '''
        test the extraction of Druckmann property Ephys measurements.
        '''
        self.dtcpop = self.test_all_tests_pop(self.dtcpop[0:3],self.electro_tests)
    
        (self.dtcpop,dm_properties) = add_dm_properties_to_cells(self.dtcpop[0:3])
        for d in dm_properties:
            #print(d)
            self.assertTrue(d is not None)
        return

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
    #@unittest.skip("Not fully developed yet")

    def test_get_rate_CV(self):
        # Dictionary of observations, in this case two ephys properties from one paper
        doi = 'doi:10.1113/jphysiol.2010.200683'
        observations={doi:{'ap_amplitude':{'mean':45.1*pq.mV,
                                           'sem':0.7*pq.mV,
                                           'n':25},
                           'ap_width':{'mean':19.7*pq.ms,
                                       'sem':1.0*pq.ms,
                                       'n':25}}}

        # Instantiate two tests based on these properties
        from neuronunit.tests import APWidthTest, APAmplitudeTest
        pdb.set_trace()
        ap_width_test = APWidthTest(observation=observations[doi]['ap_width'])
        ap_amplitude_test = APAmplitudeTest(observation=observations[doi]['ap_amplitude'])

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
        dtc = copy.copy(self.dtc)
        self.assertNotEqual(self.dtc,None)
        dtc.scores = {}
        size = len([ v for v in dtc.attrs.values()])
        assert size > 0
        self.assertGreater(size,0)
        model = VeryReducedModel(name= str('vanilla'), backend=('RAWBackend', {'DTC':dtc}))
        temp = [ v for v in model.attrs.values() ]
        assert len(temp) > 0
        self.assertGreater(len(temp),0)
        rbt = get_neab.tests[0]
        scoreN = rbt.judge(model,stop_on_error = False, deep_error = True)
        dtc.scores[str(rbt)] = copy.copy(scoreN.sort_key)
        assert scoreN.sort_key is not None
        self.assertTrue(scoreN.sort_key is not None)
        dtc.rheobase = copy.copy(scoreN.prediction)
        return dtc


    def test_rheobase_on_list(self):
        grid_points = self.grid_points
        second_point = grid_points[int(len(grid_points)/2)]
        three_points = [grid_points[0],second_point,grid_points[-1]]
        self.assertEqual(len(three_points),3)
        dtcpop = list(map(exhaustive_search.update_dtc_grid,three_points))
        for d in self.dtcpop:
            assert len(list(d.attrs.values())) > 0
        #dtcpop = self.test_rheobase_dtc(self.dtcpop)
        dtcpop = test_rheobase_dtc(dtcpop,self.electro_tests)

        self.dtcpop = dtcpop
        return dtcpop

    def test_neuron_set_attrs(self):
        self.assertNotEqual(self.dtcpop,None)
        dtc = self.dtcpop[0]
        self.model = VeryReducedModel(backend=('RAWBackend', {'DTC':dtc})) #backend=('RAW'{'DTC':dtc}))
        temp = [ v for v in self.model.attrs.values() ]
        assert len(temp) > 0
        self.AssertGreater(temp,0)
        old_ = self.model.attrs.items()
        assert self.model.attrs.keys() in old_
        assert self.model.attrs.values() in old_

    def test_etest(self):
        model = dtcpop[0].dtc_to_model()
        dtc = dtcpop[0]
        et = ETest(model,dtc)
        dtc = ETest.elephant_tests(dtc)
        res = et.runTest()
        #import pdb
        #pdb.set_trace()

    def test_rheobase_serial(self):
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

        from neuronunit.tests.waveform import InjectedCurrentAPWidthTest as T

        self.update_amplitude(T)
        score = self.run_test(T,pred=self.rheobase)
        self.assertTrue(-0.6 < score < -0.5)

    def test_ap_amplitude(self):

        from neuronunit.tests.waveform import InjectedCurrentAPAmplitudeTest as T
        #from neuronunit.optimisation.optimisation_management import format_test
        #from neuronunit.optimisation import data_transport_container
        dtc = data_transport_container.DataTC()
        dtc.rheobase = self.rheobase
        #def run_test(self, cls, pred =None):
        dtc = self.format_test(dtc)
        self.model = VeryReducedModel(get_neab.LEMS_MODEL_PATH, backend=('NEURON',{'DTC':dtc}))

        score = self.run_test(T,pred=self.rheobase)
        self.assertTrue(-1.7 < score < -1.6)

    def test_ap_threshold(self):

        from neuronunit.tests.waveform import InjectedCurrentAPThresholdTest as T
        dtc = data_transport_container.DataTC()
        dtc.rheobase = self.rheobase
        dtc = self.format_test(dtc)
        self.model = VeryReducedModel(get_neab.LEMS_MODEL_PATH, backend=('RAW',{'DTC':dtc}))
        #score = self.run_test(T)
        score = self.run_test(T,pred=self.rheobase)



    def test_rheobase_single_value_parallel_and_serial_comparison(self):

        dataset_id = 354190013  # Internal ID that AIBS uses for a particular Scnn1a-Tg2-Cre
                                # Primary visual area, layer 5 neuron.
        observation = aibs.get_observation(dataset_id,'rheobase')
        rt = RheobaseTest(observation = observation)
        rtp = RheobaseTestP(observation = observation)
        model = VeryReducedModel(backend=('RAW'))

        #model = ReducedModel(get_neab.LEMS_MODEL_PATH,name=str('vanilla'),backend='NEURON')
        self.score_p = rtp.judge(model,stop_on_error = False, deep_error = True)
        self.predictionp = self.score_p.prediction
        self.score_p = self.score_p.sort_key
        #model = ReducedModel(get_neab.LEMS_MODEL_PATH,name=str('vanilla'),backend='NEURON')

        serial_model = ReducedModel(get_neab.LEMS_MODEL_PATH,name=str('vanilla'),backend='RAW')
        self.score_s = rt.judge(serial_model,stop_on_error = False, deep_error = True)
        self.predictions = self.score_s.prediction
        self.score_s = self.score_s.sort_key
        check_less_thresh = float(np.abs(self.predictionp['value'] - self.predictions['value']))
        self.assertLessEqual(check_less_thresh, 2.0)



    #@unittest.skip("Not implemented")
    def test_update_deap_pop(self):
        self.OM.update_deap_pop

    @unittest.skip("Not implemented")
    def test_dtc_to_rheo(self,dtc):
        from neuronunit.optimisation import dtc_to_rheo
        OM.dtc_to_rheo(dtc)

    @unittest.skip("Not implemented")
    def test_evaluate(self,dtc):
        from neuronunit.optimisation_management import evaluate
        assert dtc.scores is not None
        evauate(dtc)


if __name__ == '__main__':
    unittest.main()
