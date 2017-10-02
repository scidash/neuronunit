"""Unit tests for the showcase features of NeuronUnit"""

# Run with any of:
# python showcase_tests.py
# python -m unittest showcase_tests.py
# coverage run --source . showcase_tests.py

import unittest
import os

import quantities as pq
import ipyparallel as ipp
rc = ipp.Client(profile='default')
rc[:].use_cloudpickle()
dview = rc[:]
os.system('ipcluster start -n 8 --profile=default & sleep 25;')


class ReducedModelTestCase(unittest.TestCase):
    """Test instantiation of the reduced model"""

    """Testing model optimization"""

    def setUp(self):
        from neuronunit.models.reduced import ReducedModel
        self.ReducedModel = ReducedModel

    @property
    def path(self):
            import neuronunit.models
            return os.path.join(neuronunit.models.__path__[0],
                                'NeuroML2','LEMS_2007One.xml')

    def test_reducedmodel_jneuroml(self):
        model = self.ReducedModel(self.path, backend='jNeuroML')

    #@unittest.skipIf(OSX,"NEURON unreliable on OSX")
    def test_reducedmodel_neuron(self):
        #compile_path = str(self.path)
        import os
        compile_path = os.path.join(neuronunit.models.__path__[0],'NeuroML2')
        os.system('nrnivmodl '+compile_path)
        model = self.ReducedModel(self.path, backend='NEURON')


class TestsTestCase(object):
    """Abstract base class for testing tests"""

    def setUp(self):
        from neuronunit.models.reduced import ReducedModel
        self.ReducedModel = ReducedModel
        path = ReducedModelTestCase().path
        self.model = self.ReducedModel(path, backend='NEURON')
        self.predictions = None
        self.predictionp = None
        self.score_p = None
        self.score_s = None
        self.timedp = None
        self.timeds = None


    def get_observation(self, cls):
        print(cls.__name__)
        neuron = {'nlex_id': 'nifext_50'} # Layer V pyramidal cell
        return cls.neuroelectro_summary_observation(neuron)

    def run_test(self, cls):
        observation = self.get_observation(cls)
        test = cls(observation=observation)
        score = test.judge(self.model)
        score.summarize()
        return score.score


    def nrn_backend_works(self):
        from neuronunit.tests import get_neab
        get_neab.LEMS_MODEL_PATH = '/home/jovyan/neuronunit/neuronunit/optimization/NeuroML2/LEMS_2007One.xml'

        from neuronunit.models import backends
        from neuronunit.models.reduced import ReducedModel
        model = ReducedModel(get_neab.LEMS_MODEL_PATH,name=str('vanilla'),backend='NEURON')
        #import pdb; pdb.set_trace()
        #score = get_neab.tests[0].judge(model,stop_on_error = False, deep_error = True)
    from neuronunit.tests import get_neab
    def check_parallel_path_consistency():
        '''
        import paths and test for consistency
        '''
        from neuronunit import models
        return models.__file__

    def test_check_paths(self):

        path_serial = check_paths()
        paths_parallel = dview.apply_async(check_paths).get_dict()
        self.assertEqual(path_serial, paths_parallel[0])

    def test_check_paths(self):
        self.nrn_backend_works()

    def rheobase_check(self):

        from neuronunit.tests.fi import RheobaseTest, RheobaseTestP
        from neuronunit.tests import get_neab
        from neuronunit.models import backends
        from neuronunit.models.reduced import ReducedModel
        import time
        from neuronunit import aibs
        import os
        dataset_id = 354190013  # Internal ID that AIBS uses for a particular Scnn1a-Tg2-Cre
                                # Primary visual area, layer 5 neuron.
        observation = aibs.get_observation(dataset_id,'rheobase')
        #os.system('ipcluster start -n 8 --profile=default & sleep 25;')

        rt = RheobaseTest(observation = observation)
        rtp = RheobaseTestP(observation = observation)

        get_neab.LEMS_MODEL_PATH = '/home/jovyan/neuronunit/neuronunit/optimization/NeuroML2/LEMS_2007One.xml'
        model = ReducedModel(get_neab.LEMS_MODEL_PATH,name=str('vanilla'),backend='NEURON')
        ticks = time.time()
        self.score_p = rtp.judge(model,stop_on_error = False, deep_error = True)
        self.score_p = self.score_p.sort_key
        tickp = time.time()
        self.timed_p = tickp - ticks
        self.score_s = rt.judge(model,stop_on_error = False, deep_error = True)
        self.score_s = self.score_s.sort_key
        self.timed_s = time.time() - tickp


        self.predictions = score_s.prediction
        self.predictionp = score_p.prediction
        return self.score_s, self.score_p, self.timed_s, self.timed_p, self.predictionp, self.predictions


    def test_rheobase_scores(self):
        print(' serial score {0} parallel score {1}'.format(self.score_s,self.score_p))

        #scores only need to be approximately equal to a reasonable level of precision.
        unittest.assertEqual(int(self.score_s*100), int(self.score_p*100))

    def test_rheobase_times(self):
        print(' serial time {0} parallel time {1}'.format(self.timed_s,self.timed_p))

        assert(self.timed_s > self.timed_p)


    def test_rheobase_predictions(self):
        # since methods are associated with different precision.
        # its indictative enough that the numbers are different.
        unittest.assertEqual(int(self.predictionp), int(self.predictions))



    def test_update_pop(dtcpop):
        dtcpop = list(map(dtc_to_rheo,dtcpop))
        dtcpop = [ dtc for dtc in dtcpop if type(dtc) is not type(None) ]
        dtcpop = list(map(evaluate_as_module.pre_format,dtcpop))
        dtcpop = list(dview.map(map_wrapper,dtcpop).get())
        return dtcpop
        dtcpop = update_pop(dtcpop)
        for d in dtcpop:
            assert type(d.rheobase['value']) is not type(None)

if __name__ == '__main__':
    unittest.main()
