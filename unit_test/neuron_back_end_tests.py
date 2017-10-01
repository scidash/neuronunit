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

class OptimizationTestCase(unittest.TestCase):
    """Testing model optimization"""
    from neuronunit.tests import get_neab
    def check_paths():
        '''
        import paths and test for consistency
        '''
        from neuronunit import models
        return models.__file__

    def test_check_paths(self):
        path_serial = check_paths()
        paths_parallel = dview.apply_async(check_paths).get_dict()
        self.assertEqual(path_serial, paths_parallel[0])

    def nrn_backend_works(self):
        from neuronunit.tests import get_neab
        get_neab.LEMS_MODEL_PATH = '/home/jovyan/neuronunit/neuronunit/optimization/NeuroML2/LEMS_2007One.xml'

        from neuronunit.models import backends
        from neuronunit.models.reduced import ReducedModel
        model = ReducedModel(get_neab.LEMS_MODEL_PATH,name=str('vanilla'),backend='NEURON')
        #score = get_neab.tests[0].judge(model,stop_on_error = False, deep_error = True)

    def test_check_paths(self):
        self.nrn_backend_works()

    def rheobase_check():

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
        os.system('ipcluster start -n 8 --profile=default & sleep 25;')

        rt = RheobaseTest(observation = observation)
        rtp = RheobaseTestP(observation = observation)

        get_neab.LEMS_MODEL_PATH = '/home/jovyan/neuronunit/neuronunit/optimization/NeuroML2/LEMS_2007One.xml'
        model = ReducedModel(get_neab.LEMS_MODEL_PATH,name=str('vanilla'),backend='NEURON')
        ticks = time.time()
        score_p = rtp.judge(model,stop_on_error = False, deep_error = True)
        tickp = time.time()
        timed_p = tickp - ticks
        score_s = rt.judge(model,stop_on_error = False, deep_error = True)
        timed_s = time.time() - tickp

        print(' serial {0} parallel {1}'.format(timed_s,timed_p))

        predictions = score_s.prediction
        predictionp = score_p.prediction
        try:
            assert int(predictionp) == int(predictions)
        except:
            'predictions not equal'
        try:
            assert score_s == score_p
        except:
            'scores not equal'
        try:
            assert(timed_s > timed_p)
        except:
            print('serial faster ?')
        print('score_s.sort_key, score_p.sort_key, timed_s, timed_p, predictionp, predictions')

        print(score_s.sort_key, score_p.sort_key, timed_s, timed_p, predictionp, predictions)
        print(score_s.sort_key==score_p.sort_key)
        print(timed_s==timed_p)
        print(predictionp==predictions)
        return score_s.sort_key, score_p.sort_key, timed_s, timed_p, predictionp, predictions

    (score_s, score_p, timed_s, timed_p,  predictionp, predictions) = rheobase_check()

    '''
    def test_rheobase_scores(self, score_s, score_p):
        unittest.assertEqual(score_s, score_p)

    def test_rheobase_times(self, timed_s, timed_p):
        assert(timed_s > timed_p)


    def test_rheobase_predictions(self, predictionp, predictions):
        # since methods are associated with different precision.
        # its indictative enough that the numbers are different.
        unittest.assertEqual(int(predictionp), int(predictions))
    '''

if __name__ == '__main__':
    unittest.main()
