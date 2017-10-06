"""Unit tests for the showcase features of NeuronUnit"""

# Run with any of:
# python showcase_tests.py
# python -m unittest showcase_tests.py
# coverage run --source . showcase_tests.py

import unittest
import os

os.system('ipcluster start -n 8 --profile=default & sleep 25;')

import quantities as pq
import ipyparallel as ipp
rc = ipp.Client(profile='default')
rc[:].use_cloudpickle()
dview = rc[:]


class ReducedModelTestCase(unittest.TestCase):
    """Test instantiation of the reduced model"""

    """Testing model optimization"""


    def setUp(self):
        from neuronunit.models.reduced import ReducedModel
        #self.ReducedModel = ReducedModel
        #path = ReducedModelTestCase().path
        #self.model = self.ReducedModel(path, backend='NEURON')
        self.predictions = None
        self.predictionp = None
        self.score_p = None
        self.score_s = None
        self.timed_p = None
        self.timed_s = None


    def run_test(self, cls):
        observation = self.get_observation(cls)
        test = cls(observation=observation)
        score = test.judge(self.model)
        score.summarize()
        return score.score


    def nrn_backend_works(self):
        from neuronunit.optimization import get_neab
        get_neab.LEMS_MODEL_PATH = '/home/jovyan/neuronunit/neuronunit/optimization/NeuroML2/LEMS_2007One.xml'

        from neuronunit.models import backends
        from neuronunit.models.reduced import ReducedModel
        model = ReducedModel(get_neab.LEMS_MODEL_PATH,name=str('vanilla'),backend='NEURON')
        #import pdb; pdb.set_trace()
        #score = get_neab.tests[0].judge(model,stop_on_error = False, deep_error = True)
    def check_parallel_path_consistency():
        '''
        import paths and test for consistency
        '''
        from neuronunit import models
        return models.__file__

    def test_0check_paths(self):

        path_serial = check_paths()
        paths_parallel = dview.apply_async(check_paths).get_dict()
        self.assertEqual(path_serial, paths_parallel[0])

    def test_2_run_opt(self):
        import nsga_parallel

    def test_1_run_exhaust(self):
        import model_parameters as modelp
        iter_list=iter( (i,j,k,l,m,n,o,p,q,r) for i in modelp.model_params['a'][0:-1] for j in modelp.model_params['b'][0:-1] \
        for k in modelp.model_params['vr'][0:-1] for l in modelp.model_params['vpeak'][0:-1] \
        for m in modelp.model_params['k'][0:-1] for n in modelp.model_params['c'][0:-1] \
        for o in modelp.model_params['C'][0:-1] for p in modelp.model_params['d'][0:-1] \
        for q in modelp.model_params['v0'][0:-1] for r in modelp.model_params['vt'][0:-1] )

    def test_3check_paths(self):
        self.nrn_backend_works()



    def test_4rheobase_setup(self):
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
        self.predictionp = self.score_p.prediction

        self.score_p = self.score_p.sort_key
        tickp = time.time()
        self.timed_p = tickp - ticks
        self.score_s = rt.judge(model,stop_on_error = False, deep_error = True)
        self.predictions = self.score_s.prediction

        self.score_s = self.score_s.sort_key
        self.timed_s = time.time() - tickp


        print(' serial score {0} parallel score {1}'.format(self.score_s,self.score_p))
        print(' serial time {0} parallel time {1}'.format(self.timed_s,self.timed_p))

    def test_5rheobase_check(self):
        self.assertEqual(int(self.score_s*1000), int(self.score_p*1000))

    def test_6rheobase_check(self):
        self.assertGreater(self.timed_s,self.timed_p)

    def test_7rheobase_check(self):
        self.assertEqual(int(self.predictionp['value']), int(self.predictions['value']))
        # return self.score_s, self.score_p, self.timed_s, self.timed_p, self.predictionp, self.predictions

    # def test_optimizer(self):
    #    from neuronunit.optimization import nsga_parallel
    #    difference_progress, fitnesses, pf, logbook, pop, dtcpop, stats = nsga_parallel.main(MU=12, NGEN=4, CXPB=0.9)




if __name__ == '__main__':
    unittest.main()
