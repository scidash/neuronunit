"""Unit tests for the showcase features of NeuronUnit"""

# Run with any of:
# python showcase_tests.py
# python -m unittest showcase_tests.py
# coverage run --source . showcase_tests.py

import unittest
import os
import quantities as pq
import numpy as np


def create_list():
    from neuronunit.optimization import model_parameters as modelp

    mp = modelp.model_params
    all_keys = [ key for key in mp.keys() ]
    smaller = {}
    # First create a smaller subet of the larger parameter dictionary.
    #
    for k in all_keys:
        subset = {}
        subset[k] = (mp[k][0] , mp[k][int(len(mp[k])/2.0)], mp[k][-1] )
        smaller.update(subset)


    iter_list=[ {'a':i,'b':j,'vr':k,'vpeak':l,'k':m,'c':n,'C':o,'d':p,'v0':q,'vt':r} for i in smaller['a'] for j in smaller['b'] \
    for k in smaller['vr'] for l in smaller['vpeak'] \
    for m in smaller['k'] for n in smaller['c'] \
    for o in smaller['C'] for p in smaller['d'] \
    for q in smaller['v0'] for r in smaller['vt'] ]
    # the size of this list is 59,049 approx 60,000 calls after rheobase is found.
    # assert 3**10 == 59049
    return iter_list

def parallel_method(item_of_iter_list):

    from neuronunit.optimization import get_neab
    get_neab.LEMS_MODEL_PATH = '/home/jovyan/neuronunit/neuronunit/optimization/NeuroML2/LEMS_2007One.xml'
    #from neuronunit.models import backends
    from neuronunit.models.reduced import ReducedModel
    model = ReducedModel(get_neab.LEMS_MODEL_PATH,name=str('vanilla'),backend='NEURON')
    model.set_attrs(item_of_iter_list)
    get_neab.tests[0].prediction = dtc.rheobase
    model.rheobase = dtc.rheobase['value']
    scores = []
    for k,t in enumerate(get_neab.tests):
        if k>1:
            t.params = dtc.vtest[k]
            score = t.judge(model,stop_on_error = False, deep_error = True)
            scores.append(score.norm_score,score)
    return scores

def exhaustive_search(self):
    iter_list = create_list()
    scores = list(dview.map(parallel_method,iter_list).get())
    #score_parameter_pairs = zip(scores,iter_list)

    #print(iter_list)


from neuronunit import tests
#from deap import hypervolume


#test_0_run_exhaust()

os.system('ipcluster start -n 8 --profile=default & sleep 5;')
import ipyparallel as ipp
rc = ipp.Client(profile='default')
rc[:].use_cloudpickle()
dview = rc[:]

class ReducedModelTestCase(unittest.TestCase):
    """Test instantiation of the reduced model"""

    """Testing model optimization"""


    def setUp(self):
        #import sys
        #sys.path.append('../')
        #import neuronunit

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
        self.attrs_list = None

    def run_test(self, cls):
        observation = self.get_observation(cls)
        test = cls(observation=observation)
        score = test.judge(self.model)
        score.summarize()
        return score.score


    def nrn_backend_works(self):
        from neuronunit.optimization import get_neab
        get_neab.LEMS_MODEL_PATH = '/home/jovyan/neuronunit/neuronunit/optimization/NeuroML2/LEMS_2007One.xml'

        #from neuronunit.models import backends
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

    # def test_0check_paths(self):

    #    path_serial = check_paths()
    #    paths_parallel = dview.apply_async(check_parallel_path_consistency).get_dict()
    #    self.assertEqual(path_serial, paths_parallel[0])
    @unittest.skip("This times out")

    def test_1_run_opt(self):
        from neuronunit.optimization import nsga_parallel
        with open('opt_run_data.p','rb') as handle:
            attrs = pickle.load(handle)
        self.attrs_list = attrs


        #for i in iter_list
    def test_3check_paths(self):
        self.nrn_backend_works()



    def test_4rheobase_setup(self):
        from neuronunit.tests.fi import RheobaseTest, RheobaseTestP
        from neuronunit.optimization import get_neab
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

        self.score_p = self.score_p.norm_score
        tickp = time.time()
        self.timed_p = tickp - ticks
        self.score_s = rt.judge(model,stop_on_error = False, deep_error = True)
        self.predictions = self.score_s.prediction

        self.score_s = self.score_s.norm_score
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

    #    difference_progress, fitnesses, pf, logbook, pop, dtcpop, stats = nsga_parallel.main(MU=12, NGEN=4, CXPB=0.9)




if __name__ == '__main__':
    #a = ReducedModelTestCase()
    #a.test_0_run_exhaust()
    unittest.main()
