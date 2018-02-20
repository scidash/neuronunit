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

    def test_2_backend_pyNN(self):
        from neuronunit.models.reduced import ReducedModel
        from neuronunit.optimization import get_neab
        print(get_neab.LEMS_MODEL_PATH)
        model = ReducedModel(get_neab.LEMS_MODEL_PATH, backend='pyNN')
        print(model)
        #import pdb; pdb.set_trace()




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
        #import pdb; pdb.set_trace()
        #print('gets here')
        for i in range(1,10):
        #i = 1 #later this will be a loop as comment above.
            subset = nsga_parallel.create_subset(nparams=i)
            numb_err_f = 8
            toolbox, tools, history, creator, base = evaluate_as_module.import_list(ipp,subset,numb_err_f)
            ind = toolbox.population(n = 1)
            print(len(ind),i)

            N = nsga_object.NSGA(nparams=i)
            self.assertEqual(N.nparams,i)
            N.setnparams(nparams=i)
            self.assertEqual(N.nparams,i)


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
            #import pdb; pdb.set_trace()
        return disagreement, dis

    def test_6_agreement(self):
        disagreement, dis = self.test_5_agreement()


if __name__ == '__main__':
    unittest.main()
    unittest.test_5_agreement()
