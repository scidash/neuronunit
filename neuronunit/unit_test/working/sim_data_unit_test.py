
# coding: utf-8

# # Set up the environment

import matplotlib.pyplot as plt
import matplotlib
import hide_imports
from neuronunit.optimisation.optimization_management import inject_and_plot_model, inject_and_plot_passive_model
import copy
import pickle
from neuronunit.optimisation.optimization_management import check_match_front
from scipy.stats import linregress
import warnings
warnings.filterwarnings("ignore")

# # Design simulated data tests

def jrt(use_test,backend):
    use_test = hide_imports.TSD(use_test)
    use_test.use_rheobase_score = True
    edges = hide_imports.model_parameters.MODEL_PARAMS[backend]
    OM = hide_imports.OptMan(use_test,
        backend=backend,
        boundary_dict=edges,
        protocol={'allen': False, 'elephant': True})

    return OM

import unittest
class TestSum(unittest.TestCase):
    def sim_data_tests(self,backend,MU,NGEN):
        test_frame = pickle.load(open('processed_multicellular_constraints.p','rb'))
        stds = {}
        for k,v in hide_imports.TSD(test_frame['Neocortex pyramidal cell layer 5-6']).items():
            temp = hide_imports.TSD(test_frame['Neocortex pyramidal cell layer 5-6'])[k]
            stds[k] = temp.observation['std']
        OMObjects = []
        cloned_tests = copy.copy(test_frame['Neocortex pyramidal cell layer 5-6'])

        OM = jrt(cloned_tests,backend)
        rt_outs = []

        x= {k:v for k,v in OM.tests.items() if 'mean' in v.observation.keys() or 'value' in v.observation.keys()}
        cloned_tests = copy.copy(OM.tests)
        OM.tests = hide_imports.TSD(cloned_tests)
        rt_out = OM.simulate_data(OM.tests,OM.backend,OM.boundary_dict)
        penultimate_tests = hide_imports.TSD(test_frame['Neocortex pyramidal cell layer 5-6'])
        for k,v in penultimate_tests.items():
            temp = penultimate_tests[k]

            v = rt_out[1][k].observation
            v['std'] = stds[k]
        simulated_data_tests = hide_imports.TSD(penultimate_tests)
        # # Show what the randomly generated target waveform the optimizer needs to find actually looks like
        # # first lets just optimize over all objective functions all the time.
        # # Commence optimization of models on simulated data sets
        return simulated_data_tests, OM

    def test_two_objectives_test(self):
        results = []
        tests = []
        backend = "RAW"
        MU = NGEN = 10
        simulated_data_tests, OM = self.sim_data_tests(backend,MU,NGEN)


        for i,k in enumerate(simulated_data_tests.keys()):
            for j,l in enumerate(simulated_data_tests.keys()):
                if i!=j:
                    tests = None
                    tests = hide_imports.TSD([simulated_data_tests[k],simulated_data_tests[l]])
                    results.append(tests.optimize(OM.boundary_dict,backend=OM.backend,\
                            protocol={'allen': False, 'elephant': True},\
                                MU=MU,NGEN=NGEN,plot=True))
                    opt = results[-1]['pf'][0].dtc
                    front = [p.dtc for p in results[-1]['pf']]
                    print(opt.obs_preds)
                    try:
                        self.assertLess(opt.obs_preds['total']['scores'],0.0125)
                    except:
                        y1 = [i['avg'][0] for i in results[k]['log'][0:5]]
                        y = [i['min'][0] for i in results[k]['log'][0:5]]
                        x = [i['gen'] for i in results[k]['log'][0:5]]

                        out = linregress(x, y)
                        self.assertLess(out[0],-0.005465789127244809)
                        out = linregress(x, y1)
                        self.assertLess(out[0],-0.005465789127244809)
                break
            break
    def triple_objective_test(self):
        results = []
        tests = []
        backend = "RAW"
        MU = NGEN = 20
        simulated_data_tests, OM = self.sim_data_tests(backend,MU,NGEN)

        for i,k in enumerate(simulated_data_tests.keys()):
            for j,l in enumerate(simulated_data_tests.keys()):
                for m,n in enumerate(simulated_data_tests.keys()):
                    if i!=j and i!=m and m!=j:
                        tests = None
                        tests = hide_imports.TSD([simulated_data_tests[m],simulated_data_tests[k],simulated_data_tests[l]])
                        results.append(tests.optimize(OM.boundary_dict,backend=OM.backend,\
                                protocol={'allen': False, 'elephant': True},\
                                    MU=MU,NGEN=NGEN,plot=True))
                        opt = results[-1]['pf'][0].dtc
                        front = [p.dtc for p in results[-1]['pf']]
                        print(opt.obs_preds)
                        try:
                            self.assertLess(opt.obs_preds['total']['scores'],0.0125)
                        except:
                            y1 = [i['avg'][0] for i in results[k]['log'][0:5]]
                            y = [i['min'][0] for i in results[k]['log'][0:5]]
                            x = [i['gen'] for i in results[k]['log'][0:5]]

                            out = linregress(x, y)
                            self.assertLess(out[0],-0.005465789127244809)
                            out = linregress(x, y1)
                            self.assertLess(out[0],-0.005465789127244809)
                    break
                break
            break
    def test_single_objective_test(self):
        '''
        Test the gradient of a slope
        '''
        results = {}
        tests = {}
        backend = "RAW"
        MU = NGEN = 20
        simulated_data_tests, OM = self.sim_data_tests(backend,MU,NGEN)

        for k in simulated_data_tests.keys():
            tests[k] = hide_imports.TSD([simulated_data_tests[k]])
            results[k] = tests[k].optimize(OM.boundary_dict,backend=OM.backend,\
                    protocol={'allen': False, 'elephant': True},\
                        MU=MU,NGEN=NGEN,plot=True)
            opt = results[k]['pf'][0].dtc
            front = results[k]['pf']
            print(opt.obs_preds)
            try:
                self.assertLess(opt.obs_preds['total']['scores'],0.015)
            except:
                y1 = [i['avg'][0] for i in results[k]['log'][0:6]]
                y = [i['min'][0] for i in results[k]['log'][0:6]]
                x = [i['gen'] for i in results[k]['log'][0:6]]

                out = linregress(x, y)
                self.assertLess(out[0],-0.005465789127244809)
                out = linregress(x, y1)
                self.assertLess(out[0],-0.005465789127244809)
                plt.clf()
                plt.plot(x,y)
                plt.plot(x,y1)
                plt.show()


if __name__ == '__main__':
    unittest.main()
