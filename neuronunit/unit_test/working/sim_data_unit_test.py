
# coding: utf-8

# # Set up the environment
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import matplotlib
import hide_imports
from neuronunit.optimisation.optimization_management import inject_and_plot_model, inject_and_plot_passive_model
import copy
import pickle
from neuronunit.optimisation.optimization_management import check_match_front, jrt
from scipy.stats import linregress
import unittest
import numpy as np

class Test_opt_tests(unittest.TestCase):

    def setUp(self):
        backend = "RAW"
        MU = 20
        NGEN = 20
        with open('processed_multicellular_constraints.p','rb') as f:
            test_frame = pickle.load(f)
        stds = {}
        for k,v in hide_imports.TSD(test_frame['Neocortex pyramidal cell layer 5-6']).items():
            temp = hide_imports.TSD(test_frame['Neocortex pyramidal cell layer 5-6'])[k]
            stds[k] = temp.observation['std']
        cloned_tests = copy.copy(test_frame['Neocortex pyramidal cell layer 5-6'])
        OM = jrt(cloned_tests,backend)
        self.OM = OM
        
    def test_single_objective_test(self):
        backend = "RAW"
        MU = 20
        NGEN = 20

        results = {}
        tests = {}
        simulated_data_tests, OM, target = self.OM.make_sim_data_tests(backend,MU,NGEN,free_parameters=['a','b','C'])
        simulated_data_tests = {k:v for k,v in simulated_data_tests.items() if k != str('TimeConstantTest')}
        simulated_data_tests = {k:v for k,v in simulated_data_tests.items() if k != str('CapacitanceTest')}
        simulated_data_tests = {k:v for k,v in simulated_data_tests.items() if k != str('InjectedCurrentAPWidthTest')}

        for k in simulated_data_tests.keys():
            if k =='TimeConstantTest':
                continue
            if k =='CapacitanceTest':
                continue
            if k == 'InjectedCurrentAPWidthTest':
                continue

            tests[k] = hide_imports.TSD([simulated_data_tests[k]])
            #print('resistance to optimization',tests[k].observation['std'])
            reserve = copy.copy(tests[k])
            results[k] = tests[k].optimize(OM.boundary_dict,backend=OM.backend,\
                    protocol={'allen': False, 'elephant': True},\
                        MU=MU,NGEN=NGEN,plot=True,free_params=['a','b','C'])
            min_ = np.min([ p for p in results[k]['history'].genealogy_history.values() ])
            max_ = np.max([ p for p in results[k]['history'].genealogy_history.values() ])
            model = target.dtc_to_model()
            tests[k][list(tests[k].keys())[0]].judge(model)

            assert min_<target.attrs['a']<max_
            self.assertLess(min_,target.attrs['a'])
            opt = results[k]['pf'][0].dtc
            print(opt.attrs)
            front = results[k]['pf']
            print(opt.obs_preds)
            self.assertLess(opt.obs_preds['total']['scores'],0.10)

            if opt.obs_preds['total']['scores'] < 0.100:
                y1 = [i['avg'][0] for i in results[k]['log']]
                y = [i['min'][0] for i in results[k]['log']]
                x = [i['gen'] for i in results[k]['log']]

                slopem = linregress(x, y)
                slopea = linregress(x, y1)
                gene = results[k]['pf'][0].dtc
                mm = results[k]['pf'][0].dtc.dtc_to_model()
                this_test = tests[k][list(tests[k].keys())[0]]
                score_gene = this_test.judge(mm)
                pred_gene = this_test.prediction


                model = target.dtc_to_model()
                this_test.judge(model)
                pred_target = this_test.prediction

                try:
                    inject_and_plot_passive_model(target,second=results[k]['pf'][0].dtc,figname='debug_target_gene.png')
                
                    gene.tests = target.tests = None
                    with open(str(gene.attrs)+str(gene.backend)+str(k)+'.p','wb') as f:
                        pickle.dump([target,gene,test_dump],f)

                except:
                    pass
    '''    
                
    def test_two_objectives_test(self):
        results = {}
        #tests = []
        backend = "RAW"
        MU = NGEN = 45
        #simulated_data_tests, OM = self.OM.make_sim_data_tests(backend,MU,NGEN)
        simulated_data_tests, OM, target = self.OM.make_sim_data_tests(backend,MU,NGEN,free_parameters=['a','b','C'])
        simulated_data_tests = {k:v for k,v in simulated_data_tests.items() if k != str('TimeConstantTest')}
        simulated_data_tests = {k:v for k,v in simulated_data_tests.items() if k != str('CapacitanceTest')}
        simulated_data_tests = {k:v for k,v in simulated_data_tests.items() if k != str('InjectedCurrentAPWidthTest')}
        for i,(k,v0) in enumerate(simulated_data_tests.items()):
            for j,(l,v1) in enumerate(simulated_data_tests.items()):
                if i!=j:
                    
                    tests = hide_imports.TSD([v0,v1])
                    results[k] = tests.optimize(OM.boundary_dict,backend=OM.backend,\
                                protocol={'allen': False, 'elephant': True},\
                                    MU=MU,NGEN=NGEN,plot=True,free_parameters=['a','b','C'])
                    opt = results[k]['pf'][0].dtc
                    front = results[k]['pf']
                    print(opt.obs_preds)
                    self.assertLess(opt.obs_preds['total']['scores'],0.125)
                    print('the score was bad the gradient of the optimizer good?')
                    y1 = [i['avg'][0] for i in results[k]['log'][0:7]]
                    y = [i['min'][0] for i in results[k]['log'][0:7]]
                    x = [i['gen'] for i in results[k]['log'][0:7]]

                    out = linregress(x, y)
                    self.assertLess(out[0],-0.0025465789127244809)
                    out = linregress(x, y1)
                    self.assertLess(out[0],-0.0025465789127244809)
                    break
            break
    def triple_objective_test(self):
        results = {}
        tests = []
        backend = "RAW"
        MU = NGEN = 40
        simulated_data_tests, OM = OM.make_sim_data_tests(backend,MU,NGEN,free_parameters=['a','b','C'])

        for i,k in enumerate(simulated_data_tests.keys()):
            for j,l in enumerate(simulated_data_tests.keys()):
                for m,n in enumerate(simulated_data_tests.keys()):
                    if i!=j and i!=m and m!=j:
                        tests = None
                        tests = hide_imports.TSD([simulated_data_tests[m],simulated_data_tests[k],simulated_data_tests[l]])
                        results[k] = tests.optimize(OM.boundary_dict,backend=OM.backend,\
                                protocol={'allen': False, 'elephant': True},\
                                    MU=MU,NGEN=NGEN,plot=True,free_parameters=['a','b','C'])
                        opt = results[k]['pf'][0].dtc
                        front = results[k]['pf']
                        print(opt.obs_preds)
                        y1 = [i['avg'][0] for i in results[k]['log'][0:7]]
                        y = [i['min'][0] for i in results[k]['log'][0:7]]
                        x = [i['gen'] for i in results[k]['log'][0:7]]

                        out = linregress(x, y)
                        self.assertLess(out[0],-0.0025465789127244809)
                        out = linregress(x, y1)
                        self.assertLess(out[0],-0.0025465789127244809)
                        break
                break
            break
    '''            
if __name__ == '__main__':
    unittest.main()
