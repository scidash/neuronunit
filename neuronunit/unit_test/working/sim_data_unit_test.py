
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
from neuronunit.optimisation.optimization_management import check_binary_match
from neuronunit.optimisation.optimization_management import which_key

class Test_opt_tests(unittest.TestCase):

    def setUp(self):
        backend = "RAW"
        MU = 30
        NGEN = 20
        with open('processed_multicellular_constraints.p','rb') as f: test_frame = pickle.load(f)
        stds = {}
        for k,v in hide_imports.TSD(test_frame['Neocortex pyramidal cell layer 5-6']).items():
            temp = hide_imports.TSD(test_frame['Neocortex pyramidal cell layer 5-6'])[k]
            stds[k] = temp.observation['std']
        cloned_tests = copy.copy(test_frame['Neocortex pyramidal cell layer 5-6'])
        OM = jrt(cloned_tests,backend,protocol='elephant')
        self.OM = OM
    def test_all_objective_test(self):
        backend = "RAW"
        MU = 10
        NGEN = 350

        results = {}
        tests = {}
        fps = ['k','a','c','d','b','vPeak','C','vr']
        simulated_data_tests, OM, target = self.OM.make_sim_data_tests(
            backend,MU,NGEN,free_parameters=fps)

        
        stds = {}
        for k,v in simulated_data_tests.items():
            keyed = which_key(simulated_data_tests[k].observation)
            if k == str('RheobaseTest'):
                mean = simulated_data_tests[k].observation[keyed]
                std = simulated_data_tests[k].observation['std']
                x = np.abs(std/mean)
            #import pdb
            #pdb.set_trace()
            if k == str('TimeConstantTest') or k == str('CapacitanceTest') or k == str('InjectedCurrentAPWidthTest'):
                # or k == str('InjectedCurrentAPWidthTest'):
                mean = simulated_data_tests[k].observation[keyed]
                simulated_data_tests[k].observation['std'] = np.abs(mean)*2.0
            elif k == str('InjectedCurrentAPThresholdTest') or k == str('InjectedCurrentAPAmplitudeTest'):
                mean = simulated_data_tests[k].observation[keyed]
                simulated_data_tests[k].observation['std'] = np.abs(mean)*2.0


            stds[k] = (x,mean,std)
        #simulated_data_tests.pop('InjectedCurrentAPWidthTest',None)
        #print(stds[k])
        with open('standard_scales.p','wb') as f:
            pickle.dump(stds,f)
        with open('standard_scales.p','rb') as f:
            standards = pickle.load(f)    
        

        target.tests = simulated_data_tests
        for t in simulated_data_tests.values(): 
            score0 = t.judge(target.dtc_to_model())
            score1 = target.tests[t.name].judge(target.dtc_to_model())

            try:
                assert float(score0.score)==0.0
                assert float(score1.score)==0.0

            except:
                import pdb
                pdb.set_trace()


        tests = hide_imports.TSD(copy.copy(simulated_data_tests))
        check_tests = copy.copy(tests)
        reserve = copy.copy(tests)
        results = tests.optimize(backend=OM.backend,\
                protocol={'allen': False, 'elephant': True},\
                    MU=MU,NGEN=NGEN,plot=True,free_params=fps)
        min_ = np.min([ p for p in results['history'].genealogy_history.values() ])
        max_ = np.max([ p for p in results['history'].genealogy_history.values() ])
        temp = [ p for p in results['history'].genealogy_history.values() ]
        try:
            plt.clf()
            #plt.hline(list(range(0,len(temp))),fps[0])
            ax2[i].axhline(y=fps[0], xmin=0.02, xmax=0.99,color='blue')

            plt.plot(list(range(0,len(temp))),temp)
            plt.savefig(str(fps[0])+"progress.png")
        except:
            pass
 
        model = target.dtc_to_model()
        opt = results['pf'][0].dtc

        check_binary_match(opt,target,figname='checkbin.png')
        opt = OM.format_test(opt)

        opt = self.OM.get_agreement(opt)
        print(opt.obs_preds)
        
        #pickle.dump(model,open('model_pickle.p','wb'))
        target = OM.format_test(target)
        
        simulated_data_tests = target.tests
        for k,t in enumerate(simulated_data_tests): 
            print(t.judge(target.dtc_to_model()))
            print(target.tests[k].judge(target.dtc_to_model()))
            
            score0 = t.judge(target.dtc_to_model())
            score1 = target.tests[k].judge(target.dtc_to_model())

            try:
                assert float(score0.score)==0.0
                assert float(score1.score)==0.0

            except:
                import pdb
                pdb.set_trace()


        for t in target.tests:  
            print(t.judge(target.dtc_to_model()))
            model = target.dtc_to_model()
            score = t.judge(model)
            try:
                print(score.log_norm_score)
                assert float(score.score)==0.0
            except:
                import pdb
                pdb.set_trace()

 
        print(opt.attrs)
        print(target.attrs)
        front = results['pf']
        front = results['hof']
        for i,t in enumerate(opt.tests): 
            assert t.observation['mean']==target.tests[i].observation['mean']
        #self.assertLess(opt.obs_preds['total']['scores'],1.250)
        with open(str('RAW')+str('optimum_versus_target.p'),'wb') as f:
            pickle.dump([target,opt],f)
        a = pickle.load(open("RAWoptimum_versus_target.p","rb"))
        import pdb
        pdb.set_trace()
        import sys
        sys.exit()

        if opt.obs_preds['total']['scores'] < 0.100:
            y1 = [i['avg'][0] for i in results['log']]
            y = [i['min'][0] for i in results['log']]
            x = [i['gen'] for i in results['log']]

            slopem = linregress(x, y)
            slopea = linregress(x, y1)
            mm = results['pf'][0].dtc.dtc_to_model()
            this_test = tests[list(tests.keys())[0]]
            score_gene = this_test.judge(mm)
            pred_gene = this_test.prediction


            model = target.dtc_to_model()
            this_test.judge(model)
            pred_target = this_test.prediction
if __name__ == '__main__':
    unittest.main()

            
        
    '''
    def test_all_objective_test_HH(self):
        backend = "HH"
        MU = 10
        NGEN = 10

        from neuronunit.optimisation.optimization_management import which_key

        simulated_data_tests, OM, target = self.OM.make_sim_data_tests(backend,MU,NGEN)#,free_parameters=['a','b','C'])
        for k,v in simulated_data_tests.items():
            keyed = which_key(simulated_data_tests[k].observation)

            if k == str('TimeConstantTest') or k == str('CapacitanceTest') or k == str('InjectedCurrentAPWidthTest'):
                mean = simulated_data_tests[k].observation[keyed]
                simulated_data_tests[k].observation['std'] = np.abs(x*mean)
            else:
                mean = simulated_data_tests[k].observation[keyed]
                std = simulated_data_tests[k].observation['std']
                x = np.abs(std/mean)
        tests = hide_imports.TSD(simulated_data_tests)
        reserve = copy.copy(tests)
        results = tests.optimize(backend=OM.backend,\
                protocol={'allen': False, 'elephant': True},\
                    MU=MU,NGEN=NGEN,plot=True)#,free_params=['a','b','C'])
        model = target.dtc_to_model()
        tests[list(tests.keys())[0]].judge(model)

        opt = results['pf'][0].dtc
        print(opt.attrs)
        front = results['pf']
        print(opt.obs_preds)
        #self.assertLess(opt.obs_preds['total']['scores'],0.250)
        gene = results['pf'][0].dtc
            
        gene.tests = target.tests = None
        with open(str('HH')+str(gene.attrs)+str(gene.backend)+str('all_tests')+'.p','wb') as f:
            pickle.dump([target,gene,test_dump],f)

        if opt.obs_preds['total']['scores'] < 0.500:
            y1 = [i['avg'][0] for i in results['log']]
            y = [i['min'][0] for i in results['log']]
            x = [i['gen'] for i in results['log']]

            slopem = linregress(x, y)
            slopea = linregress(x, y1)
            gene = results['pf'][0].dtc
            mm = results['pf'][0].dtc.dtc_to_model()
            this_test = tests[list(tests.keys())[0]]
            score_gene = this_test.judge(mm)
            pred_gene = this_test.prediction


            model = target.dtc_to_model()
            this_test.judge(model)
            pred_target = this_test.prediction
            #inject_and_plot_passive_model(target,second=results[k]['pf'][0].dtc,figname='debug_target_gene.png')
            try:

                gene.tests = target.tests = None
                with open(str(gene.attrs)+str(gene.backend)+str('all_tests')+'.p','wb') as f:
                    pickle.dump([target,gene,test_dump],f)

            except:
                print('does not plot')
'''
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
