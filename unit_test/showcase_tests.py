"""Unit tests for the showcase features of NeuronUnit"""

# Run with any of:  
# python showcase_tests.py
# python -m unittest showcase_tests.py
# coverage run --source . showcase_tests.py

import unittest
import sys
import os
import warnings
import pdb

from scoop import futures
import quantities as pq

from sciunit.utils import NotebookTools

# CONVERT_NOTEBOOKS environment variable controls whether notebooks are
# executed as notebooks or converted to regular python files first. 

class DocumentationTestCase(NotebookTools,unittest.TestCase):
    """Testing documentation notebooks"""

    path = 'docs'

    #@unittest.skip("Skipping chapter 2")
    def test_chapter2(self):
        self.do_notebook('chapter2')

    #@unittest.skip("Skipping chapter 3")
    def test_chapter3(self):
        self.do_notebook('chapter3')

    @unittest.skip("Skipping chapter 4")
    def test_chapter4(self):
        self.do_notebook('chapter4')


class OptimizationTestCase(unittest.TestCase):
    """Testing model optimization"""

    def func2map(self, file2map):
        exec_string='python '+str(file2map)
        os.system(exec_string)
        return 0

    def test_main(self, ind, guess_attrs=None):
        vm = VirtualModel()
        if guess_attrs!=None:
            for i, p in enumerate(param):
                value=str(guess_attrs[i])
                model.name=str(model.name)+' '+str(p)+str(value)
                if i==0:
                    attrs={'//izhikevich2007Cell':{p:value }}
                else:
                    attrs['//izhikevich2007Cell'][p]=value
            vm.attrs=attrs
            guess_attrs=None#stop reentry into this condition during while,
        else:
            import copy
            vm.attrs=ind.attrs

        begin_time = time.time()
        while_true = True
        while(while_true):
            from itertools import repeat
            if len(vm.lookup)==0:
                steps2 = np.linspace(50,190,4.0)
                steps = [ i*pq.pA for i in steps2 ]
                lookup2=list(map(f,steps,repeat(vm)))#,repeat(model)))

            m = lookup2[0]
            assert(type(m))!=None

            sub = []
            supra = []
            assert(type(m.lookup))!=None
            for k,v in m.lookup.items():
                if v==1:
                    while_true=False
                    end_time=time.time()
                    total_time=end_time-begin_time
                    return (m.run_number,k,m.attrs)#a
                    break
                elif v==0:
                    sub.append(k)
                elif v>0:
                    supra.append(k)
            sub = np.array(sub)
            supra = np.array(supra)
            if len(sub) and len(supra):
                steps2 = np.linspace(sub.max(),supra.min(),4.0)
                steps = [ i*pq.pA for i in steps2 ]

            elif len(sub):
                steps2 = np.linspace(sub.max(),2*sub.max(),4.0)
                steps = [ i*pq.pA for i in steps2 ]
            elif len(supra):
                steps2 = np.linspace(-1*(supra.min()),supra.min(),4.0)
                steps = [ i*pq.pA for i in steps2 ]

            lookup2=list(map(f,steps,repeat(vm)))

    def test_build_single(self, rh_value):
        '''
        This method is only used to check singlular sets of hard coded parameters.
        '''
        get_neab.suite.tests[0].prediction = {}
        get_neab.suite.tests[0].prediction['value'] = rh_value*pq.pA
        print(get_neab.suite.tests[0].prediction['value'])
        attrs={}
        attrs['//izhikevich2007Cell']={}
        attrs['//izhikevich2007Cell']['a']=0.045
        attrs['//izhikevich2007Cell']['b']=-5e-09
        #attrs['//izhikevich2007Cell']['vpeak']=30.0
        #attrs['//izhikevich2007Cell']['vr']=-53.4989145966
        model.update_run_params(attrs)
        score = get_neab.suite.judge(model)#passing in model, changes model
        error = []
        error = [ abs(i.score) for i in score.unstack() ]
        return model

    def test_run_all_files(self):
        '''
        run all files as different CPU threads, thus saving time on travis
        Since scoop is designed to facilitate nested forking/parallel job dispatch
        This approach should be scalable to larger CPU pools.
        '''
        files_to_exec=['neuronunit/tests/exhaustive_search.py',
                       'neuronunit/tests/nsga.py']
        clean_or_dirty=list(futures.map(self.func2map,files_to_exec))
        return clean_or_dirty
        

if __name__ == '__main__':
    unittest.main()
