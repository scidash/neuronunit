"""Unit tests for the core of NeuronUnit"""

# Run with any of:
# python core_tests.py
# python -m unittest core_tests.py
# coverage run --source . core_tests.py

import unittest
import sys
import os
os.system('ipcluster start -n 8 --profile=default &')
#os.system('sleep 5')

import ipyparallel as ipp
#from ipyparallel.apps import iploggerapp
rc = ipp.Client(profile='default')
THIS_DIR = os.path.dirname(os.getcwd())

this_nu = os.path.join(THIS_DIR,'../')
sys.path.insert(0,this_nu)
#from neuronunit.optimization import nsga_p_for_nb

from neuronunit import tests

rc[:].use_cloudpickle()
dview = rc[:]

import warnings
import pdb


import ipyparallel as ipp
rc = ipp.Client(profile='default')
rc[:].use_cloudpickle()
dview = rc[:]
import os
with dview.sync_imports():
    import os
dview.apply(os.chdir,os.getcwd())

def modify_path(THIS_DIR):
    this_nu = os.path.join(THIS_DIR,'../../')
    sys.path.insert(0,this_nu)

THIS_DIR = os.path.dirname(os.getcwd())
dview.apply(modify_path,THIS_DIR)

with dview.sync_imports():
    from neuronunit.optimization import get_neab
    from neuronunit.optimization import utilities
    from utilities import VirtualModel
    from neuronunit import tests

dview.push({'VirtualModel':VirtualModel})

with dview.sync_imports():
    import nsga_p_for_nb

class OptimizationTestCase(unittest.TestCase):
    """Rheobase finder """

    def test_path_mod_and_import(self):
        with dview.sync_imports():
            import os
        dview.apply(os.chdir,os.getcwd())

        def modify_path(THIS_DIR):
            this_nu = os.path.join(THIS_DIR,'../../')
            sys.path.insert(0,this_nu)

        THIS_DIR = os.path.dirname(os.getcwd())
        dview.apply(modify_path,THIS_DIR)

        with dview.sync_imports():
            from neuronunit.optimization import get_neab
            from neuronunit.optimization import utilities
            from utilities import VirtualModel
            from neuronunit import tests

        dview.push({'VirtualModel':VirtualModel})

        with dview.sync_imports():
            import nsga_p_for_nb


    def test_evaluation(self):


        import nsga_p_for_nb

        with open('../optimization/complete_dump.p','rb') as handle:
            variables = pickle.load(handle)
            vmpop,pop,pf,history=variables[0],variables[1],variables[2],variables[3]

        def test_pevaluation(self,vmpop,pop):
            fitnesses = list(dview.map(nsga_p_for_nb.evaluate, vmpop))
            for ind, fit in zip(pop, fitnesses):
                ind.fitness.values = fit
            return fitnesses

        def test_sevaluation(self,vmpop,pop):
            serial_fitnesses = []
            for vm in vmpop:
                serial_fitnesses.append(nsga_p_for_nb.evaluate(vm))
            return serial_fitnesses

        parallel_fitnesses = test_pevaluation()
        serial_fitnesses = test_pevaluation()
        assert serial_fitnesses == parallel_fitnesses

'''
from sciunit.utils import NotebookTools

# CONVERT_NOTEBOOKS environment variable controls whether notebooks are
# executed as notebooks or converted to regular python files first.

class DocumentationTestCase(NotebookTools,unittest.TestCase):
    """Testing documentation notebooks"""
    #@unittest.skip("Skipping chapter 1")
    def test_chapter1(self):
        self.do_notebook('explorter_voyager')
'''

if __name__ == '__main__':
    unittest.main()
