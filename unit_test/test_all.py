"""Unit tests for NeuronUnit"""

# Run with any of:  
# python test_all.py
# python -m unittest test_all.py

# coverage run --source . test_all.py

import unittest
import sys
import os
import warnings

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

# CONVERT_NOTEBOOKS environment variable controls whether notebooks are
# executed as notebooks or converted to regular python files first. 


class DocumentationTestCase(NotebookTools,unittest.TestCase):
    """Testing documentation notebooks"""

    path = 'docs'

    #@unittest.skip("Skipping chapter 1")
    def test_chapter1(self):
        self.do_notebook('chapter1')

    #@unittest.skip("Skipping chapter 2")
    def test_chapter2(self):
        self.do_notebook('chapter2')

    #@unittest.skip("Skipping chapter 3")
    def test_chapter3(self):
        self.do_notebook('chapter3')

    @unittest.skip("Skipping chapter 4")
    def test_chapter4(self):
        self.do_notebook('chapter4')


class SciunitTestTestCase(NotebookTools,unittest.TestCase):
    """Testing documentation notebooks"""

    #@unittest.skip("Skipping get_tau test")
    def test_get_tau(self):
        self.do_notebook('get_tau')


class NeuroelectroTestCase(unittest.TestCase):
    def test_neuroelectro():
        from neuronunit.neuroelectro import NeuroElectroDataMap,\
                                            NeuroElectroSummary
        x = NeuroElectroDataMap()
        x.set_neuron(id=72)
        x.set_ephysprop(id=2)
        x.set_article(pmid=18667618)
        x.get_values()
        x.check()

        x = NeuroElectroSummary()
        x.set_neuron(id=72)
        x.set_ephysprop(id=2)
        x.get_values()
        x.check()


if __name__ == '__main__':
    unittest.main()
