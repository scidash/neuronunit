"""Unit tests for the core of NeuronUnit"""

# Run with any of:  
# python core_tests.py
# python -m unittest core_tests.py
# coverage run --source . core_tests.py

import unittest
import sys
import os
import warnings
import pdb

try: # Python 2
    from urllib import urlretrieve
except ImportError: # Python 3
    from urllib.request import urlretrieve

if sys.platform == 'darwin':
    import matplotlib as mpl
    mpl.use('Agg')

from sciunit.utils import NotebookTools
from neuronunit import neuroelectro,bbp


class DocumentationTestCase(NotebookTools, 
                            unittest.TestCase):
    """Testing documentation notebooks"""

    path = '../docs'

    #@unittest.skip("Skipping chapter 1")
    def test_chapter1(self):
        self.do_notebook('chapter1')

class EphysPropertiesTestCase(NotebookTools, 
                              unittest.TestCase):
    """Testing sciunit tests of ephys properties"""

    path = '.'

    #@unittest.skip("Skipping get_tau test")
    def test_get_tau(self):
        self.do_notebook('get_tau')


class NeuroElectroTestCase(unittest.TestCase):
    """Testing retrieval of data from NeuroElectro.org"""

    @unittest.skipUnless(neuroelectro.is_neuroelectro_up(), 
                         "NeuroElectro URL is not responsive")
    def test_neuroelectro(self):
        x = neuroelectro.NeuroElectroDataMap()
        x.set_neuron(id=72)
        x.set_ephysprop(id=2)
        x.set_article(pmid=18667618)
        x.get_values()
        x.check()

        x = neuroelectro.NeuroElectroSummary()
        x.set_neuron(id=72)
        x.set_ephysprop(id=2)
        x.get_values()
        x.check()
        

class BlueBrainTestCase(NotebookTools,
                        unittest.TestCase):
     
    path = '.'
    
    @unittest.skipUnless(bbp.is_bbp_up(), 
                         "Blue Brain URL is not responsive")
    def test_bluebrain(self):
        self.do_notebook('bbp')
        

if __name__ == '__main__':
    unittest.main()
