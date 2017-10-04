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

import matplotlib as mpl
mpl.use('Agg') # Avoid any problems with Macs or headless displays.

from sciunit.utils import NotebookTools,import_all_modules
from neuronunit import neuroelectro,bbp,aibs

OSX = sys.platform == 'darwin'

class ImportTestCase(unittest.TestCase):
    """Testing imports of modules and packages"""

    def test_import_everything(self):
        import neuronunit
        # Recursively import all submodules
        import_all_modules(neuronunit)


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


class AIBSTestCase(unittest.TestCase):

    @unittest.skipUnless(aibs.is_aibs_up(), 
                         "AIBS URL is not responsive")
    def test_aibs(self):
        dataset_id = 354190013  # Internal ID that AIBS uses for a particular Scnn1a-Tg2-Cre 
                                # Primary visual area, layer 5 neuron.
        observation = aibs.get_observation(dataset_id,'rheobase')


class ReducedModelTestCase(unittest.TestCase):
    """Test instantiation of the reduced model"""

    def setUp(self):
        from neuronunit.models.reduced import ReducedModel
        self.ReducedModel = ReducedModel
        
    def runTest(self):
        pass # Needed so that Python<3 can access the path attribute.  

    @property
    def path(self):
        result = os.path.join(__file__,'..','..','neuronunit',
                              'models','NeuroML2','LEMS_2007One.xml')
        result = os.path.realpath(result)
        return result

    def test_reducedmodel_jneuroml(self):
        model = self.ReducedModel(self.path, backend='jNeuroML')

    @unittest.skip("Ignoring NEURON until we make it an install requirement")#If(OSX,"NEURON unreliable on OSX")
    def test_reducedmodel_neuron(self):
        model = self.ReducedModel(self.path, backend='NEURON')


class TestsTestCase(object):
    """Abstract base class for testing tests"""

    def setUp(self):
        #from neuronunit import neuroelectro
        from neuronunit.models.reduced import ReducedModel
        path = ReducedModelTestCase().path
        self.model = ReducedModel(path, backend='jNeuroML')

    def get_observation(self, cls):
        print(cls.__name__)
        neuron = {'nlex_id': 'nifext_50'} # Layer V pyramidal cell
        return cls.neuroelectro_summary_observation(neuron)

    def run_test(self, cls):
        observation = self.get_observation(cls)
        test = cls(observation=observation)
        score = test.judge(self.model)
        score.summarize()
        return score.score


class TestsPassiveTestCase(TestsTestCase, unittest.TestCase):
    """Test passive validation tests"""

    def test_inputresistance(self):
        from neuronunit.tests.passive import InputResistanceTest
        score = self.run_test(InputResistanceTest)
        self.assertTrue(-0.6 < score < -0.5)

    def test_restingpotential(self):
        from neuronunit.tests.passive import RestingPotentialTest
        score = self.run_test(RestingPotentialTest)
        self.assertTrue(1.2 < score < 1.3)
        

class TestsFITestCase(TestsTestCase, unittest.TestCase):
    """Test F/I validation tests"""

    @unittest.skip("This test takes a long time")
    def test_rheobase_serial(self):
        from neuronunit.tests.fi import RheobaseTest
        score = self.run_test(RheobaseTest)
        self.assertTrue(0.2 < score < 0.3)


if __name__ == '__main__':
    unittest.main()
