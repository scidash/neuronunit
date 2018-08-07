"""Tests of NeuronUnit model classes"""

from .base import *

class ReducedModelTestCase(unittest.TestCase):
    """Test instantiation of the reduced model"""

    def setUp(self):
        from neuronunit.models.reduced import ReducedModel
        self.ReducedModel = ReducedModel

    def runTest(self):
        pass # Needed so that Python<3 can access the path attribute.

    @property
    def path(self):
        result = os.path.join(__file__,'..','..','models',
                              'NeuroML2','LEMS_2007One.xml')
        result = os.path.realpath(result)
        return result

    def test_reducedmodel_jneuroml(self):
        model = self.ReducedModel(self.path, backend='jNeuroML')

    @unittest.skip("Ignoring NEURON until we make it an install requirement")#If(OSX,"NEURON unreliable on OSX")
    def test_reducedmodel_neuron(self):
        model = self.ReducedModel(self.path, backend='NEURON')


class ExtraCapabilitiesTestCase(NotebookTools,
                           unittest.TestCase):
    """Testing extra capability checks"""

    path = '.'

    def test_receives_current(self):
        self.do_notebook('nml_extra_capability_check')
        

if __name__ == '__main__':
    unittest.main()
