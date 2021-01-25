"""Tests of NeuronUnit model classes"""

from .base import *
import quantities as pq
import numpy as np
from neo.core import AnalogSignal

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

class GeppettoBackendTestCase(unittest.TestCase):
    """Testing GeppettoBackend"""

    def test_geppetto_backend(self):
        from neuronunit.models.backends.geppetto import GeppettoBackend
        gb = GeppettoBackend()
        #gb.init_backend()
        gb._backend_run()

class HasSegmentTestCase(unittest.TestCase):
    """Testing HasSegment and SingleCellModel"""

    def test_(self):
        from neuronunit.models.section_extension import HasSegment
        hs = HasSegment()

        def section(location):
            return location

        hs.setSegment(section)
        self.assertEqual(hs.getSegment(), 0.5)

class VeryReducedModelTestCase(unittest.TestCase):
    def setUp(self):
        from sciunit.models.backends import Backend
        
        class MyBackend(Backend):
            def _backend_run(self) -> str:
                return sum(self.run_params.items())

            def local_run(self):
                return

            def set_run_params(self, **params):
                self.run_params = params

            def inject_square_current(*args, **kwargs):
                pass

        self.backend = MyBackend
        from sciunit.models.backends import register_backends
        register_backends({"My" : self.backend})

    def test_very_reduced_using_lems(self):
        from neuronunit.models.reduced import VeryReducedModel
        
        vrm = VeryReducedModel(name="test very redueced model", backend="My", attrs={})
        
        vrm.rerun = False
        vrm.run_defaults = {"param3": 3}
        vrm.set_default_run_params(param1=1)

        vrm.set_attrs(a="a")
        vrm.get_membrane_potential()
        vrm.get_APs()
        vrm.get_spike_train()
        vrm.inject_square_current(0)

        self.assertIsInstance(vrm.get_backend(), self.backend)
        vrm.run(param2=2)

    def test_very_reduced_not_using_lems(self):
        from neuronunit.models.very_reduced import VeryReducedModel
        import quantities as pq
        class MyVRM(VeryReducedModel):
            def run(self):
                self.results = {
                    'vm': [0.01*pq.V, 0.05*pq.V, 0*pq.V], 
                    't': [0.1*pq.s, 0.2*pq.s, 0.3*pq.s]
                }
        vrm = MyVRM(name="test very redueced model", backend="My", attrs={})
        vrm.get_APs()
        vrm.get_spike_train()
        vrm.inject_square_current(0.01 * pq.mA)
        pass


class StaticExternalTestCase(unittest.TestCase):
    def setUp(self):
        array = np.ones(10000) * -60.0 * pq.mV
        dt = 1 * pq.ms
        self.vm = AnalogSignal(array,units=pq.mV,sampling_rate=1.0/dt)

    def test_external_model(self):
        from neuronunit.models.static import ExternalModel
        import quantities as pq
        em = ExternalModel(name='test external model obj')
        em.set_membrane_potential(self.vm)
        em.set_model_attrs({'a': 1, 'b': 2, 'c': 3})
        vm = em.get_membrane_potential()
        self.assertIsInstance(vm, AnalogSignal)

    def test_static_model(self):
        from neuronunit.models.static import StaticModel
        with self.assertRaises(TypeError):
            sm = StaticModel(None)
        sm = StaticModel(self.vm)
        sm.inject_square_current(0.1 * pq.mA)

if __name__ == '__main__':
    unittest.main()
