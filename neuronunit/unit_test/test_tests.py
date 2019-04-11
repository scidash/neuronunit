"""Tests of NeuronUnit test classes."""

import os

from .base import unittest, nu_tests, ReducedModel, NU_BACKEND, NU_HOME
from neuronunit.neuroelectro import is_neuroelectro_up


def quick_test_builder(test_class=None, backend=NU_BACKEND):
    if test_class is None:
        from neuronunit.tests.fi import RheobaseTestP as test_class
    else:
        if isinstance(test_class, str):
            test_class = nu_tests.__dict__[test_class]
    path = os.path.join(NU_HOME, 'models/NeuroML2/LEMS_2007One.xml')
    model = ReducedModel(path, backend=backend)
    observation = test_class.neuroelectro_summary_observation(
                  {'nlex_id': 'nifext_50'})
    test = test_class(observation=observation)
    return model, test


class TestsTestCase(object):
    """Abstract base class for testing tests."""

    def setUp(self):
        from .model_tests import ReducedModelTestCase
        path = ReducedModelTestCase().path
        self.model = ReducedModel(path, backend=NU_BACKEND)
        if not is_neuroelectro_up():
            return self.skipTest("Neuroelectro is down")

    def get_observation(self, cls):
        neuron = {'nlex_id': 'nifext_50'}  # Layer V pyramidal cell
        return cls.neuroelectro_summary_observation(neuron)

    def make_test(self, cls):
        observation = self.get_observation(cls)
        self.test = cls(observation=observation)
        return self.test

    def run_test(self, cls):
        self.make_test(cls)
        score = self.test.judge(self.model)
        score.summarize()
        return score.score


class TestsPassiveTestCase(TestsTestCase, unittest.TestCase):
    """Test passive validation tests."""

    def test_inputresistance(self):
        from neuronunit.tests.passive import InputResistanceTest as T
        score = self.run_test(T)
        self.assertTrue(-0.6 < score < -0.5, "Score was %.2f" % score)

    def test_restingpotential(self):
        from neuronunit.tests.passive import RestingPotentialTest as T
        score = self.run_test(T)
        self.assertTrue(1.2 < score < 1.3, "Score was %.2f" % score)

    def test_capacitance(self):
        from neuronunit.tests.passive import CapacitanceTest as T
        score = self.run_test(T)
        self.assertTrue(-0.25 < score < -0.15, "Score was %.2f" % score)

    def test_timeconstant(self):
        from neuronunit.tests.passive import TimeConstantTest as T
        score = self.run_test(T)
        self.assertTrue(-0.85 < score < -0.75, "Score was %.2f" % score)


class TestsWaveformTestCase(TestsTestCase, unittest.TestCase):
    """Test passive validation tests"""

    def test_ap_width(self):
        from neuronunit.tests.waveform import InjectedCurrentAPWidthTest as T
        score = self.run_test(T)
        self.assertTrue(-0.65 < score < -0.55)

    def test_ap_amplitude(self):
        from neuronunit.tests.waveform import InjectedCurrentAPAmplitudeTest as T
        score = self.run_test(T)
        self.assertTrue(-1.7 < score < -1.6)

    def test_ap_threshold(self):
        from neuronunit.tests.waveform import InjectedCurrentAPThresholdTest as T
        score = self.run_test(T)
        self.assertTrue(2.25 < score < 2.35)


class TestsFITestCase(TestsTestCase, unittest.TestCase):
    """Test F/I validation tests"""

    #@unittest.skip("This test takes a long time")
    def test_rheobase_parallel(self):
        from neuronunit.tests.fi import RheobaseTestP as T
        score = self.run_test(T)
        self.assertTrue(0.2 < score < 0.3)


class TestsDynamicsTestCase(TestsTestCase, unittest.TestCase):
    """Tests dynamical systems properties tests"""

    @unittest.skip("This test is not yet implemented")
    def test_threshold_firing(self):
        from neuronunit.tests.dynamics import TFRTypeTest as T
        #score = self.run_test(T)
        #self.assertTrue(0.2 < score < 0.3)

    @unittest.skip("This test is not yet implemented")
    def test_rheobase_parallel(self):
        from neuronunit.tests.dynamics import BurstinessTest as T
        #score = self.run_test(T)
        #self.assertTrue(0.2 < score < 0.3)


class TestsChannelTestCase(unittest.TestCase):
    @unittest.skip("This test is not yet implemented")
    def test_iv_curve_ss(self):
        from neuronunit.tests.channel import IVCurveSSTest as T
        #score = self.run_test(T)
        #self.assertTrue(0.2 < score < 0.3)

    @unittest.skip("This test is not yet implemented")
    def test_iv_curve_peak(self):
        from neuronunit.tests.channel import IVCurvePeakTest as T
        #score = self.run_test(T)
        #self.assertTrue(0.2 < score < 0.3)


if __name__ == '__main__':
    unittest.main()
