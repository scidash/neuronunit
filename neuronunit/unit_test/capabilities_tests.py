import unittest

class CapabilitiesTestCases(unittest.TestCase):
    def test_produces_swc(self):
        from neuronunit.capabilities.morphology import ProducesSWC
        p = ProducesSWC()
        self.assertEqual(NotImplementedError, p.produce_swc().__class__)


if __name__ == '__main__':
    unittest.main()
