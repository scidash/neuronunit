import unittest

class CapabilitiesTestCases(unittest.TestCase):

    def test_produces_swc(self):
        from neuronunit.capabilities.morphology import ProducesSWC
        p = ProducesSWC()
        self.assertEqual(NotImplementedError, p.produce_swc().__class__)

    def test_produces_membrane_potential(self):
        from neuronunit.capabilities import ProducesMembranePotential
        pmp = ProducesMembranePotential()
        with self.assertRaises(NotImplementedError):
            pmp.get_membrane_potential()
            pmp.get_mean_vm()
            pmp.get_median_vm()
            pmp.get_std_vm()
            pmp.get_iqr_vm()
            pmp.get_initial_vm()
            pmp.plot_membrane_potential()

        from neo.core import AnalogSignal
        class MyPMP(ProducesMembranePotential):
            def get_membrane_potential(self, signal=[[1, 2, 3], [4, 5, 6]], units='V'):
                
                import quantities as pq
                result = AnalogSignal(signal, units, sampling_rate=1*pq.Hz)
                return result

        my_pmp = MyPMP()
        self.assertIsInstance(my_pmp.get_membrane_potential(), AnalogSignal)
        self.assertEqual(my_pmp.get_mean_vm().item(), 3.5)
        self.assertEqual(my_pmp.get_median_vm().item(), 3.5)
        self.assertAlmostEqual(my_pmp.get_std_vm().item(), 1.70782, 4)
        self.assertEqual(my_pmp.get_iqr_vm().item(), 2.5)
        self.assertEqual(my_pmp.get_mean_vm().item(), 3.5)

        import quantities as pq
        self.assertEqual(my_pmp.get_initial_vm()[0], 1* pq.V)
        self.assertEqual(my_pmp.get_initial_vm()[1], 2* pq.V)
        self.assertEqual(my_pmp.get_initial_vm()[2], 3* pq.V)

if __name__ == '__main__':
    unittest.main()
