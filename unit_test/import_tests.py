"""Tests of imports of neuronunit submodules and other dependencies"""

from .base import *


class ImportTestCase(unittest.TestCase):
    """Testing imports of modules and packages"""

    def test_import_everything(self):
        import neuronunit
        # Recursively import all submodules
        import_all_modules(neuronunit,
                           skip=['neuroconstruct','optimization',
                                 'backends','unit_test'],
                           verbose=True)



if __name__ == '__main__':
    unittest.main()