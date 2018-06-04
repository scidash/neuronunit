"""Unit tests for the core of NeuronUnit"""

# Run with:
# python -m unittest -bv unit_test.core_tests.py
# coverage run --source . core_tests.py

#from .base import *

from test_optimization import testOptimizationBackend
tob = testOptimizationBackend()
#tob.main()
from test_backends import TestBackend
tbe = TestBackend()
#tbe.main()



if __name__ == '__main__':
    unittest.main()
