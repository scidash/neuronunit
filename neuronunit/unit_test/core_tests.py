"""Unit tests for the core of NeuronUnit"""

# Run with:  
# python -m unittest -bv unit_test.core_tests.py
# coverage run --source . core_tests.py

from .base import *

from .import_tests import ImportTestCase
from .doc_tests import DocumentationTestCase
from .resource_tests import NeuroElectroTestCase,BlueBrainTestCase,AIBSTestCase
from .model_tests import ReducedModelTestCase
from .test_tests import TestsPassiveTestCase,TestsWaveformTestCase,\
                        TestsFITestCase,TestsDynamicsTestCase,\
                        TestsChannelTestCase
from .misc_tests import EphysPropertiesTestCase
from .sciunit_tests import SciUnitTestCase


if __name__ == '__main__':
    unittest.main()
