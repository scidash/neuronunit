"""Unit testing module for NeuronUnit"""

import warnings

import matplotlib as mpl

mpl.use('Agg')  # Needed for headless testing
warnings.filterwarnings('ignore')  # Suppress all warning messages

from .base import *
from .import_tests import ImportTestCase
from .test_high_level import testHighLevelOptimisation
from .test_low_level import testLowLevelOptimisation
from .doc_tests import DocumentationTestCase
from .resource_tests import NeuroElectroTestCase, BlueBrainTestCase,\
                            AIBSTestCase
from .model_tests import ReducedModelTestCase, ExtraCapabilitiesTestCase
from .observation_tests import ObservationsTestCase
from .test_tests import TestsPassiveTestCase, TestsWaveformTestCase,\
                        TestsFITestCase, TestsDynamicsTestCase,\
                        TestsChannelTestCase
from .misc_tests import EphysPropertiesTestCase
from .sciunit_tests import SciUnitTestCase
from .cache_tests import BackendCacheTestCase

from .test_druckmann2013 import Model1TestCase, Model2TestCase, \
    Model3TestCase, Model4TestCase, Model5TestCase, \
    Model6TestCase, Model7TestCase, Model8TestCase, Model9TestCase, \
    Model10TestCase, Model11TestCase
