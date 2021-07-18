"""Unit testing module for NeuronUnit"""

import warnings

import matplotlib as mpl

mpl.use("Agg")  # Needed for headless testing
warnings.filterwarnings("ignore")  # Suppress all warning messages

from .base import *
from .import_tests import ImportTestCase
from .doc_tests import DocumentationTestCase
from .resource_tests import NeuroElectroTestCase, BlueBrainTestCase, AIBSTestCase
from .model_tests import (
    ReducedModelTestCase,
    ExtraCapabilitiesTestCase,
    HasSegmentTestCase,
    GeppettoBackendTestCase,
    VeryReducedModelTestCase,
    StaticExternalTestCase,
)

# from .observation_tests import ObservationsTestCase
"""
from .test_tests import (
    TestsPassiveTestCase,
    TestsWaveformTestCase,
    TestsFITestCase,
    TestsDynamicsTestCase,
    TestsChannelTestCase,
    FakeTestCase,
)
"""
from .misc_tests import EphysPropertiesTestCase

# from .cache_tests import BackendCacheTestCase
<<<<<<< HEAD
from .opt_ephys_properties import testOptimizationEphysCase
from .scores_unit_test import testOptimizationAllenMultiSpike
=======
#from .opt_ephys_properties import testOptimizationEphysCase
#from .scores_unit_test import testOptimizationAllenMultiSpike
from .rheobase_model_test import testModelRheobase
>>>>>>> 9fb0c2e613a1bf059f38eeeae80582d0cfb11f2f

# from .adexp_opt import *
"""
from .capabilities_tests import *

from .test_druckmann2013 import (
    Model1TestCase,
    Model2TestCase,
    Model3TestCase,
    Model4TestCase,
    Model5TestCase,
    Model6TestCase,
    Model7TestCase,
    Model8TestCase,
    Model9TestCase,
    Model10TestCase,
    Model11TestCase,
    OthersTestCase,
)
"""
# from .test_morphology import MorphologyTestCase
# from .test_optimization import DataTCTestCase
# from .sciunit_tests import SciUnitTestCase

# from .adexp_opt import *
