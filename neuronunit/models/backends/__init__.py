"""Neuronunit-specific model backends."""

import inspect
import warnings

import sciunit.models.backends as su_backends
from sciunit.utils import PLATFORM, PYTHON_MAJOR_VERSION
from .base import Backend

warnings.filterwarnings('ignore', message='nested set')
warnings.filterwarnings('ignore', message='mpi4py')

def heavy_backends():
    try:
        from .jNeuroML import jNeuroMLBackend
    except:
        print('Could not load jNeuroML backend')

    try:
        from .neuron import NEURONBackend
    except ImportError:
        NEURONBackend = None
        print('Could not load NEURON backend')
    try:
        from .general_pyNN import PYNNBackend
    except Exception as e:
        print('Could not load PyNN backend')

heavy_backends()

"""
try:
    from .hh_wraper import JHHBackend
except ImportError:
    JHHBackend = None
    print('Could not load JHHBackend.')
"""

try:
    from .rawpy import RAWBackend
except ImportError:
    RAWBackend = None
    print('Could not load RAW Backend.')

try:
    from .hhrawf import HHBackend
except ImportError:
    HHBackend = None
    print('Could not load HH Backend.')

try:
    from .glif import GLIFBackend
except Exception as e:
    GLIFBackend = None
    print('Could not load GLIF Backend')

try:
    from .badexp import ADEXPBackend
except Exception as e:
    ADEXPBackend = None
    print('Could not load BRIAN Adaptive Exponentional backend')

try:
    from .bhh import BHHBackend
except Exception as e:
    BHHBackend = None
    print('Could not load Brian HH backend')

available_backends = {x.replace('Backend', ''): cls for x, cls
                      in locals().items()
                      if inspect.isclass(cls) and
                      issubclass(cls, Backend)}

su_backends.register_backends(locals())