"""Neuronunit-specific model backends."""

import inspect
import warnings

import sciunit.models.backends as su_backends
from sciunit.utils import PLATFORM, PYTHON_MAJOR_VERSION
from .base import Backend

warnings.filterwarnings('ignore', message='nested set')
warnings.filterwarnings('ignore', message='mpi4py')

try:
    from .jNeuroML import jNeuroMLBackend
except ImportError:
    jNeuroMLBackend = None
    print('Could not load jNeuroMLBackend')

try:
    from .jNeuroML import jNeuroMLBackend
except ImportError:
    jNeuroMLBackend = None
    print('Could not load jNeuroMLBackend')

try:
    from .geppetto import GeppettoBackend
except ImportError:
    GeppettoBackend = None
    print('Could not load GeppettoBackend')

try:
    from .neuron import NEURONBackend
except ImportError:
    NEURONBackend = None
    print('Could not load NEURONBackend')

try:
    from .rawpy import RAWBackend
except ImportError:
    RAWBackend = None
    print('Could not load RAWBackend.')

try:
    from .hhrawf import HHBackend
except ImportError:
    HHBackend = None
    print('Could not load HHBackend.')

"""
try:
    from .general_pyNN import HHpyNNBackend
except ImportError:
    HHpyNNBackend = None
    print('Could not load HHpyNNBackend.')
except (AttributeError, IOError) as e:
    if any([x in str(e) for x in ('NEURON', "'hoc.HocObject' object")]):
        print('Could not load PyNNBackend due to NEURON issues: %s' % str(e))
    else:
        raise e
"""

try:
    from .glif import GLIFBackend
except Exception as e:
    print('Could not load GLIFBackend')

available_backends = {x.replace('Backend', ''): cls for x, cls
                      in locals().items()
                      if inspect.isclass(cls) and
                      issubclass(cls, Backend)}

su_backends.register_backends(locals())
