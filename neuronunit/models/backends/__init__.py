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
        print('Error in jNeuroMLBackend')

    try:
        from .neuron import NEURONBackend
    except ImportError:
        NEURONBackend = None
        print('Could not load NEURONBackend')
    try:
        from .general_pyNN import PYNNBackend
    except Exception as e:
        print('pynn python Error')

    try:
        from .glif import GLIFBackend
    except Exception as e:
        print('glif python Error')



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


try:
    from .badexp import ADEXPBackend
except Exception as e:
    print('pynn python Error')

try:
    from .bhh import BHHBackend
except Exception as e:
    print('could not import brian2 neuronaldynamicsError')

available_backends = {x.replace('Backend',''):cls for x, cls \
                   in locals().items() \
                   if inspect.isclass(cls) and \
                   issubclass(cls, Backend)}


available_backends = {x.replace('Backend', ''): cls for x, cls
                      in locals().items()
                      if inspect.isclass(cls) and
                      issubclass(cls, Backend)}

su_backends.register_backends(locals())
