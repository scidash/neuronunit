"""Neuronunit-specific model backends."""

import inspect
import sciunit.models.backends as su_backends
from .base import Backend


try:
    from .jNeuroML import jNeuroMLBackend
except ImportError:
    print('Could not load jNeuroMLBackend')

try:
    from .neuron import NEURONBackend
except ImportError:
    print('Silent Error eminating from NEURON syntax')

try:
    from .rawpy import RAWBackend
except ImportError:
    print('Could not load RAWBackend.')

try:
    from .hhrawf import HHBackend
except ImportError:
    print('Could not load HHBackend.')

try:
    from .general_pyNN import HHpyNNBackend
except ImportError:
    print('Could not load HHpyNNBackend.')

available_backends = {x.replace('Backend', ''): cls for x, cls
                      in locals().items()
                      if inspect.isclass(cls) and
                      issubclass(cls, Backend)}

su_backends.register_backends(locals())
