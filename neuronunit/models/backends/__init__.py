"""Neuronunit-specific model backends."""

import inspect

import sciunit.models.backends as su_backends
from sciunit.utils import PLATFORM, PYTHON_MAJOR_VERSION
from .base import Backend


try:
    from .jNeuroML import jNeuroMLBackend
except ImportError:
    jNeuroMLBackend = None
    print('Could not load jNeuroMLBackend')

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

try:
    from .general_pyNN import HHpyNNBackend
except ImportError:
    HHpyNNBackend = None
    print('Could not load HHpyNNBackend.')
except AttributeError as e:
    if PLATFORM == 'darwin' and PYTHON_MAJOR_VERSION == 2:
        # Likely a NEURON mod file compilation error.
        pass
    else:
        raise e


available_backends = {x.replace('Backend', ''): cls for x, cls
                      in locals().items()
                      if inspect.isclass(cls) and
                      issubclass(cls, Backend)}

su_backends.register_backends(locals())
