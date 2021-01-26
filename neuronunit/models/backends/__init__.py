"""Neuronunit-specific model backends."""

import inspect

import sciunit.models.backends as su_backends
from sciunit.utils import PLATFORM
from .base import Backend


try:
    from .static import StaticBackend
except ImportError:
    StaticBackend = None
    print('Could not load StaticBackend')

try:
    from .geppetto import GeppettoBackend
except ImportError:
    GeppettoBackend = None
    print('Could not load GeppettoBackend')

try:
    from .jNeuroML import jNeuroMLBackend
except ImportError:
    jNeuroMLBackend = None
    print('Could not load jNeuroMLBackend')

##
# Neuron support depreciated
##
#try:
#    from .neuron import NEURONBackend
#except ImportError:
#    NEURONBackend = None
#    print('Could not load NEURONBackend')


available_backends = {x.replace('Backend', ''): cls for x, cls
                      in locals().items()
                      if inspect.isclass(cls) and
                      issubclass(cls, Backend)}

su_backends.register_backends(locals())
