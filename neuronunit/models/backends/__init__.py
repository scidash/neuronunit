"""Neuronunit-specific model backends."""

import sciunit.models.backends as su_backends

try:
    from .jNeuroML import jNeuroMLBackend
except Exception as e:
    raise e
    pass

try:
    from .neuron import NEURONBackend
except Exception:
    pass

try:
    from .pyNN import pyNNBackend
except Exception:
    pass

su_backends.register_backends(locals())
