import inspect

from .base import Backend

try:
    from .jNeuroML import jNeuroMLBackend
except:
    pass

try:
    from .neuron import NEURONBackend
except:
    pass

try:
    from .pyNN import pyNNBackend
except:
    pass

available_backends = {x.replace('Backend',''):cls for x, cls \
                   in locals().items() \
                   if inspect.isclass(cls) and \
                   issubclass(cls, Backend)}
