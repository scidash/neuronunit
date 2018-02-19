import inspect

from .base import Backend
from .jNeuroML import jNeuroMLBackend
from .neuron import NEURONBackend
from .pyNN import pyNNBackend
#from .section_extension import section_extension


available_backends = {x.replace('Backend',''):cls for x, cls \
                   in locals().items() \
                   if inspect.isclass(cls) and \
                   issubclass(cls, Backend)}
