import inspect

from .base import Backend

try:
    from .jNeuroML import jNeuroMLBackend
except:
    print('Error in jNeuroMLBackend')

try:
    from .neuron import NEURONBackend
except Exception as e:
    print('Silent Error eminating from NEURON syntax')

try:
    from .rawpy import RAWBackend
except Exception as e:
    print('raw python Error')

available_backends = {x.replace('Backend',''):cls for x, cls \
                   in locals().items() \
                   if inspect.isclass(cls) and \
                   issubclass(cls, Backend)}
#print(available_backends)
