import inspect

from .base import Backend

try:
    from .jNeuroML import jNeuroMLBackend
except:
    print('Error in jNeuroMLBackend')

try:
    from .neuron import NEURONBackend
except Exception as e:
    import pdb
    print('Silent Error eminating from NEURON syntax')

    #pdb.set_trace()
try:
    from .rawpy import RAWBackend
except Exception as e:
    print('raw python Error')
try:
    from .hhrawf import HHBackend
except Exception as e:
    print('HH python Error')

try:
    from .glif import GLIFBackend
except Exception as e:
    print('glif python Error')

try:
    from .general_pyNN import PYNNBackend
except Exception as e:
    print('pynn python Error')

try:
    from .badexp import ADEXPBackend
except Exception as e:
    print('pynn python Error')



available_backends = {x.replace('Backend',''):cls for x, cls \
                   in locals().items() \
                   if inspect.isclass(cls) and \
                   issubclass(cls, Backend)}




# try:
#    from .external import ExternalSim
# except Exception as e:
#    print('External sim python Error')
