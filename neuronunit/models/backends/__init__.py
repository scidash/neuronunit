import inspect

from .base import Backend

try:
    from .jNeuroML import jNeuroMLBackend
except:
    print('Error in jNeuroMLBackend')

'''
try:
    from .neuron import NEURONBackend
except Exception as e:
    print('Silent Error eminating from NEURON syntax')
'''
try:
    from .rawpy import RAWBackend
except Exception as e:
    print('raw python Error')
try:
    from .hhrawf import HHBackend
except Exception as e:
    print('HH python Error')


available_backends = {x.replace('Backend',''):cls for x, cls \
                   in locals().items() \
                   if inspect.isclass(cls) and \
                   issubclass(cls, Backend)}

# try:
#    from .general_pyNN import HHpyNNBackend
    #general_pyNN
# except Exception as e:
#   print('HHpyBackend python Error')



# try:
#    from .external import ExternalSim
# except Exception as e:
#    print('External sim python Error')

# try:
#     from .neuron import NEURONBackend
# except Exception as e:
#     print('Silent Error eminating from NEURON syntax')

#print(available_backends)
