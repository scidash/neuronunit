import inspect

from .base import Backend

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
