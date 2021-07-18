"""Neuronunit-specific model backends."""

import contextlib
import io
import importlib
import inspect
<<<<<<< HEAD
=======
import pathlib
import re
import warnings
>>>>>>> 9fb0c2e613a1bf059f38eeeae80582d0cfb11f2f

import sciunit.models.backends as su_backends
from sciunit.utils import PLATFORM, PYTHON_MAJOR_VERSION
from .base import Backend


<<<<<<< HEAD
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

=======

backend_paths = ['adexp.JIT_ADEXPBackend',
                 'izhikevich.JIT_IZHIBackend']
def check_backend(partial_path):
    full_path = 'jithub.models.backends.%s' % partial_path
    class_name = full_path.split('.')[-1]
    module_path = '.'.join(full_path.split('.')[:-1])
    try:
        backend_stdout = io.StringIO()
        with contextlib.redirect_stdout(backend_stdout):
            module = importlib.import_module(module_path)
            backend = getattr(module, class_name)
    except Exception as e:
        msg = "Import of %s failed due to:" % partial_path
        stdout = backend_stdout.read()
        if stdout:
            msg += '\n%s' % stdout
        msg += '\n%s' % e
        print(msg)
        #warnings.warn(msg)
        return (None, None)
    else:
        return (backend.name, backend)

def register_backends(backend_paths):
    provided_backends = {}
    for partial_path in backend_paths:
        name, backend = check_backend(partial_path)
        if name is not None:
            provided_backends[name] = backend
    su_backends.register_backends(provided_backends)


try:
    register_backends(backend_paths)
>>>>>>> 9fb0c2e613a1bf059f38eeeae80582d0cfb11f2f

except:
    register_backends(backend_paths)

available_backends = su_backends.available_backends
#from .adexp import ADEXPBackend
#from .glif import GLIFBackend
#from .l5pcSciUnit import L5PCBackend
