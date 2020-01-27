"""Neuronunit-specific model backends."""

import contextlib
import io
import importlib
import inspect
import pathlib
import re
import warnings

import sciunit.models.backends as su_backends
from sciunit.utils import PLATFORM, PYTHON_MAJOR_VERSION
from .base import Backend

warnings.filterwarnings('ignore', message='nested set')
warnings.filterwarnings('ignore', message='mpi4py')


backend_paths = ['jNeuroML.jNeuroMLBackend',
                 'neuron.NEURONBackend',
                 'general_pyNN.PYNNBackend',
                 'hh_wraper.JHHBackend',
                 'rawpy.RAWBackend',
                 'hhrawf.HHBackend',
                 'glif.GLIFBackend',
                 'badexp.ADEXPBackend',
                 'bhh.BHHBackend',
                ]

def check_backend(partial_path):
    full_path = 'neuronunit.models.backends.%s' % partial_path
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
        return None
    else:
        return (backend.name, backend)
    
available_backends = {}
for backend_path in backend_paths:
    result = check_backend(backend_path)
    if result is not None:
        name, backend = result
        available_backends[name] = backend

#su_backends.register_backends(available_backends)