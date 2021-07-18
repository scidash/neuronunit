"""Simulator backends for NeuronUnit models"""
import io
import contextlib
import sys
import os
import platform
import re
import copy
import tempfile
import pickle
import importlib
import shelve
import subprocess

import neuronunit.capabilities as cap
import quantities as pq
from pyneuroml import pynml
from neo.core import AnalogSignal
import neuronunit.capabilities.spike_functions as sf
import sciunit
from sciunit.models.backends import Backend, BackendException, available_backends
from sciunit.models.backends import register_backends as su_register_backends
from sciunit.utils import dict_hash, import_module_from_path

# Test for NEURON support in a separate python process
NEURON_SUPPORT = (os.system("python -c 'import neuron' > /dev/null 2>&1") == 0)
PYNN_SUPPORT = (os.system("python -c 'import pyNN' > /dev/null 2>&1") == 0)


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
        return (None, None)
    else:
        return (backend.name, backend)

def nu_register_backends(backend_paths):
    provided_backends = {}
    for partial_path in backend_paths:
        name, backend = check_backend(partial_path)
        if name is not None:
            provided_backends[name] = backend
    su_register_backends(provided_backends)
