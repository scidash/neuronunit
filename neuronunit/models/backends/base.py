"""Simulator backends for NeuronUnit models"""
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
from sciunit.models.backends import Backend, BackendException
from sciunit.utils import dict_hash, import_module_from_path, \
                          TemporaryDirectory

# Test for NEURON support in a separate python process
NEURON_SUPPORT = (os.system("python -c 'import neuron' > /dev/null 2>&1") == 0)  

try:
    import pyNN
    pyNN_SUPPORT = True
except:
    pyNN = None
    pyNN_SUPPORT = False
