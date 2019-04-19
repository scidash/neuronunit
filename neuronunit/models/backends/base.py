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

try:
    # Never import neuron in the current directory, or it will automatically
    # load mechanisms in that directory, which will then cause future calls
    # to load_mechanisms() to fail due to already loaded mechanisms.
    temp = TemporaryDirectory()
    curr = os.getcwd()
    os.chdir(temp.name)
    import neuron
    from neuron import h
    NEURON_SUPPORT = True
    os.chdir(curr)
    temp.cleanup()
except:
    neuron = None
    h = None
    NEURON_SUPPORT = False

try:
    import pyNN
    pyNN_SUPPORT = True
except:
    pyNN = None
    pyNN_SUPPORT = False
