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
from sciunit.models.backends import Backend as SU_Backend, BackendException
from sciunit.utils import dict_hash, import_module_from_path

try:
    import neuron
    from neuron import h
    NEURON_SUPPORT = True
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
    

class Backend(SU_Backend):
    
    name = 'Unimplemented'
    
    def inject_square_current(self, current):
        """Inject a square current into the cell."""
        raise NotImplementedError("")

    def set_stop_time(self, t_stop):
        """Set the stop time of the simulation."""
        raise NotImplementedError("")

    def set_time_step(self, dt):
        """Set the time step of the simulation."""
        raise NotImplementedError("")


class EmptyBackend(Backend):
    
    name = 'Empty'
    
    def inject_square_current(self, current):
        """Inject a square current into the cell."""
        pass

    def set_stop_time(self, t_stop):
        """Set the stop time of the simulation."""
        pass

    def set_time_step(self, dt):
        """Set the time step of the simulation."""
        pass