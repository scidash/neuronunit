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
from quantities import ms, mV, nA
from pyneuroml import pynml
from quantities import ms, mV
from neo.core import AnalogSignal
import neuronunit.capabilities.spike_functions as sf
import sciunit
from sciunit.utils import dict_hash, import_module_from_path
try:
    import neuron
    from neuron import h
    NEURON_SUPPORT = True
except:
    NEURON_SUPPORT = False


class Backend(object):
    """Base class for simulator backends that implement simulator-specific
    details of modifying, running, and reading results from the simulation
    """

    def init_backend(self, *args, **kwargs):
        self.model.attrs = {}

        self.use_memory_cache = kwargs.get('use_memory_cache', True)
        if self.use_memory_cache:
            self.init_memory_cache()
        self.use_disk_cache = kwargs.get('use_disk_cache', False)
        if self.use_disk_cache:
            self.init_disk_cache()
        self.load_model()
        self.model.unpicklable += ['_backend']

    # Name of the backend
    backend = None

    #The function (e.g. from pynml) that handles running the simulation
    f = None

    def init_cache(self):
        self.init_memory_cache()
        self.init_disk_cache()

    def init_memory_cache(self):
        self.memory_cache = {}

    def init_disk_cache(self):
        try:
            # Cleanup old disk cache files
            path = self.disk_cache_location
            os.remove(path)
        except:
            pass
        self.disk_cache_location = os.path.join(tempfile.mkdtemp(),'cache')

    def get_memory_cache(self, key):
        """Returns result in memory cache for key 'key' or None if it
        is not found"""
        self._results = self.memory_cache.get(key)
        return self._results

    def get_disk_cache(self, key):
        """Returns result in disk cache for key 'key' or None if it
        is not found"""
        if not getattr(self,'disk_cache_location',False):
            self.init_disk_cache()
        disk_cache = shelve.open(self.disk_cache_location)
        self._results = disk_cache.get(key)
        disk_cache.close()
        return self._results

    def set_memory_cache(self, results, key=None):
        """Stores result in memory cache with key
        corresponding to model state"""
        key = self.model.hash if key is None else key
        self.memory_cache[key] = results

    def set_disk_cache(self, results, key=None):
        """Stores result in disk cache with key
        corresponding to model state"""
        if not getattr(self,'disk_cache_location',False):
            self.init_disk_cache()
        disk_cache = shelve.open(self.disk_cache_location)
        key = self.model.hash if key is None else key
        disk_cache[key] = results
        disk_cache.close()

    def set_attrs(self, **attrs):
        """Set model attributes, e.g. input resistance of a cell"""
        #If the key is in the dictionary, it updates the key with the new value.
        self.model.attrs.update(attrs)
        #pass

    def set_run_params(self, **params):
        """Set run-time parameters, e.g. the somatic current to inject"""
        self.model.run_params.update(params)
        self.check_run_params()
        #pass

    def check_run_params(self):
        """Check to see if the run parameters are reasonable for this model
        class.  Raise a sciunit.BadParameterValueError if any of them are not.
        """
        pass

    def load_model(self):
        """Load the model into memory"""
        pass

    def local_run(self):
        """Checks for cached results in memory and on disk, then runs the model
        if needed"""
        key = self.model.hash
        if self.use_memory_cache and self.get_memory_cache(key):
            return self._results
        if self.use_disk_cache and self.get_disk_cache(key):
            return self._results
        results = self._local_run()
        if self.use_memory_cache:
            self.set_memory_cache(results, key)
        if self.use_disk_cache:
            self.set_disk_cache(results, key)
        return results

    def _local_run(self):
        """Runs the model via the backend"""
        raise NotImplementedError("Each backend must implement '_local_run'")

    def save_results(self, path='.'):
        with open(path,'wb') as f:
            pickle.dump(self.results,f)


class BackendException(Exception):
    pass
