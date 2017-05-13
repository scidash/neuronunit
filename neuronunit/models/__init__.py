import os
from copy import deepcopy
import tempfile
import shutil
import inspect
import types

import numpy as np
import sciunit
import neuronunit.capabilities as cap
from pyneuroml import pynml
from neo.core import AnalogSignal
from quantities import ms,mV,Hz

from .channel import *
from . import backends



class LEMSModel(sciunit.Model, cap.Runnable):
    """A generic LEMS model"""

    def __init__(self, LEMS_file_path=None, name=None, backend=None, attrs={}):
        """
        LEMS_file_path: Path to LEMS file (an xml file).
        name: Optional model name.
        """

        super(LEMSModel,self).__init__(name=name)


        self.orig_lems_file_path = LEMS_file_path
        self.create_lems_file(name,attrs)
        self.run_defaults = pynml.DEFAULTS
        self.run_defaults['nogui'] = True
        self.run_params = {}
        self.last_run_params = {}
        self.skip_run = False
        self.rerun = True # Needs to be rerun since it hasn't been run yet!
        if name is None:
            name = os.path.split(self.lems_file_path)[1].split('.')[0]
        self.set_backend(backend)
        self.load_model()
        self.attrs={}

    #This is the part that decides if it should inherit from NEURON backend.

    def set_backend(self, backend):
        if type(backend) is str:
            name = backend
            args = []
            kwargs = {}
        elif type(backend) in (tuple,list):
            name = backend[0]
            args = backend[1]
            kwargs = backend[2]
        else:
            raise "Backend must be string, tuple, or list"
        options = {x.replace('Backend',''):cls for x, cls \
                   in backends.__dict__.items() \
                   if inspect.isclass(cls) and \
                   issubclass(cls, backends.Backend)}
        if name in options:
            self.backend = name
            self._backend = options[name](*args,**kwargs)
            # Add all of the backend's methods to the model instance
            #self.__class__.__bases__ = tuple(set((self._backend.__class__,) + \
            #                            self.__class__.__bases__))
            if self._backend.__class__ not in self.__class__.__bases__:
                self.__class__.__bases__ = (self._backend.__class__,) + \
                                        self.__class__.__bases__

        elif name is None:
            # The base class should not be called.
            raise Exception(("A backend (e.g. 'jNeuroML' or 'NEURON') "
                             "must be selected"))
        else:
            raise Exception("Backend %s not found in backends.py" \
                            % backend_name)

    def create_lems_file(self, name, attrs):
        if not hasattr(self,'temp_dir'):
            self.temp_dir = tempfile.gettempdir()
        self.lems_file_path  = os.path.join(self.temp_dir, '%s.xml' % name)
        shutil.copy2(self.orig_lems_file_path, self.lems_file_path)
        if attrs:
            self.set_lems_attrs(attrs)

    def set_lems_attrs(self, attrs):
        from lxml import etree
        tree = etree.parse(self.lems_file_path)
        for key1,value1 in attrs.items():
            nodes = tree.findall(key1)
            for node in nodes:
                for key2,value2 in value1.items():
                    node.attrib[key2] = value2
        tree.write(self.lems_file_path)

    def run(self, rerun=None, **run_params):
        if rerun is None:
            rerun = self.rerun
        self.run_params.update(run_params)
        for key,value in self.run_defaults.items():
            if key not in self.run_params:
                self.run_params[key] = value
        if (not rerun) and hasattr(self,'last_run_params') and \
           self.run_params == self.last_run_params:
            return

        self.update_run_params(run_params)
        #self.update_run_params(self.attrs)

        self.results = self.local_run()
        self.last_run_params = deepcopy(self.run_params)
        self.rerun = False
        self.run_params = {} # Reset run parameters so the next test has to pass
                             # its own run parameters and not use the same ones

    def update_lems_run_params(self):
        from lxml import etree
        from neuroml import nml
        lems_tree = etree.parse(self.lems_file_path)
        trees = {self.lems_file_path:lems_tree}

        # Edit LEMS files.
        nml_file_rel_paths = [x.attrib['file'] for x in \
                              lems_tree.xpath("Include[contains(@file, '.nml')]")]
        nml_file_paths = [os.path.join(os.path.split(self.lems_file_path)[0],x) \
                          for x in nml_file_rel_paths]
        trees.update({x:nml.nml.parsexml_(x) for x in nml_file_paths})

        # Edit NML files.
        for file_path,tree in trees.items():
            for key,value in self.run_params.items():
                if key == 'injected_square_current':
                    pulse_generators = tree.findall('pulseGenerator')
                    for i,pg in enumerate(pulse_generators):
                        for attr in ['delay', 'duration', 'amplitude']:
                            if attr in value:
                                #print('Setting %s to %f' % (attr,value[attr]))
                                pg.attrib[attr] = '%s' % value[attr]

            tree.write(file_path)
