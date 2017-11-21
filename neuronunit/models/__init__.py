"""Model classes for NeuronUnit"""

import os
from copy import deepcopy
import tempfile
import inspect
import shutil
import random

from lxml import etree

import sciunit
#from sciunit.utils import dict_hash
import neuronunit.capabilities as cap
from pyneuroml import pynml
from . import backends


class LEMSModel(sciunit.Model,
                cap.Runnable,
                ):
    """A generic LEMS model"""

    def __init__(self, LEMS_file_path, name=None, 
                    backend='jNeuroML', attrs=None):

        #for base in cls.__bases__:
        #    sciunit.Model.__init__()
        if name is None:
            name = os.path.split(LEMS_file_path)[1].split('.')[0]
        self.name = name
        #sciunit.Modelsuper(LEMSModel,self).__init__(name=name)
        self.attrs = attrs if attrs else {}
        self.orig_lems_file_path = os.path.abspath(LEMS_file_path)
        assert os.path.isfile(self.orig_lems_file_path),\
            "'%s' is not a file" % self.orig_lems_file_path
        self.run_defaults = pynml.DEFAULTS
        self.run_defaults['nogui'] = True
        self.run_params = {}
        self.last_run_params = None
        self.skip_run = False
        self.rerun = True # Needs to be rerun since it hasn't been run yet!
        self.unpicklable = []
        self.set_backend(backend)

    def get_backend(self):
        return self._backend

    def set_backend(self, backend):
        if isinstance(backend,str):
            name = backend
            args = []
            kwargs = {}
        elif isinstance(backend,(tuple,list)):
            name = ''
            args = []
            kwargs = {}
            for i in range(len(backend)):
                if i==0:
                    name = backend[i]
                else:
                    if isinstance(backend[i],dict):
                        kwargs.update(backend[i])
                    else:
                        args += backend[i]
        else:
            raise TypeError("Backend must be string, tuple, or list")
        options = {x.replace('Backend',''):cls for x, cls \
                   in backends.__dict__.items() \
                   if inspect.isclass(cls) and \
                   issubclass(cls, backends.Backend)}
        if name in options:
            self.backend = name
            self._backend = options[name]()
        elif name is None:
            # The base class should not be called.
            raise Exception(("A backend (e.g. 'jNeuroML' or 'NEURON') "
                             "must be selected"))
        else:
            raise Exception("Backend %s not found in backends.py" \
                            % name)
        self._backend.model = self
        self._backend.init_backend(*args, **kwargs)

    def get_nml_paths(self, lems_tree=None, absolute=True, original=False):
        if not lems_tree:
            lems_tree = etree.parse(self.lems_file_path)
        nml_paths = [x.attrib['file'] for x in \
                     lems_tree.xpath("Include[contains(@file, '.nml')]")]
        if absolute: # Turn into absolute paths
            lems_file_path = self.orig_lems_file_path if original \
                                                      else self.lems_file_path
            nml_paths = [os.path.join(os.path.dirname(lems_file_path),x) \
                         for x in nml_paths]
        return nml_paths

    def create_lems_file(self, name):
        if not hasattr(self,'temp_dir'):
            self.temp_dir = tempfile.gettempdir()
        rand = random.randint(0,1e15)
        self.lems_file_path  = os.path.join(self.temp_dir, '%s_%d.xml' % (name,rand))
        shutil.copy2(self.orig_lems_file_path,self.lems_file_path)
        nml_paths = self.get_nml_paths(original=True)
        for orig_nml_path in nml_paths:
            new_nml_path = os.path.join(self.temp_dir,
                                        os.path.basename(orig_nml_path))
            shutil.copy2(orig_nml_path,new_nml_path)
        if self.attrs:
            self.set_lems_attrs(self.attrs)

    def set_attrs(self,attrs):
        self._backend.set_attrs(**attrs)

    def inject_square_current(self,current):
        self._backend.inject_square_current(current)
    #    
    #def inject_square_current(self,current):
    #    self._backend.inject_square_current(current)
    #
    #def local_run(self):
        
    def set_lems_attrs(self, attrs):
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
        self.set_run_params(**run_params)
        for key,value in self.run_defaults.items():
            if key not in self.run_params:
                self.set_run_params(**{key:value})
        #if (not rerun) and hasattr(self,'last_run_params') and \
        #   self.run_params == self.last_run_params:
        #    print("Same run_params; skipping...")
        #    return

        self.results = self._backend.local_run()
        self.last_run_params = deepcopy(self.run_params)
        #self.rerun = False
        # Reset run parameters so the next test has to pass its own
        # run parameters and not use the same ones
        self.run_params = {}
        
    def set_run_params(self, **params):
        self._backend.set_run_params(**params)

    def set_lems_run_params(self, verbose=False):
        from lxml import etree
        from neuroml import nml
        lems_tree = etree.parse(self.lems_file_path)
        trees = {self.lems_file_path:lems_tree}

        # Edit LEMS files.
        nml_paths = self.get_nml_paths(lems_tree=lems_tree)
        trees.update({x:nml.nml.parsexml_(x) for x in nml_paths})

        # Edit NML files.
        for file_path,tree in trees.items():
            for key,value in self.run_params.items():
                if key == 'injected_square_current':
                    pulse_generators = tree.findall('pulseGenerator')
                    for pg in pulse_generators:
                        for attr in ['delay', 'duration', 'amplitude']:
                            if attr in value:
                                if verbose:
                                    print('Setting %s to %f' % (attr,value[attr]))
                                pg.attrib[attr] = '%s' % value[attr]

            tree.write(file_path)

    def inject_square_current(self, current):
        self._backend.inject_square_current(current)

    @property
    def state(self):
        return self._state(keys=['name','url', 'attrs','run_params'])