import os
from copy import deepcopy
import tempfile
import shutil

import numpy as np
import sciunit
import neuronunit.capabilities as cap
from pyneuroml import pynml
from neo.core import AnalogSignal
from quantities import ms,mV,Hz
from .channel import *


class SimpleModel(sciunit.Model,
                  cap.ReceivesCurrent,
                  cap.ProducesMembranePotential):
    def __init__(self, v_rest, name=None):
        self.v_rest = v_rest
        sciunit.Model.__init__(self, name=name)

    def get_membrane_potential(self):
        array = np.ones(10000) * self.v_rest
        dt = 1*ms # Time per sample in milliseconds.  
        vm = AnalogSignal(array,units=mV,sampling_rate=1.0/dt)
        return vm

    def inject_current(self,current):
        pass # Does not actually inject any current.  


class LEMSModel(sciunit.Model, cap.Runnable):
    """A generic LEMS model"""
    
    def __init__(self, LEMS_file_path, name=None, attrs={}):
        """
        LEMS_file_path: Path to LEMS file (an xml file).
        name: Optional model name.
        """
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
        super(LEMSModel,self).__init__(name=name)

    def create_lems_file(self, name, attrs):
        if not hasattr(self,'temp_dir'):
            self.temp_dir = tempfile.gettempdir()
        self.lems_file_path  = os.path.join(self.temp_dir, '%s.xml' % name)
        shutil.copy2(self.orig_lems_file_path, self.lems_file_path)
        if attrs:
            self.set_attrs(attrs)    

    def set_attrs(self, attrs):
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
        self.update_run_params()
        
        f = pynml.run_lems_with_jneuroml_neuron
        #print(self.lems_file_path)
        self.results = f(self.lems_file_path, skip_run=self.skip_run,
                         nogui=self.run_params['nogui'], 
                         load_saved_data=True, plot=False, 
                         verbose=self.run_params['v'])
        self.last_run_params = deepcopy(self.run_params)
        self.rerun = False
        self.run_params = {} # Reset run parameters so the next test has to pass
                             # its own run parameters and not use the same ones

    def update_run_params(self):
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
            
        