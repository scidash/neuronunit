import os
import numpy as np
import sciunit
from neuronunit.capabilities import ReceivesCurrent,ProducesMembranePotential
from neuronunit.capabilities import LEMS_Runnable
from pyneuroml import pynml
from neo.core import AnalogSignal
from quantities import ms,mV,Hz
from .channel import *


class SimpleModel(sciunit.Model,
                  ReceivesCurrent,
                  ProducesMembranePotential):
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


class LEMSModel(sciunit.Model, LEMS_Runnable):
    """A generic LEMS model"""
    
    def __init__(self, LEMS_file_path, name=None):
        """
        LEMS_file_path: Path to LEMS file (an xml file).
        name: Optional model name.
        """
        self.lems_file_path = LEMS_file_path
        self.run_defaults = pynml.DEFAULTS
        if name is None:
            name = os.path.split()[1].split('.')[0]
        super(LEMSModel,self).__init__(name=name)
    
    def LEMS_run(self, rerun=False, **run_params):
        for key,value in self.run_defaults:
            if key not in run_params:
                run_params[key] = value
        self.results = pynml.run_lems_with_jneuroml(self.lems_file_path, 
                                                    nogui=run_params['nogui'], 
                                                    load_saved_data=True, 
                                                    plot=False, 
                                                    verbose=run_params['v']) 
