import numpy as np
import sciunit
from neuronunit.capabilities import ReceivesCurrent,ProducesMembranePotential
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

