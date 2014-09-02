import numpy as np
import sciunit
from neuronunit.capabilities import ReceivesCurrent,ProducesMembranePotential
from NeuroTools.signals import AnalogSignal

class SimpleModel(sciunit.Model,
                  ReceivesCurrent,
                  ProducesMembranePotential):
    def __init__(self, v_rest, name=None):
        self.v_rest = v_rest
        sciunit.Model.__init__(self, name=name)

    def get_membrane_potential(self):
        array = np.ones(10000) * self.v_rest
        dt = 0.001
        return AnalogSignal(array,dt)

    def inject_current(self,current):
        pass # Does not actually inject any current.  

