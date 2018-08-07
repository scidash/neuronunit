"""Model classes for NeuronUnit"""

import pickle

from neo.core import AnalogSignal

import sciunit
import neuronunit.capabilities as cap


class StaticModel(sciunit.Model,
                  cap.ProducesMembranePotential):
    """A model which produces a frozen membrane potential waveform"""

    def __init__(self, vm):
        """vm is either a neo.core.AnalogSignal or a path to a
        pickled neo.core.AnalogSignal"""

        if isinstance(vm, str):
            with open(vm, 'r') as f:
                vm = pickle.load(f)
        if not isinstance(vm, AnalogSignal):
            raise TypeError('vm must be a neo.core.AnalogSignal')

        self.vm = vm

    def get_membrane_potential(self, **kwargs):
        return self.vm
