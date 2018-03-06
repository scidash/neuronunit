
"""NeuronUnit model class for reduced neuron models"""

import numpy as np
from neo.core import AnalogSignal
import quantities as pq

import neuronunit.capabilities as cap
import neuronunit.models as mod
import neuronunit.capabilities.spike_functions as sf
import neuronunit.capabilities as cap

#from neuronunit.models import backends

import sciunit
import pickle
from neo.core import AnalogSignal

class StaticModel(sciunit.Model, cap.ProducesActionPotentials):
    # cap.ProducesMembranePotential,
    def __init__(self, vm = None, st = None, name = None):

        super(sciunit.Model,self).__init__()
        self.vm = vm
        self.st = st
        if name is not None:
            self.name = name



    def get_spike_train(self, **kwargs):
        """Returns a neo.core.SpikeTrain object"""
        vm = self.get_membrane_potential()
        assert self.st == cap.spike_functions.get_spike_train(vm)
        # A neo.core.AnalogSignal object
        return cap.spike_functions.get_spike_train(vm)

    def get_membrane_potential(self):
        return self.vm

    def get_spike_train(self):
        return self.st

    def get_spike_count(self):
        return len(self.st)

    def get_APs(self, rerun=False, threshold=0*pq.mV **run_params):
        vm = self.get_membrane_potential()
        # waveforms = sf.get_spike_waveforms(vm, threshold=threshold)
        waveforms = sf.get_spike_waveforms(vm, threshold=-52.0 * pq.mV)

        return waveforms
