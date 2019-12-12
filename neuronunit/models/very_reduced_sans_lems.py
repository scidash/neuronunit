"""NeuronUnit model class for reduced neuron models."""

#from .static import ExternalModel
import sciunit
'''
class StaticModel(sciunit.Model,
                  cap.ReceivesSquareCurrent,
                  cap.ProducesActionPotentials,
                  cap.ProducesMembranePotential):
'''
import neuronunit.capabilities as cap


import numpy as np
from neo.core import AnalogSignal
import quantities as pq
from sciunit.models.runnable import RunnableModel

import neuronunit.capabilities.spike_functions as sf
class VeryReducedModel(RunnableModel,
                       cap.ReceivesSquareCurrent,
                       cap.ProducesActionPotentials,
                       cap.ProducesMembranePotential):
    """Base class for reduced models, not using LEMS,
    and not requiring file paths this is to wrap pyNN models, Brian models,
    and other self contained models+model descriptions"""

    def __init__(self,name='',backend=None, attrs={}):
        """Instantiate a reduced model.

        LEMS_file_path: Path to LEMS file (an xml file).
        name: Optional model name.
        """
        #sciunit.Model()

        super(VeryReducedModel, self).__init__(name=name,backend=backend, attrs=attrs)
        self.backend = backend
        self.attrs = {}
        self.run_number = 0
        self.tstop = None

    def inject_square_current(self, current):
        #pass
        vm = self._backend.inject_square_current(current)
        return vm

    def get_membrane_potential(self,**run_params):
        vm = self._backend.get_membrane_potential()
        return vm

    def get_APs(self, **run_params):
        vm = self.get_membrane_potential(**run_params)
        waveforms = sf.get_spike_waveforms(vm)
        return waveforms

    def get_spike_train(self, **run_params):
        vm = self.get_membrane_potential(**run_params)
        spike_train = sf.get_spike_train(vm)
        return spike_train
    def set_attrs(self,attrs):
        self.attrs.update(attrs)
