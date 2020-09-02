"""NeuronUnit model class for reduced neuron models."""

import numpy as np
from neo.core import AnalogSignal
import quantities as pq

import neuronunit.capabilities as cap

from .static import ExternalModel
import neuronunit.capabilities.spike_functions as sf
from neuronunit.optimisation.model_parameters import path_params

from .lems import LEMSModel
class ReducedModel(LEMSModel,
                   cap.ReceivesSquareCurrent,
                   cap.ProducesActionPotentials,
                   ):
    """Base class for reduced models, using LEMS"""

    def __init__(self, LEMS_file_path, name=None, backend=None, attrs={}):
        """Instantiate a reduced model.

        LEMS_file_path: Path to LEMS file (an xml file).
        name: Optional model name.
        """

        super(ReducedModel, self).__init__(LEMS_file_path, name=name,
                                           backend=backend, attrs=attrs)
        self.run_number = 0
        self.tstop = None
        #self.attrs = attrs

    def get_membrane_potential(self, **run_params):
        self.run(**run_params)
        for rkey in self.results.keys():
            if 'v' in rkey or 'vm' in rkey:
                v = np.array(self.results[rkey])
        t = np.array(self.results['t'])
        dt = (t[1]-t[0])*pq.s  # Time per sample in seconds.
        vm = AnalogSignal(v, units=pq.V, sampling_rate=1.0/dt)
        return vm

    def get_APs(self, **run_params):
        try:
            vm = self._backend.get_membrane_potential(**run_params)
        except:
            vm = self.get_membrane_potential(**run_params)
        if str('ADEXP') in self._backend.name:

            self._backend.threshold = np.max(vm)-np.max(vm)/250.0
            waveforms = sf.get_spike_waveforms(vm,self._backend.threshold)
        else:
            waveforms = sf.get_spike_waveforms(vm)
        return waveforms

    def get_spike_train(self, **run_params):
        vm = self.get_membrane_potential(**run_params)
        #spike_train = sf.get_spike_train(vm)
        if str('ADEXP') in self._backend.name:
        #if hasattr(self._backend,'name'):
            self._backend.threshold = np.max(vm)-np.max(vm)/250.0
            spike_train = sf.get_spike_train(vm,self._backend.threshold)
        else:
            spike_train = sf.get_spike_train(vm)

        return spike_train
    '''
    def set_attrs(self,attrs):
        self._backend.set_attrs(**attrs)


    def get_backend(self):
        return self._backend
    '''
