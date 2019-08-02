"""NeuronUnit model class for reduced neuron models."""

import numpy as np
from neo.core import AnalogSignal
import quantities as pq

import neuronunit.capabilities as cap

import neuronunit.capabilities.spike_functions as sf
from .static import ExternalModel

class VeryReducedModel(ExternalModel,
                       cap.ReceivesCurrent,
                       cap.ProducesActionPotentials,
                       ):
    """Base class for reduced models, not using LEMS,
    and not requiring file paths this is to wrap pyNN models, Brian models,
    and other self contained models+model descriptions"""

    def __init__(self, backend=None, attrs=None):
        """Instantiate a reduced model.

        LEMS_file_path: Path to LEMS file (an xml file).
        name: Optional model name.
        """
        super(VeryReducedModel, self).__init__()
        self.backend = backend
        self.attrs = attrs
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
 
    '''
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
        vm = self.get_membrane_potential(**run_params)
        waveforms = sf.get_spike_waveforms(vm)
        return waveforms

    def get_spike_train(self, **run_params):
        vm = self.get_membrane_potential(**run_params)
        spike_train = sf.get_spike_train(vm)
        return spike_train

    def inject_square_current(self, current):
        self.set_run_params(injected_square_current=current)
        self._backend.inject_square_current(current)
    '''
