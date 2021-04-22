"""NeuronUnit model class for reduced neuron models."""

import numpy as np
from neo.core import AnalogSignal
import quantities as pq

import neuronunit.capabilities as cap
#import neuronunit.models as mod
import neuronunit.capabilities.spike_functions as sf

from sciunit.models import RunnableModel

class VeryReducedModel(RunnableModel,
                       cap.ReceivesCurrent,
                       cap.ProducesActionPotentials,
                       ):
    """Base class for reduced models, not using LEMS,
    and not requiring file paths this is to wrap pyNN models, Brian models,
    and other self contained models+model descriptions"""

    def __init__(self, name=None, backend=None, attrs=None):
        """Instantiate a reduced model.

        """
        super(VeryReducedModel,self).__init__(name)
        self.backend = backend
        self.attrs = attrs
        self.run_number = 0
        self.tstop = None

    def get_membrane_potential(self, **run_params):
        self.run(**run_params)
        v = None
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

    #def inject_square_current(self, current):
    #    self.set_run_params(injected_square_current=current)
    #    self._backend.inject_square_current(current)
