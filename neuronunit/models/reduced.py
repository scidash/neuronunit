"""NeuronUnit model class for reduced neuron models."""

import numpy as np
from neo.core import AnalogSignal
import quantities as pq

import neuronunit.capabilities as cap
from .lems import LEMSModel
from .static import ExternalModel
import neuronunit.capabilities.spike_functions as sf
from copy import deepcopy

class ReducedModel(LEMSModel,
                   cap.ReceivesSquareCurrent,
                   cap.ProducesActionPotentials,
                   ):
    """Base class for reduced models, using LEMS"""

    def __init__(self, LEMS_file_path, name=None, backend=None, attrs=None):
        """Instantiate a reduced model.

        LEMS_file_path: Path to LEMS file (an xml file).
        name: Optional model name.
        """

        super(ReducedModel, self).__init__(LEMS_file_path, name=name,
                                           backend=backend, attrs=attrs)
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

    def inject_square_current(self, current):
        assert isinstance(current, dict)
        assert all(x in current for x in
                   ['amplitude', 'delay', 'duration'])
        self.set_run_params(injected_square_current=current)
        self._backend.inject_square_current(current)


class VeryReducedModel(ExternalModel,
                   cap.ReceivesCurrent,
                   cap.ProducesActionPotentials,
                   ):
    """Base class for reduced models, using LEMS"""

    def __init__(self, name=None, backend=None, attrs=None):
        """Instantiate a reduced model.
        LEMS_file_path: Path to LEMS file (an xml file).
        name: Optional model name.
        """
        super(VeryReducedModel,self).__init__(name=name, backend=backend, attrs=attrs)
        self.run_number = 0
        self.tstop = None

    def run(self, rerun=None, **run_params):
        if rerun is None:
            rerun = self.rerun
        self.set_run_params(**run_params)
        for key,value in self.run_defaults.items():
            if key not in self.run_params:
                self.set_run_params(**{key:value})
        #if (not rerun) and hasattr(self,'last_run_params') and \
        #   self.run_params == self.last_run_params:
        #    print("Same run_params; skipping...")
        #    return

        self.results = self._backend.local_run()
        self.last_run_params = deepcopy(self.run_params)
        #self.rerun = False
        # Reset run parameters so the next test has to pass its own
        # run parameters and not use the same ones
        self.run_params = {}

    def set_run_params(self, **params):
        self._backend.set_run_params(**params)

    # Methods to override using inheritance.
    def get_membrane_potential(self, **run_params):
        pass
    def get_APs(self, **run_params):
        pass
    def get_spike_train(self, **run_params):
        pass
    def inject_square_current(self, current):
        pass
