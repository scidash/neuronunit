from neo.core import AnalogSignal
import neuronunit.capabilities as cap
import neuronunit.capabilities.spike_functions as sf
import numpy as np
import pickle
import quantities as pq
import sciunit
import sciunit.capabilities as scap
from sciunit.models import RunnableModel


class StaticModel(RunnableModel,
                  cap.ReceivesSquareCurrent,
                  cap.ProducesActionPotentials,
                  cap.ProducesMembranePotential):
    """A model which produces a frozen membrane potential waveform."""

    def __init__(self, vm):
        """Create an instace of a model that produces a static waveform.

        :param vm: either a neo.core.AnalogSignal or a path to a
                   pickled neo.core.AnalogSignal
        """
        if isinstance(vm, str):
            with open(vm, 'r') as f:
                vm = pickle.load(f)

        if not isinstance(vm, AnalogSignal):
            raise TypeError('vm must be a neo.core.AnalogSignal')

        self.vm = vm
        self.backend = 'static_model'        
    def run(self, **kwargs):
        pass

    def get_membrane_potential(self, **kwargs):
        """Return the Vm passed into the class constructor."""
        return self.vm

    def get_APs(self, **run_params):
        """Return the APs, if any, contained in the static waveform."""
        vm = self.get_membrane_potential(**run_params)
        waveforms = sf.get_spike_waveforms(vm)
        return waveforms

    def inject_square_current(self, current):
        """Static model always returns the same waveform.
        This method is for compatibility only."""
        pass


class ExternalModel(sciunit.models.RunnableModel,
                    cap.ProducesMembranePotential,
                    scap.Runnable):
    """A model which produces a frozen membrane potential waveform."""

    def __init__(self, *args, **kwargs):
        """Create an instace of a model that produces a static waveform."""
        super(ExternalModel, self).__init__(*args, **kwargs)

    def set_membrane_potential(self, vm):
        self.vm = vm

    def set_model_attrs(self, attrs):
        self.attrs = attrs

    def get_membrane_potential(self):
        return self.vm
    def get_APs(self, **run_params):
        """Return the APs, if any, contained in the static waveform."""
        vm = self.get_membrane_potential(**run_params)
        waveforms = sf.get_spike_waveforms(vm)
        return waveforms
    
class RandomVmModel(RunnableModel, cap.ProducesMembranePotential, cap.ReceivesCurrent):
    def get_membrane_potential(self):
        # Random membrane potential signal
        vm = (np.random.randn(10000)-60)*pq.mV
        vm = AnalogSignal(vm, sampling_period=0.1*pq.ms)
        return vm
    
    def inject_square_current(self, current):
        pass
