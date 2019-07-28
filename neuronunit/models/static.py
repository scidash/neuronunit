import pickle
from neo.core import AnalogSignal
import sciunit
import sciunit.capabilities as scap
import neuronunit.capabilities as cap
import neuronunit.capabilities.spike_functions as sf


class StaticModel(sciunit.Model,
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


class ExternalModel(sciunit.Model,
                    cap.ProducesMembranePotential,
                    scap.Runnable):
    """A model which produces a frozen membrane potential waveform."""

    def __init__(self):
        """Create an instace of a model that produces a static waveform."""

    def set_membrane_potential(self, vm):
        self.vm = vm

    def set_model_attrs(self, attrs):
        self.attrs = attrs

    def get_membrane_potential(self):
        return self.vm
