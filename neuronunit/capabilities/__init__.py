"""NeuronUnit abstract Capabilities.

The goal is to enumerate all possible capabilities of a model that would be
tested using NeuronUnit. These capabilities exchange 'neo' objects.
"""

import numpy as np
import quantities as pq
import sciunit
import matplotlib.pyplot as plt
from .spike_functions import spikes2amplitudes, spikes2widths,\
                             spikes2thresholds


class ProducesMembranePotential(sciunit.Capability):
    """Indicates that the model produces a somatic membrane potential."""

    def get_membrane_potential(self, **kwargs):
        """Must return a neo.core.AnalogSignal."""
        raise NotImplementedError()

    def get_mean_vm(self, **kwargs):
        """Get the mean membrane potential."""
        vm = self.get_membrane_potential(**kwargs)
        return np.mean(vm.base)

    def get_median_vm(self, **kwargs):
        """Get the median membrane potential."""
        vm = self.get_membrane_potential(**kwargs)
        return np.median(vm.base)

    def get_std_vm(self, **kwargs):
        """Get the standard deviation of the membrane potential."""
        vm = self.get_membrane_potential(**kwargs)
        return np.std(vm.base)

    def get_iqr_vm(self, **kwargs):
        """Get the inter-quartile range of the membrane potential."""
        vm = self.get_membrane_potential(**kwargs)
        return (np.percentile(vm, 75) - np.percentile(vm, 25))*vm.units

    def get_initial_vm(self, **kwargs):
        """Return a quantity corresponding to the starting membrane potential.

        This will in some cases be the resting potential.
        """
        vm = self.get_membrane_potential(**kwargs)
        return vm[0]  # A neo.core.AnalogSignal object

    def plot_membrane_potential(self, ax=None, ylim=(None, None), **kwargs):
        """Plot the membrane potential."""
        vm = self.get_membrane_potential(**kwargs)
        if ax is None:
            ax = plt.gca()
        vm = vm.rescale('mV')
        ax.plot(vm.times, vm)
        y_min = float(vm.min()-5.0*pq.mV) if ylim[0] is None else ylim[0]
        y_max = float(vm.max()+5.0*pq.mV) if ylim[1] is None else ylim[1]
        ax.set_xlim(vm.times.min(), vm.times.max())
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Vm (mV)')


class ProducesSpikes(sciunit.Capability):
    """Indicate that the model produces spikes.

    No duration is required for these spikes.
    """

    def get_spike_train(self):
        """Get computed spike times from the model.

        Arguments: None.
        Returns: a neo.core.SpikeTrain object.
        """
        raise NotImplementedError()

    def get_spike_count(self):
        """Get the number of spikes."""
        spike_train = self.get_spike_train()
        return len(spike_train)


class ProducesActionPotentials(ProducesSpikes, 
                               ProducesMembranePotential):
    """Indicate the model produces action potential waveforms.

    Waveforms must have a temporal extent.
    """

    def get_APs(self):
        """Get action potential waveform chunks from the model.

        Returns
        -------
        Must return a neo.core.AnalogSignal.
        Each column of the AnalogSignal should be a spike waveform.
        """
        raise NotImplementedError()

    def get_AP_widths(self):
        """Get widths of action potentials."""
        action_potentials = self.get_APs()
        widths = spikes2widths(action_potentials)
        return widths

    def get_AP_amplitudes(self):
        """Get amplitudes of action potentials."""
        action_potentials = self.get_APs()
        amplitudes = spikes2amplitudes(action_potentials)
        return amplitudes

    def get_AP_thresholds(self):
        """Get thresholds of action potentials."""
        action_potentials = self.get_APs()
        thresholds = spikes2thresholds(action_potentials)
        return thresholds


class ReceivesSquareCurrent(sciunit.Capability):
    """Indicate that somatic current can be injected into the model as
    a square pulse.
    """

    def inject_square_current(self, current):
        """Injects somatic current into the model.

        Parameters
        ----------
        current : a dictionary like:
                        {'amplitude':-10.0*pq.pA,
                         'delay':100*pq.ms,
                         'duration':500*pq.ms}}
                  where 'pq' is the quantities package
        This describes the current to be injected.
        """
        raise NotImplementedError()


class ReceivesCurrent(ReceivesSquareCurrent):
    """Indicate that somatic current can be injected into the model as
    either an arbitrary waveform or as a square pulse.
    """

    def inject_current(self, current):
        """Inject somatic current into the model.

        Parameters
        ----------
        current : neo.core.AnalogSignal
        This is a time series of the current to be injected.
        """
        raise NotImplementedError()


class Runnable(sciunit.Capability):
    """Capability for models that can be run."""
    
    def run(self, **run_params):
        return NotImplementedError("%s not implemented" % inspect.stack()[0][3])
