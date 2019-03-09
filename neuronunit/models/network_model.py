"""NeuronUnit model class for reduced neuron models"""

import numpy as np
from neo.core import AnalogSignal
import quantities as pq

import neuronunit.capabilities as cap
import neuronunit.models as mod
import neuronunit.capabilities.spike_functions as sf
from neuronunit.models import backends
from generic_network import net_sim_runner, get_dummy_synapses


class NetworkModel(cap.ReceivesCurrent,
                   cap.ProducesMultiMembranePotentials,
                   cap.ProducesSpikeRasters,
                   ):

    """Base class for network models
    todo replace receives current with receives patterned input."""

    def __init__(self, name=None, backend=pyNN, synapses=None):
        """Instantiate a network model.
        name: Optional model name.
        """
        self.run_number = 0
        self.backend = backend
        self.tstop = None
        self.data = None
        self.vms = None
        self.binary_trains = None
        self.t_spike_axis = None
        self.synapses = get_dummy_synapses()
        try:
            self.sim = generic_network.sim
        except:
            pass
    def get_membrane_potentials(self):
        return self.vms

    def getSpikeRasters(self, **run_params):
        return self.binary_train

    def inject_noise_current(self, stim_current, syn_weights):
        import pyNN.neuron as sim
        noisee = sim.NoisyCurrentSource(mean=0.74/1000.0, stdev=4.00/1000.0, start=0.0, stop=2000.0, dt=1.0)
        noisei = sim.NoisyCurrentSource(mean=1.440/1000.0, stdev=4.00/1000.0, start=0.0, stop=2000.0, dt=1.0)
        stim_noise_currents = [noisee,noisei]
        self.data,self.vms,self.binary_trains,self.t_spike_axis = net_sim_runner(syn_weights,sim,self.synapses,stim_noise_currents)
        return (self.vms,self.binary_train,self.data)
