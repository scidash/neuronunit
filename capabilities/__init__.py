"""SciUnit capability classes for NeuroUnit.
The goal is to enumerate all possible capabilities of a model 
that would be tested using NeuroUnit."""

import sciunit
from sciunit import Capability

class ProducesMembranePotential(Capability):
	"""The model produces a somatic membrane potential."""
	
	def get_membrane_potential(self):
		raise NotImplementedError()

	def get_median_vm(self):
		raise NotImplementedError()

	def get_initial_vm(self):
		raise NotImplementedError()


class ProducesSpikes(Capability):
	"""The model produces action potentials.""" 
	
	def get_spikes(self):
		raise NotImplementedError()
	def get_spike_train(self):
		raise NotImplementedError()

class ReceivesCurrent(Capability):
	"""Somatic current can be injected into the model cell."""
	
	def inject_current(self,current):
		raise NotImplementedError()

