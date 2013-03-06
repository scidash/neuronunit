"""Sciunit capability classes for neuroscience"""

from sciunit import Capability

class ProducesMembranePotential(Capability):
	"""An array of somatic membrane potential samples"""
	def get_membrane_potential(self):
		raise NotImplementedError()

class ProducesSpikes(Capability):
	"""A 2-D array: spike_waveform x num_spikes""" 
	def get_spikes(self):
		raise NotImplementedError()

class ReceivesCurrent(Capability):
	"""An array of somatic injected current samples"""
	def set_current(self,current):
		raise NotImplementedError()

