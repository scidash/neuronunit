"""SciUnit capability classes for NeuroUnit.
The goal is to enumerate all possible capabilities of a model 
that would be tested using NeuroUnit."""

import numpy as np

import sciunit
from sciunit import Capability

class ProducesMembranePotential(Capability):
	"""Indicates that the model produces a somatic membrane potential."""
	
	def get_membrane_potential(self):
		"""Returns a NeuroTools.signals.AnalogSignal."""
		raise NotImplementedError()

	def get_median_vm(self):
		vm = self.get_membrane_potential()
		return np.median(vm.signal)

class ProducesSpikes(sciunit.Capability):
	"""
	Indicates that the model produces spikes.
	No duration is required for these spikes.
	"""

	def get_spikes(self):
		"""Gets computed spike times from the model.
		
		Arguments: None.
		Returns: a NeuroTools SpikeTrain object.
		"""
		
		raise NotImplementedError()


class ProducesActionPotentials(ProducesSpikes):
	"""Indicates the model produces action potential waveforms.
	Waveforms must have a temporal extent.
	""" 

	def get_action_potentials(self):
		"""Gets action potential waveform chunks from the model.
		
    	Returns
    	-------
    	NeuroTools.signals.AnalogSignalList
        	A list of spike waveforms
		"""

		raise NotImplementedError()

	def get_action_potential_widths(self):
		action_potentials = self.get_action_potentials()
		widths = [utils.ap_width(x) for x in action_potentials]
		return widths

class ReceivesCurrent(Capability):
	"""Indicates that somatic current can be injected into the model."""
	
	def inject_current(self,current):
		"""Injects somatic current into the model.  

	    Parameters
	    ----------
	    current : NeuroTools.signals.AnalogSignal
	        A times series of the current to be injected.  
	    """
		
		raise NotImplementedError()

