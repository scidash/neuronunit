"""SciUnit capability classes for NeuronUnit.
The goal is to enumerate all possible capabilities of a model 
that would be tested using NeuronUnit.
These capabilities exchange 'neo' objects."""

import numpy as np

import sciunit
from sciunit import Capability
from .channel import *

class ProducesMembranePotential(Capability):
	"""Indicates that the model produces a somatic membrane potential."""
	
	def get_membrane_potential(self):
		"""Must return a neo.core.AnalogSignal."""
		raise NotImplementedError()

	def get_median_vm(self):
		vm = self.get_membrane_potential()
		print("Vm is", vm)
		print("Vm base is", vm.base)
		print("Vm mean is", np.mean(vm))
		print("Vm base mean is ", np.mean(vm.base))
		print("Vm base median is ", np.median(vm.base))
		print("Vm median is ", np.median(vm))
		return np.median(vm)

class ProducesSpikes(sciunit.Capability):
	"""
	Indicates that the model produces spikes.
	No duration is required for these spikes.
	"""

	def get_spike_train(self):
		"""Gets computed spike times from the model.
		
		Arguments: None.
		Returns: a neo.core.SpikeTrain object.
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
    	Must return a neo.core.AnalogSignalArray.
        Each neo.core.AnalogSignal in the array should be a spike waveform.
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
	    current : neo.core.AnalogSignal
	    This is a time series of the current to be injected.  
	    """
		
		raise NotImplementedError()

class LEMS_Runnable(sciunit.Capability):
    """Capability for models that can be run by executing LEMS files."""
    def LEMS_run(self,**run_params):
        return NotImplementedError("%s not implemented" % inspect.stack()[0][3])
 

