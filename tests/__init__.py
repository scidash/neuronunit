import inspect
from sciunit.capabilities import Runnable
from neurounit.capabilities import ProducesMembranePotential,ProducesSpikes
from neurounit.capabilities import ReceivesCurrent,spike_functions
from neurounit import neuroelectro
from sciunit.tests import StandardTest
from sciunit.comparators import ZComparator # Comparators.  
from sciunit.comparators import ZScoreToBooleanScore # Converters.  
from sciunit.scores import BooleanScore # Scores.  
try:
	import numpy as np
except:
	print "NumPy not loaded."

class SpikeWidthTest(ZTest):
	"""Tests the full widths of spikes at their half-maximum."""
	
	def __init__(self,
			     reference_data={'mean':None,'std':None},
			     model_args={}):
		"""Takes the mean and standard deviation of reference spike widths"""
		print "Instantiating a spike width test."
		ZTest.__init__(self,reference_data,model_args) 
		"""Register reference data and model arguments."""  
		self.required_capabilities += (ProducesMembranePotential,
							  		   ProducesSpikes,)

	desc = "spike widths"

	def get_output_data(self,model):
		"""Extracts data from the model and returns it."""
		spikes = model.get_spikes() # Method implementation guaranteed by 
									# ProducesSpikes capability. 
		widths = spike_functions.spikes2widths(spikes)
		widths *= 1000 # Convert from s to ms.  
		model_output_data = {'mean':mean(widths),
	  					  	 'std':std(widths)}
		return model_output_data

	def get_model_stats(self,output_data):
		"""Puts model stats in a form that the Comparator will understand."""
		return {'value':output_data['mean']}
		
class SpikeWidthTestDynamic(SpikeWidthTest):
	"""Tests the full widths of spikes at their half-maximum under current injection."""

	def __init__(self,
				 reference_data=SpikeWidthTest.__init__.im_func.func_defaults[0],  
			     model_args={'injected_current':0.0}):
		"""Takes a steady-state current to be injected into a cell."""

		SpikeWidthTest.__init__(self,reference_data,model_args) 
		self.required_capabilities += (ReceivesCurrent,)

	desc = "spike widths under current injection"


class RestingPotentialTest(ZTest):
	"""Tests the resting potential under zero current injection."""
	
	def __init__(self,
			     reference_data={'mean':None,'std':None},
			     model_args={}):
		"""Takes the mean and standard deviation of reference spike widths"""
		
		ZTest.__init__(self,reference_data,model_args) 
		"""Register reference data and model arguments."""  
		
		self.required_capabilities += (ProducesMembranePotential,
							  		   ReceivesCurrent,)

	desc = "resting potential (zero current injection)"

	def _judge(self,model,**kwargs):
		current_ampl = 0.0
		model.set_current_ampl(current_ampl) 
		print "Setting current amplitude to %f" % current_ampl
		# Setting injected current to zero.  
		score = super(RestingPotentialTest,self)._judge(model,**kwargs) # Run the model.  
		score.related_data.update({'vm':model.get_membrane_potential()})
		return score

	def get_model_data(self,model):
		"""Extracts data from the model and returns it."""
		
		resting_potential = model.get_median_vm() # Resting potential in mV.
		model_data = {'value':resting_potential,}
		return model_data
		
		