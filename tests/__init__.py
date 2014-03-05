import inspect
from sciunit import Test,Score,ObservationError
from neuronunit.capabilities import ProducesMembranePotential,ProducesSpikes
from neuronunit.capabilities import ReceivesCurrent,spike_functions
from neuronunit import neuroelectro
from sciunit.comparators import zscore # Converters.  
from sciunit.scores import ErrorScore,BooleanScore,ZScore # Scores.  

try:
	import numpy as np
except:
	print "NumPy not loaded."

class SpikeWidthTest(Test):
	"""Tests the full widths of spikes at their half-maximum."""
	
	def __init__(self,
			     observation={'mean':None,'std':None},
			     name="Action potential width"):
		"""Takes the mean and standard deviation of reference spike widths"""
		
		Test.__init__(self,observation,name) 
	
	required_capabilities = (ProducesMembranePotential,ProducesSpikes,)

	description = "A test of the widths of action potentials \
				   at half of their maximum height."

	score_type = ZScore

	def validate_observation(self, observation):
		try:
			assert type(observation['mean']) is float
			assert type(observation['std']) is float
		except Exception,e:
			raise ObservationError("Observation must be of the form \
				{'mean':float,'std':float}") 

	def generate_prediction(self, model):
		"""Implementation of sciunit.Test.generate_prediction."""
		spikes = model.get_spikes() # Method implementation guaranteed by 
									# ProducesSpikes capability. 
		widths = spike_functions.spikes2widths(spikes)
		widths *= 1000 # Convert from s to ms.  
		prediction = {'mean':mean(widths),
	  				  'std':std(widths)}
		return prediction

	def compute_score(self, observation, prediction):
		"""Implementation of sciunit.Test.score_prediction."""
		score = zscore(observation,prediction)	
		return ZScore(score)
		

class DynamicSpikeWidthTest(SpikeWidthTest):
	"""Tests the full widths of spikes at their half-maximum under current injection."""

	def __init__(self,
				 observation={'mean':None,'std':None},
			     name="Action potential width under current injection",
			     params={'injected_current':{'ampl':0.0}}):
		"""Takes a steady-state current to be injected into a cell."""

		SpikeWidthTest.__init__(self,observation,name) 
		self.required_capabilities += (ReceivesCurrent,)

	description = "A test of the widths of action potentials \
				   at half of their maximum height when current \
				   is injected into cell."

	def generate_prediction(self, model):
		"""Implementation of sciunit.Test.generate_prediction."""

		model.inject_current(self.params['injected_current']) 	
		spikes = model.get_spikes()
		widths = spike_functions.spikes2widths(spikes)
		widths *= 1000 # Convert from s to ms.  
		prediction = {'mean':mean(widths),
	  				  'std':std(widths)}
		return prediction


class RestingPotentialTest(Test):
	"""Tests the resting potential under zero current injection."""
	
	def __init__(self,
			     observation={'mean':None,'std':None},
			     name="Resting potential test"):
		"""Takes the mean and standard deviation of reference membrane potentials."""
		
		Test.__init__(self,observation,name)
		self.required_capabilities += (ProducesMembranePotential,
							  		   ReceivesCurrent,)

	description = "A test of the resting potential of a cell\
				   where injected current is set to zero."

	score_type = ZScore

	def validate_observation(self, observation):
		try:
			assert type(observation['mean']) is float
			assert type(observation['std']) is float
		except Exception,e:
			raise ObservationError("Observation must be of the form \
				{'mean':float,'std':float}") 

	def generate_prediction(self, model):
		"""Implementation of sciunit.Test.generate_prediction."""
		current_ampl = 0.0
		model.inject_current({'ampl':0.0}) 
		vm = model.get_median_vm() # Use median instead of mean for robustness.  
		prediction = {'mean':vm}
		return prediction

	def compute_score(self, observation, prediction):
		"""Implementation of sciunit.Test.score_prediction."""
		score = zscore(observation,prediction)	
		return ZScore(score)
	