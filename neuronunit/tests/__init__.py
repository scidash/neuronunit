import inspect
from quantities.quantity import Quantity
import sciunit
from sciunit import Test,Score,ObservationError
from neuronunit.capabilities import ProducesMembranePotential,ProducesSpikes
from neuronunit.capabilities import ReceivesCurrent,spike_functions
from neuronunit import neuroelectro
from sciunit.comparators import zscore # Converters.  
from sciunit.scores import ErrorScore,BooleanScore,ZScore # Scores.  
from .channel import *

try:
	import numpy as np
except:
	print("NumPy not loaded.")

class SpikeWidthTest(sciunit.Test):
	"""Tests the full widths of spikes at their half-maximum."""
	
	def __init__(self,
			     observation={'mean':None,'std':None},
			     name="Action potential width"):
		"""Takes the mean and standard deviation of observed spike widths"""
		
		Test.__init__(self,observation,name) 
	
	required_capabilities = (ProducesMembranePotential,ProducesSpikes,)

	description = "A test of the widths of action potentials \
				   at half of their maximum height."

	score_type = ZScore

	def validate_observation(self, observation):
		try:
			assert type(observation['mean']) is Quantity
			assert type(observation['std']) is Quantity
		except Exception as e:
			raise ObservationError(("Observation must be of the form "
									"{'mean':float*mV,'std':float*mV}")) 

	def generate_prediction(self, model):
		"""Implementation of sciunit.Test.generate_prediction."""
		# Method implementation guaranteed by ProducesSpikes capability.
		widths = model.get_spike_widths() 
		# Put prediction in a form that compute_score() can use.  
		prediction = {'mean':np.mean(widths),
	  				  'std':np.std(widths)}
		return prediction

	def compute_score(self, observation, prediction):
		"""Implementation of sciunit.Test.score_prediction."""
		print("%s: Observation = %s, Prediction = %s" % (self.name,str(observation),str(prediction)))

		score = zscore(observation,prediction)	
		score = ZScore(score)
		score.related_data['mean_spikewidth'] = prediction['mean']
		return score
		

class InjectedCurrentSpikeWidthTest(SpikeWidthTest):
	"""Tests the full widths of spikes at their half-maximum under current injection."""

	def __init__(self,
				 observation={'mean':None,'std':None},
			     name="Action potential width under current injection",
			     params={'injected_current':{'ampl':0.0}}):
		"""Takes a steady-state current to be injected into a cell."""

		SpikeWidthTest.__init__(self,observation,name)
		self.params = params 
		self.required_capabilities += (ReceivesCurrent,)

	description = "A test of the widths of action potentials \
				   at half of their maximum height when current \
				   is injected into cell."

	def generate_prediction(self, model):
		"""Implementation of sciunit.Test.generate_prediction."""

		model.inject_current(self.params['injected_current']) 	
		spikes = model.get_spikes()
		widths = spike_functions.spikes2widths(spikes)
		#widths *= 1000 # Convert from s to ms.  
		prediction = {'mean':np.mean(widths),
	  				  'std':np.std(widths)}
		return prediction

#def injection_params(amplitude):
#	return {'injected_current':{'ampl':amplitude}}):

#width_test_1 = InjectedCurrentSpikeWidthTest(observation, injection_params(25.0))
#width_test_2 = InjectedCurrentSpikeWidthTest(observation, injection_params(50.0))
#width_test_2 = InjectedCurrentSpikeWidthTest(observation, injection_params(75.0))

#width_suite = sciunit.TestSuite([width_test_1,width_test_2,width_test_3])

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
			assert type(observation['mean']) is Quantity
			assert type(observation['std']) is Quantity
		except Exception as e:
			raise ObservationError(("Observation must be of the form "
									"{'mean':float*mV,'std':float*mV}")) 

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
		score = ZScore(score)
		score.related_data['mean_vm'] = prediction['mean']
		return score

		