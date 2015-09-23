import inspect
from types import MethodType

from quantities.quantity import Quantity
import numpy as np
import matplotlib.pyplot as plt

import sciunit
from sciunit import Test,Score,ObservationError
from sciunit.comparators import compute_zscore # Converters.  
from sciunit.scores import ErrorScore,InsufficientDataScore,\
						   BooleanScore,ZScore # Scores.  

from neuronunit.capabilities import ProducesMembranePotential,ProducesSpikes
from neuronunit.capabilities import ReceivesCurrent,spike_functions
from neuronunit import neuroelectro
from .channel import *

class VmTest(sciunit.Test):
	"""Base class for tests involving the membrane potential of a model."""

	def bind_score(self,score,model,observation,prediction):
		score.related_data['vm'] = model.get_membrane_potential()
		score.related_data['model_name'] = '%s_%s' % (model.name,self.name)
		
		def plot_vm(self,ax=None):
			"""A plot method the score can use for convenience."""
			if ax is None:
				ax = plt.gca()
			vm = score.related_data['vm']
			ax.plot(vm.times,vm)
			ax.set_ylim(-80,20)
			ax.set_xlabel('Time (s)')
			ax.set_ylabel('Vm (mV)')
		score.plot_vm = MethodType(plot_vm, score) # Bind to the score.  
		return score
		

class SpikeWidthTest(VmTest):
	"""Tests the full widths of spikes at their half-maximum."""
	
	def __init__(self,
				 observation={'mean':None,'std':None},
				 name="Action potential width"):
		"""Takes the mean and standard deviation of observed spike widths"""
		
		Test.__init__(self,observation,name) 
	
	required_capabilities = (ProducesMembranePotential,ProducesSpikes,)

	description = ("A test of the widths of action potentials "
				   "at half of their maximum height.")

	score_type = ZScore

	def validate_observation(self, observation):
		try:
			assert type(observation['mean']) is Quantity
			assert type(observation['std']) is Quantity
		except Exception as e:
			raise ObservationError(("Observation must be of the form "
									"{'mean':float*ms,'std':float*ms}")) 

	def generate_prediction(self, model):
		"""Implementation of sciunit.Test.generate_prediction."""
		# Method implementation guaranteed by ProducesSpikes capability.
		model.rerun = True
		current_ampl = self.params['injected_current']['ampl']
		model.inject_current({'ampl':current_ampl}) 
		widths = model.get_spike_widths() 
		# Put prediction in a form that compute_score() can use.  
		prediction = {'mean':np.mean(widths) if len(widths) else None,
					  'std':np.std(widths) if len(widths) else None,
					  'n':len(widths)}
		return prediction

	def compute_score(self, observation, prediction):
		"""Implementation of sciunit.Test.score_prediction."""
		#print("%s: Observation = %s, Prediction = %s" % \
		#	 (self.name,str(observation),str(prediction)))
		if prediction['n'] == 0:
			score = InsufficientDataScore(None)
		else:
			score = compute_zscore(observation,prediction)	
		return score
		

class InjectedCurrentSpikeWidthTest(SpikeWidthTest):
	"""
	Tests the full widths of spikes at their half-maximum 
	under current injection.
	"""

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


class RestingPotentialTest(VmTest):
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
		model.rerun = True
		current_ampl = 0.0
		model.inject_current({'ampl':0.0}) 
		vm = model.get_median_vm() # Use median instead of mean for robustness.  
		prediction = {'mean':vm}
		return prediction

	def compute_score(self, observation, prediction):
		"""Implementation of sciunit.Test.score_prediction."""
		score = compute_zscore(observation,prediction)	
		return score

