import sciunit
import capabilities
import spike_functions
import neuroelectro
from numpy import *
from sciunit.test import RickTest
from sciunit.comparators import ZComparator # Comparators.  
from sciunit.comparators import ZScoreToBooleanScore # Converters.  
from sciunit.scores import BooleanScore # Scores.  

"""
class NeuronTest(sciUnit.Test):
	def __init__(self,neurolex_id):
		"""neurolex_id is the id on neurolex.org for the neuron to be used as a reference."""
		x = NeuroElectroSummaryTest()
		x.set_neuron(nlex_id=neurolex_id)
		x.set_ephysprop(name='width')
		x.get_values()
		self.reference_data['width'] = x.mean
"""

class SpikeWidthTest(RickTest):
	required_capabilities += (capabilities.ProducesMembranePotential,
							 capabilities.ProducesSpikes)
	def __init__(self,mean,std):
		"""Takes the mean and standard deviation of reference spike widths"""
		super(self).__init__(self,model_stats,reference_stats)

	comparator = ZComparator
	converter = ZScoreToBooleanScore
	conversion_params = {'thresh':2}
		
	def get_data(self,model):
		"""Extracts data from the model and returns it."""
		spikes = model.get_spikes() # Method implementation guaranteed by ProducesSpikes capability.  
		widths = spike_functions.spikes2widths(spikes)
		model_data = {'mean':mean(widths),'std':std(widths)}
		return model_data

	def get_stats(self,model_data):
		"""Puts stats in a form that the Comparator will understand."""
		model_stats = {'value':self.model_data['mean']}
		reference_stats = {'mean':self.reference_data['mean'],
						   'std':self.reference_data['std']}
		return (model_stats,reference_stats)
	
	def score(self,model_data):
		"""Return a score for the model on this test."""  
		(model_stats,reference_stats) = self.get_stats(model_data)
		comparator = self.comparator(model_stats,reference_stats) # A Z-score.
		score = comparator.score(converter=self.converter,{'thresh':2})
		score.model_data = model_data
		score.reference_data = self.reference_data
		return score
	

	