import sciunit
import capabilities
import functions
import neuroelectro
from numpy import *

class SpikeWidthTest(neuroelectro.NeuroElectroSummaryTest):
	required_capabilities = (capabilities.ProducesMembranePotential,
							 capabilities.ProducesSpikes)
	def run_test(self,model):
		self.get_values()
		if not model.ran:
			model.run()
		spikes = model.get_spikes()
		widths = functions.spikes2widths(spikes)
		width_stats = mean(widths),std(widths)
		score = sciunit.ZScoreMap(width_stats)
		return sciunit.ZScore(score)