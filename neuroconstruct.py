"""
Implementation of a model built in neuroConstruct.
http://www.neuroconstruct.org/
"""

from sciunit import Model
from ncutils import SimulationManager # neuroConstruct/pythonNeuroML/nCUtils/ncutils.py
from capabilities import ProducesMembranePotential,ProducesSpikes,Runnable
import spike_functions

class neuroConstructModel(SimulationManager,
						  Model,
						  Runnable,
						  ProducesMembranePotential,
						  ProducesSpikes)
	ran = False
	
	def get_membrane_potential(self):
		return self.volts

	def get_spikes(self):
		vm = self.get_membrane_potential()
		return spike_functions.vm2spikes(vm)

	def run(self,**kwargs):
		self.runMultipleSims(**kwargs) # runMultipleSims implemented in SimulationManager.
		ran = True
	
