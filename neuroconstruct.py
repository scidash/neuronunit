import sciunit
import ncutils # neuroConstruct/pythonNeuroML/nCUtils/ncutils.py
from capabilities import ProducesMembranePotential,ProducesSpikes
import functions

class neuroConstructModel(ncutils.SimulationManager,ProducesMembranePotential,ProducesSpikes)
	ran = False
	
	def get_membrane_potential(self):
		return self.volts

	def get_spikes(self):
		vm = self.get_membrane_potential()
		return functions.vm2spikes(vm)

	def run(**kwargs):
		self.runMultipleSims(**kwargs)
		ran = True
	
