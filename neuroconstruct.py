"""
Implementation of a model built in neuroConstruct.
http://www.neuroconstruct.org/
"""

from sciunit import Candidate
#from ncutils import SimulationManager # neuroConstruct/pythonNeuroML/nCUtils/ncutils.py
from neurounit.capabilities import ProducesMembranePotential,ProducesSpikes
from neurounit.capabilities import ReceivesCurrent
from sciunit.capabilities import Runnable
import spike_functions
import numpy as np

class NeuroConstructModel(#SimulationManager,
						  Candidate,
						  Runnable,
						  ProducesMembranePotential,
						  ProducesSpikes):
	"""Implementation of a candidate model usable by neuroConstruct (written in neuroML).
	Execution takes places in the neuroConstruct program.
	Methods will be implemented using the neuroConstruct python 
	API (in progress)."""

	def __init__(self,**kwargs):
		for key,value in kwargs.items():
			setattr(self,key,value)

	ran = False
	
	def get_membrane_potential(self):
		return self.vm

	def get_spikes(self):
		vm = self.get_membrane_potential()
		return spike_functions.vm2spikes(vm)

	def run(self,**kwargs):
		"""Put some SimulationManager method here when Padraig writes it."""  
		self.ran = True
	
class FakeNeuroConstructModel(NeuroConstructModel,
							  ReceivesCurrent):
	"""A fake neuroConstruct model that generates a gaussian noise 
	membrane potential with some 'spikes'. Eventually I will make the membrane
	potential and the spikes change as a function of the current."""
	
	def run(self,**kwargs):
		n_samples = getattr(self,'n_samples',10000)
		self.vm = np.random.randn(n_samples)-65.0 # -65 mV with gaussian noise.  
		for i in range(200,n_samples,200): # Make 50 spikes.  
			self.vm[i:i+10] += 10.0*np.array(range(10)) # Shaped like right triangles.  
		super(FakeNeuroConstructModel,self).run(**kwargs)