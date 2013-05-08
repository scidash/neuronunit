"""
Implementation of a model built in neuroConstruct.
http://www.neuroconstruct.org/
"""
from __future__ import absolute_import

import sys,os
try:
	NC_HOME = os.environ["NC_HOME"]
except KeyError:
	raise Exception("Please add an NC_HOME environment variable corresponding\
					 to the location of the neuroConstruct directory.")

sys.path.append(NC_HOME)

from pythonnC.utils import putils # From the neuroConstruct pythonnC package.  
from sciunit import Candidate
from neurounit.capabilities import ProducesMembranePotential,ProducesSpikes
from neurounit.capabilities import ReceivesCurrent
from sciunit.capabilities import Runnable
import spike_functions
import numpy as np

class NeuroConstructModel(Candidate,
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
		"""Returns a NeuroTools.signals.analogs.AnalogSignal object"""
		return putils.get_vm(run=(not self.ran))   

	def get_spikes(self):
		"""Returns an array of spike snippets"""
		vm = self.get_membrane_potential() # A NeuroTools.signals.AnalogSignal object
		return spike_functions.vm2spikes(vm.signal)

	def get_spike_train(self):
		"""Returns a NeuroTools.signals.spikes.SpikeTrain object"""
		vm = self.get_membrane_potential() # A NeuroTools.signals.AnalogSignal object
		return vm.threshold_detection()

	def run(self,**kwargs):
		"""Runs the model using jython via execnet and returns a 
		directory of simulation results"""
		print "Running simulation..."
		simDir = putils.run_sim(useSocket=True,useNC=True,useNeuroTools=True)
  		self.ran = True
  		return simDir
	
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