"""NeuroUnit capability classes implemented using a combination of
NeuroConstruct and NeuroTools."""

from __init__ import *
from datetime import datetime
from sciunit import Capability
from sciunit.capabilities import Runnable
from neuronunit import *
from neuronunit.capabilities import *
if CPYTHON:
	from pythonnC.utils import putils as utils # From the neuroConstruct pythonnC package. 
	import numpy as np
if JYTHON:
	from pythonnC.utils import jutils as utils # From the neuroConstruct pythonnC package. 
from pythonnC.utils import neurotools as nc_neurotools
JUTILS_PATH = 'pythonnC.utils.jutils'

class Runnable_NC(Runnable):
	"""Implementation of Runnable for neuroConstruct."""

	def __init__(self):
		self.ran = False
		self.rerun = False
		self.runtime_methods = {}
		if CPYTHON:
			self.gateway = utils.open_gateway(useSocket=True,
									automatic_socket=utils.AUTOMATIC_SOCKET)
			cmd = 'import %s as j;' % JUTILS_PATH
			cmd += 'import sys;'
			cmd += 'j.sim = j.Sim(project_path="%s");' % self.project_path
			cmd += 'channel.send(0)'
  			channel = self.gateway.remote_exec(cmd)
  			channel.receive()

	def run(self):
		"""Runs the model using jython via execnet and returns a 
		directory of simulation results"""
		if self.ran is False or self.rerun is True:
			print "Running simulation..."
			self.sim_path = utils.run_sim(project_path=self.project_path,
									   	  useSocket=True,
									      useNC=True,
									      useNeuroTools=True,
									      runtime_methods=self.runtime_methods,
									      gw=self.gateway)
			self.run_t = datetime.now()
			self.ran = True
			self.rerun = False
		else:
			print "Already ran simulation..."


class ProducesMembranePotential_NC(ProducesMembranePotential,Runnable):
	"""An array of somatic membrane potential samples"""
	
	def get_membrane_potential(self,**kwargs):
		"""Returns a NeuroTools.signals.analogs.AnalogSignal object"""
		
		if self.sim_path is None or self.ran is False:
			self.run(**kwargs)
		elif self.sim_path == '':
			vm = None
		else:
			vm = nc_neurotools.get_analog_signal(self.sim_path,
												 self.population_name) 
			# An AnalogSignal instance. 
		return vm 
  		
	def get_median_vm(self,**kwargs):
		"""Returns a float corresponding the median membrane potential.
		This will in some cases be the resting potential."""
		
		vm = self.get_membrane_potential(**kwargs) 
		# A NeuroTools.signals.AnalogSignal object
		median_vm = np.median(vm.signal)
		return median_vm

	def get_initial_vm(self):
		"""Returns a float corresponding to the starting membrane potential.
		This will in some cases be the resting potential."""
		vm = self.get_membrane_potential() 
		# A NeuroTools.signals.AnalogSignal object
		return vm.signal[0]


class ProducesSpikes_NC(ProducesSpikes,ProducesMembranePotential):
	"""Requires ProducesMembranePotential.
	Produces MembranePotentialNC is a logical choice.""" 
	
	def get_spikes(self,**kwargs):
		"""Returns an array of spike snippets"""
		vm = self.get_membrane_potential(**kwargs) 
		# A NeuroTools.signals.AnalogSignal object
		return spike_functions.vm2spikes(vm.signal)

	def get_spike_train(self,**kwargs):
		"""Returns a NeuroTools.signals.spikes.SpikeTrain object"""
		vm = self.get_membrane_potential(**kwargs) 
		# A NeuroTools.signals.AnalogSignal object
		return vm.threshold_detection()


class ReceivesCurrent_NC(ReceivesCurrent,Runnable_NC):
	"""An array of somatic injected current samples"""
	
	def __init__(self):
		self.current = self.Current()

	class Current(object):
		ampl = 0
		duration = 0
		offset = 0

	def set_current_ampl(self,current_ampl):
		cmd = 'import %s as j;' % JUTILS_PATH
		cmd += 'import sys;'
		cmd += 'err = j.sim.set_current_ampl(%f);' % current_ampl
		cmd += 'channel.send(err);'
		channel = self.gateway.remote_exec(cmd)
		err = channel.receive() # This will be an error code.  
		if len(err):
			raise NotImplementedError(err)
		#self.current.ampl = current_ampl
		#self.runtime_methods['set_current_ampl']=[current_ampl]

