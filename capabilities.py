import sciunit.Capability

class ProducesMembranePotential(sciunit.Capability):
	''' An array of membrane potential samples. ''' 
	def get_membrane_potential(self):
		raise NotImplementedError()

class ProducesSpikes(sciunit.Capability):
	''' A 2-D array: spike_waveform x num_spikes. ''' 
	def get_spikes(self):
		raise NotImplementedError()

