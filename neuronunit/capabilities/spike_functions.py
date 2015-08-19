"""A module of auxiliary helper functions, not capabilities."""

import neo
from elephant.spike_train_generation import threshold_detection
from quantities import mV, ms

# Membrane potential trace (1D numpy array) to matrix of spike snippets (2D numpy array)
def get_spike_waveforms(vm, threshold=0.0*mV, width=5*ms): 
	""" 
	 vm: a neo.core.AnalogSignal corresponding to a membrane potential trace.   
	 threshold: the value (in mV) above which vm has to cross for there 
	 			to be a spike.  Scalar float.  
	 width: the length (in ms) of the snippet extracted, 
	 		centered at the spike peak.  
	
	Returns:
	 a neo.core.AnalogSignalArray of membrane potential snippets 
	 corresponding to each spike.  
	"""
	spike_train = threshold_detection(vm,threshold=threshold)
	vm_array = neo.core.AnalogSignalArray(vm,units=vm.units,
											 sampling_rate=vm.sampling_rate)
	snippets = [vm_array.time_slice(t-width/2,t+width/2) for t in spikes]
	return neo.core.AnalogSignalArray(snippets,units=vm.units,
											   sampling_rate=vm.sampling_rate)

def get_spike_amplitudes(spike_waveforms):
	""" 
	IN:
	 spikes: Spike waveforms, e.g. from get_spike_waveforms(). 
	 		 neo.core.AnalogSignalArray
	OUT:
	 1D numpy array of spike amplitudes, i.e. the maxima in each waveform.     
	"""
	return np.max(spikes,axis=1)
	

def spikes2widths(spikes, scale=[0.1*ms,1*mV]):
	""" 
	IN:
	 spikes: Spike waveforms, e.g. from vm2spikes(). 2D numpy array, see vm2spikes output.    
	 scale[0]: the duration of time (in s) corresponding to one sample (point), i.e. dt. Scalar float.    
	 scale[1]: the scale (in mV) of the vm array, i.e. vm=3 corresponds to 3*scale[1] mV.  Scalar float.  
	OUT:
	 1D numpy array of spike widths, specifically the full width at half the maximum amplitude.     
	"""
	widths = []
	print("There are %d spikes" % len(spikes))
	for spike in spikes:
		#print("This spikes has duration %d samples" % len(spike))
		x_high = np.argmax(spike)
		print("Spikes has duration %d samples, and sample %d is the high point" % (len(spike),x_high))
		high = spike[x_high]
		if x_high > 0:
			low = np.min(spike[:x_high])
			mid = (high+low)/2
			spike_top = spike[(spike>mid)]
			widths.append(len(spike_top))
			print(low,mid,high)
	widths = np.array(widths)*scale[0]
	print("Spike widths are %s" % str(widths))
	return widths

# https://gist.github.com/endolith/250860
def peakdet(v, delta, x = None):
	"""
	(maxtab, mintab) = peakdet(v, delta) finds the local maxima and minima ("peaks") 
	in the vector V. maxtab and mintab consists of two columns. 
	Column 1 contains indices in v, and column 2 the found values.
		  
	With [maxtab, mintab] = peakdet(v, delta, x) the indices
	in maxtab and mintab are replaced with the corresponding x-values.
	
	A point is considered a maximum peak if it has the maximal
	value, and was preceded (to the left) by a value lower by delta.
	"""
	maxtab = []
	mintab = []
	   
	if x is None:
		x = np.arange(len(v))
	
	v = np.asarray(v)
	
	if len(v) != len(x):
		sys.exit('Input vectors v and x must have same length')
	
	if not np.isscalar(delta):
		sys.exit('Input argument delta must be a scalar')
	
	if delta <= 0:
		pass#sys.exit('Input argument delta must be positive')
	
	mn, mx = np.inf, -np.inf
	mnpos, mxpos = np.nan, np.nan
	
	lookformax = True
	
	for i in np.arange(len(v)):
		this = v[i]
		if this > mx:
			mx = this
			mxpos = x[i]
		if this < mn:
			mn = this
			mnpos = x[i]
		
		if lookformax:
			if this < mx-delta:
				maxtab.append((mxpos, mx))
				mn = this
				mnpos = x[i]
				lookformax = False
		else:
			if this > mn+delta:
				mintab.append((mnpos, mn))
				mx = this
				mxpos = x[i]
				lookformax = True
 
	return np.array(maxtab), np.array(mintab)
