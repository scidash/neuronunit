from numpy import *
import scipy
#from scipy.units.api import UnitArray,has_units
#from scimath.units.electromagnetism import volts,amps
#from scimath.units.time import seconds

SCALE = [1e-4,1] # dt = 0.0001; units = mV or pA.  

# Membrane potential trace (1D numpy array) to matrix of spike snippets (2D numpy array)
def vm2spikes(vm,scale=SCALE,threshold=0,width=0.01): # Width in s.  
	# vm: the membrane potential trace.  
	# scale[0]: the duration of time (in s) corresponding to one sample (point), i.e. dt.  
	# scale[1]: the scale (in mV) of the vm array, i.e. vm=3 corresponds to 3*scale[1] mV.   
	# threshold: the value (in mV) above which vm has to cross for there to be a spike.  
	# width: the length (in s) of the snippet extracted, centered at the spike peak.  
	vm = array(vm)  
	n_samples = len(vm)
	(maxima,minima) = peakdet(vm,0)#,x=times)
	threshold = float(threshold)/scale[1]
	width = float(width)/scale[0] # Convert to samples.
	vm = concatenate((ones(width)*vm[0],vm,ones(width)*vm[n_samples-1])) # Prepend and postpend buffer
	spikes = [vm[maximum[0]:maximum[0]+2*width] for maximum in maxima if maximum[1]>threshold]
	return array(spikes)

def spikes2amplitudes(spikes,scale=SCALE):
	return amax(spikes,axis=1)*scale[1]

def spikes2widths(spikes,scale=SCALE):
	widths = []
	for spike in spikes:
		x_high = argmax(spike)
		high = spike[x_high]
		low = min(spike[:x_high])
		mid = (high+low)/2
		spike_top = spike[(spike>mid)]
		widths.append(len(spike_top))
	return array(widths)*scale[0]

# https://gist.github.com/endolith/250860
def peakdet(v, delta, x = None):
    """
    [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local maxima and minima ("peaks") 
    in the vector V. MAXTAB and MINTAB consists of two columns. 
    Column 1 contains indices in V, and column 2 the found values.
          
    With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
    in MAXTAB and MINTAB are replaced with the corresponding X-values.
    
    A point is considered a maximum peak if it has the maximal
    value, and was preceded (to the left) by a value lower by DELTA.
    """
    maxtab = []
    mintab = []
       
    if x is None:
        x = arange(len(v))
    
    v = asarray(v)
    
    if len(v) != len(x):
        sys.exit('Input vectors v and x must have same length')
    
    if not isscalar(delta):
        sys.exit('Input argument delta must be a scalar')
    
    if delta <= 0:
        pass#sys.exit('Input argument delta must be positive')
    
    mn, mx = Inf, -Inf
    mnpos, mxpos = NaN, NaN
    
    lookformax = True
    
    for i in arange(len(v)):
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
 
    return array(maxtab), array(mintab)
