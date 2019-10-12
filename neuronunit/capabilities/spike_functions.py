"""Auxiliary helper functions for analysis of spiking."""

import numpy as np
import neo
from elephant.spike_train_generation import threshold_detection
from quantities import mV, ms
from numba import jit
import sciunit
import math
import pdb

def get_spike_train(vm, threshold=0.0*mV):
    """
    Inputs:
     vm: a neo.core.AnalogSignal corresponding to a membrane potential trace.
     threshold: the value (in mV) above which vm has to cross for there
                to be a spike.  Scalar float.

    Returns:
     a neo.core.SpikeTrain containing the times of spikes.
    """
    spike_train = threshold_detection(vm, threshold=threshold)
    return spike_train

#@jit
def get_diff(vm):
    #differentiated = np.diff(vm)
    differentiated = np.diff([float(v) for v in vm])
    spikes = len([np.any(differentiated) > 0.0000193667327364])
    times = vm.times[np.where(np.any(differentiated) > 0.0000193667327364)]
    return (spikes, times)


#@jit
def diff(vm):
    differentiated = np.diff(vm)
    return differentiated

# Membrane potential trace (1D numpy array) to matrix of spike snippets (2D numpy array)
def get_spike_waveforms(vm, threshold=0.0*mV, width=10*ms):
    """
    Membrane potential trace (1D numpy array) to matrix of
    spike snippets (2D numpy array)

    Inputs:
     vm: a neo.core.AnalogSignal corresponding to a membrane potential trace.
     threshold: the value (in mV) above which vm has to cross for there
                to be a spike.  Scalar float.
     width: the length (in ms) of the snippet extracted,
            centered at the spike peak.

    Returns:
     a neo.core.AnalogSignal where each column contains a membrane potential
     snippets corresponding to one spike.
    """
    spike_train = threshold_detection(vm, threshold=threshold)

    # Fix for 0-length spike train issue in elephant.
    if len(spike_train) == 0:
        diff,times = get_diff(vm)
        temp = float(np.mean(vm))+float(np.std(vm))
        spike_train = threshold_detection(vm,threshold=temp)
        if len(spike_train) == 0:
            spike_train = threshold_detection(vm,threshold=np.max(vm))
            if len(spike_train) == 0:
                return None

    too_short = False
    too_long = False
    last_t = spike_train[-1]
    first_t = spike_train[0]

    if first_t-width/2.0 < 0.0*ms:
        too_short = True
    if last_t+width/2.0 > vm.times[-1]:
        too_long = True

    if not too_short and not too_long:
        snippets = [vm.time_slice(t-width/2, t+width/2) for t in spike_train]
    elif too_long:
        snippets = [vm.time_slice(t-width/2, t) for t in spike_train]
    elif too_short:
        snippets = [vm.time_slice(t, t+width/2) for t in spike_train]

    result = neo.core.AnalogSignal(np.array(snippets).T.squeeze(),
                                   units=vm.units,
                                   sampling_rate=vm.sampling_rate)
    return result


def spikes2amplitudes(spike_waveforms):
    """
    IN:
     spike_waveforms: Spike waveforms, e.g. from get_spike_waveforms().
        neo.core.AnalogSignal
    OUT:
     1D numpy array of spike amplitudes, i.e. the maxima in each waveform.
    """

    if spike_waveforms is not None:
        if len(spike_waveforms)==1:
            ampls = np.max(np.array(spike_waveforms))
        elif len(spike_waveforms)>1:
            pre_ampls = []
            for mp in spike_waveforms:
                pre_ampls.append(np.max(np.array(mp)))
            ampls = pre_ampls[0]
        elif len(spike_waveforms) == 0:
            ampls = np.array([])

    else:
        ampls = np.array([])

    return ampls * spike_waveforms.units

def spikes2widths(spike_waveforms):
    """
    IN:
     spike_waveforms: Spike waveforms, e.g. from get_spike_waveforms().
        neo.core.AnalogSignal
    OUT:
     1D numpy array of spike widths, specifically the full width
     at half the maximum amplitude.
    """
    n_spikes = spike_waveforms.shape[1]
    widths = []
    index_high = 0

    for i in range(n_spikes):
        s = spike_waveforms[:,i].squeeze()
        try:
            index_high = int(np.argmax(s))
            high = s[index_high]
        except:
            index_high = 0
            # dont assume spikes are above zero.
            high = np.mean(s)
            for k in s:
                for i,j in enumerate(k):
                    if j>high:
                        high  = j
                        index_high = i

        if index_high > 0:
            try: # Use threshold to compute half-max.
                y = np.array(s)
                dvdt = diff(y)
                trigger = dvdt.max()/10.0
                x_loc = int(np.where(dvdt >= trigger)[0][0])
                thresh = (s[x_loc]+s[x_loc+1])/2
                mid = (high+thresh)/2
            except: # Use minimum value to compute half-max.
                #sciunit.log(("Could not compute threshold; using pre-spike "
                #             "minimum to compute width"))
                low = np.min(s[:index_high])
                mid = (high+low)/2
            n_samples = sum(s > mid)  # Number of samples above the half-max.
            widths.append(n_samples)
    widths = np.array(widths, dtype='float')
    if n_spikes:
        # Convert from samples to time.
        widths_ = widths*spike_waveforms.sampling_period.simplified
    return widths_


def spikes2thresholds(spike_waveforms):
    """
    IN:
     spike_waveforms: Spike waveforms, e.g. from get_spike_waveforms().
        neo.core.AnalogSignal
    OUT:
     1D numpy array of spike thresholds, specifically the membrane potential
     at which 1/10 the maximum slope is reached.

    If the derivative contains NaNs, probably because vm contains NaNs
    Return an empty list with the appropriate units

    """

    n_spikes = spike_waveforms.shape[1]
    thresholds = []
    if n_spikes > 1:
        # good to know can handle multispikeing
        pass
    for i in range(n_spikes):
        s = spike_waveforms[:, i].squeeze()
        s = np.array(s)
        dvdt = np.diff(s)
        for j in dvdt:
            if math.isnan(j):
                return thresholds * spike_waveforms.units
        try:
            trigger = dvdt.max()/10.0
        except:
            return None
            # try this next.
            # return thresholds * spike_waveforms.units


        x_loc = np.where(dvdt >= trigger)[0][0]
        thresh = (s[x_loc]+s[x_loc+1])/2
        thresholds.append(thresh)
    return thresholds * spike_waveforms.units
