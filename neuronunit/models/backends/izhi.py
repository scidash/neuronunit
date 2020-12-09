import math

from quantities import mV, ms, s, V
import sciunit
from neo import AnalogSignal
import neuronunit.capabilities as cap
import numpy as np
from .base import *
import quantities as qt
#import matplotlib as mpl
try:
    import asciiplotlib as apl
except:
    pass
import numpy
voltage_units = mV

from elephant.spike_train_generation import threshold_detection
import time
def timer(func):
    def inner(*args, **kwargs):
        t1 = time.time()
        f = func(*args, **kwargs)
        t2 = time.time()
        print('time taken on block {0} '.format(t2-t1))
        return f
    return inner

from numba import jit, autojit
import cython
@jit(nopython=True)
def get_vm_matlab_four(C=89.7960714285714,
         a=0.01, b=15, c=-60, d=10, k=1.6,
         vPeak=(86.364525297619-65.2261863636364),
          vr=-65.2261863636364, vt=-50,celltype=1, N=0,start=0,stop=0,amp=0,ramp=None):
    tau = dt = 0.25
    if ramp is not None:
        N = len(ramp)
    v = np.zeros(N)
    u = np.zeros(N)
    v[0] = vr
    for i in range(N-1):
        I = 0
        if ramp is not None:
            I = ramp[i]
        elif start <= i <= stop:
               I = amp
        # forward Euler method
        v[i+1] = v[i] + tau * (k * (v[i] - vr) * (v[i] - vt) - u[i] + I) / C
        u[i+1] = u[i]+tau*a*(b*(v[i]-vr)-u[i]); # Calculate recovery variable

        if v[i+1] > (vPeak - 0.1*u[i+1]):
            v[i] = vPeak - 0.1*u[i+1]
            v[i+1] = c + 0.04*u[i+1]; # Reset voltage
            if (u[i]+d)<670:
                u[i+1] = u[i+1]+d; # Reset recovery variable
            else:
                u[i+1] = 670;

    return v

@jit(nopython=True)
def get_vm_matlab_five(C=89.7960714285714,
         a=0.01, b=15, c=-60, d=10, k=1.6,
         vPeak=(86.364525297619-65.2261863636364),
          vr=-65.2261863636364, vt=-50,celltype=1, N=0,start=0,stop=0,amp=0,ramp=None):

    tau= dt = 0.25; #dt
    if ramp is not None:
        N = len(ramp)
    v = np.zeros(N)
    u = np.zeros(N)
    v[0] = vr
    for i in range(N-1):
        I = 0
        if ramp is not None:
            I = ramp[i]
        elif start <= i <= stop:
            I = amp
	# forward Euler method
        v[i+1] = v[i] + tau * (k * (v[i] - vr) * (v[i] - vt) - u[i] + I) / C

        #u[i+1]=u[i]+tau*a*(b*(v[i]-vr)-u[i]); # Calculate recovery variable
        if v[i+1] < d:
            u[i+1] = u[i] + tau*a*(0-u[i])
        else:
            u[i+1] = u[i] + tau*a*((0.025*(v[i]-d)**3)-u[i])
        if v[i+1]>=vPeak:
            v[i]=vPeak;
            v[i+1]=c;

    return v


@jit(nopython=True)
def get_vm_matlab_seven(C=89.7960714285714,
         a=0.01, b=15, c=-60, d=10, k=1.6,
         vPeak=(86.364525297619-65.2261863636364),
          vr=-65.2261863636364, vt=-50,celltype=1, N=0,start=0,stop=0,amp=0,ramp=None):
    tau= dt = 0.25; #dt

    if ramp is not None:
        N = len(ramp)
    v = np.zeros(N)
    u = np.zeros(N)
    v[0] = vr
    for i in range(N-1):
        I = 0
        if ramp is not None:
            I = ramp[i]
        elif start <= i <= stop:
            I = amp

        # forward Euler method
        v[i+1] = v[i] + tau * (k * (v[i] - vr) * (v[i] - vt) - u[i] + I) / C


        if v[i+1] > -65:
            b=2;
        else:
            b=10;

        u[i+1]=u[i]+tau*a*(b*(v[i]-vr)-u[i]);
        if v[i+1]>=vPeak:
            v[i]=vPeak;
            v[i+1]=c;
            u[i+1]=u[i+1]+d;  # reset u, except for FS cells


    return v

@jit(nopython=True)
def get_vm_matlab_six(C=89.7960714285714,
         a=0.01, b=15, c=-60, d=10, k=1.6,
         vPeak=(86.364525297619-65.2261863636364),
          vr=-65.2261863636364, vt=-50,celltype=1, N=0,start=0,stop=0,amp=0,ramp=None):
    tau= dt = 0.25; #dt

    if ramp is not None:
        N = len(ramp)
    v = np.zeros(N)
    u = np.zeros(N)
    v[0] = vr
    for i in range(N-1):
        I = 0
        if ramp is not None:
            I = ramp[i]
        elif start <= i <= stop:
            I = amp
       # forward Euler method
        v[i+1] = v[i] + tau * (k * (v[i] - vr) * (v[i] - vt) - u[i] + I) / C


        u[i+1]=u[i]+tau*a*(b*(v[i]-vr)-u[i]);
        if v[i+1] > -65:
            b=0;
        else:
            b=15;
        if v[i+1] > (vPeak + 0.1*u[i+1]):
            v[i]= vPeak + 0.1*u[i+1];
            v[i+1] = c-0.1*u[i+1]; # Reset voltage
            u[i+1]=u[i+1]+d;

    return v



@jit(nopython=True)
def get_vm_matlab_one_two_three(C=89.7960714285714,
         a=0.01, b=15, c=-60, d=10, k=1.6,
         vPeak=(86.364525297619-65.2261863636364),
          vr=-65.2261863636364, vt=-50,
          N=0,start=0,stop=0,amp=0,ramp=None):
    tau= dt = 0.25; #dt
    if ramp is not None:
        N = len(ramp)
    v = np.zeros(N)
    u = np.zeros(N)
    v[0] = vr
    for i in range(N-1):
        I = 0
        if ramp is not None:
            I = ramp[i]
        elif start <= i <= stop:
            I = amp
       # forward Euler method
        v[i+1] = v[i] + tau * (k * (v[i] - vr) * (v[i] - vt) - u[i] + I) / C
        u[i+1]=u[i]+tau*a*(b*(v[i]-vr)-u[i]); # Calculate recovery variable
        #u[i+1]=u[i]+tau*a*(b*(v[i]-vr)-u[i]); # Calculate recovery variable

        if v[i+1]>=vPeak:
            v[i]=vPeak
            v[i+1]=c
            u[i+1]=u[i+1]+d  # reset u, except for FS cells
    return v



class IZHIBackend(Backend):

    name = 'IZHI'

    def init_backend(self, attrs=None,DTC=None,
                     debug = False):
        super(IZHIBackend,self).init_backend()
        self.model._backend.use_memory_cache = False

        self.attrs = attrs

        self.temp_attrs = None
        self.default_attrs = {'C':89.7960714285714,
            'a':0.01, 'b':15, 'c':-60, 'd':10, 'k':1.6,
            'vPeak':(86.364525297619-65.2261863636364),
            'vr':-65.2261863636364, 'vt':-50, 'celltype':3}

        if type(attrs) is not type(None):
            self.attrs = attrs
            # set default parameters anyway.
        if type(DTC) is not type(None):
            if type(DTC.attrs) is not type(None):
                self.set_attrs(**DTC.attrs)
                print('gets here b')
        if self.attrs is None:
            #print('gets here a')
            self.attrs = self.default_attrs


    def get_spike_count(self):
        thresh = threshold_detection(self.vM,0*qt.mV)
        return len(thresh)


    def set_stop_time(self, stop_time = 650*pq.ms):
        """Sets the simulation duration
        stopTimeMs: duration in milliseconds
        """
        self.tstop = float(stop_time.rescale(pq.ms))


    def get_membrane_potential(self):
        """Must return a neo.core.AnalogSignal.
        """
        if type(self.vM) is not type(None):
            return self.vM


        if type(self.vM) is type(None):

            everything = copy.copy(self.attrs)
            if hasattr(self,'Iext'):
                everything.update({'Iext':self.Iext})

            if 'current_inj' in everything.keys():
                everything.pop('current_inj',None)
            everything = copy.copy(self.attrs)

            self.attrs['celltype'] = int(round(self.attrs['celltype']))
            assert type(self.attrs['celltype']) is type(int())
            if self.attrs['celltype'] <= 3:
                everything.pop('celltype',None)
                v = get_vm_matlab_one_two_three(**everything)
            else:
                if self.attrs['celltype'] == 4:
                    v = get_vm_matlab_four(**everything)
                if self.attrs['celltype'] == 5:
                    v = get_vm_matlab_five(**everything)
                if self.attrs['celltype'] == 6:
                    v = get_vm_matlab_six(**everything)
                if self.attrs['celltype'] == 7:

                    v = get_vm_matlab_seven(**everything)

            self.vM = AnalogSignal(v,
                                units=pq.mV,
                                sampling_period=0.25*pq.ms)


        return self.vM

    def set_attrs(self, attrs):
        self.attrs = attrs
        self.model.attrs.update(attrs)



    def step(amplitude, t_stop):
       """
       Generate the waveform for a current that starts at zero and is stepped up
       to the given amplitude at time t_stop/10.
       """
       times = np.array([0, t_stop/10, t_stop])
       amps = np.array([0, amplitude, amplitude])
       return times, amps


    def pulse(amplitude, onsets, width, t_stop, baseline=0.0):
        """
        Generate the waveform for a series of current pulses.
        Arguments:
		amplitude - absolute current value during each pulse
		onsets - a list or array of times at which pulses begin
		width - duration of each pulse
		t_stop - total duration of the waveform
		baseline - the current value before, between and after pulses.
        """
        times = [0]
        amps = [baseline]
        for onset in onsets:
           times += [onset, onset + width]
           amps += [amplitude, baseline]
        times += [t_stop]
        amps += [baseline]
        return np.array(times), np.array(amps)


    def ramp(self,gradient, onset, t_stop, baseline=0.0, time_step=0.125, t_start=0.0):
        """
        Generate the waveform for a current which is initially constant
        and then increases linearly with time.
        Arguments:
		gradient - gradient of the ramp
		onset - time at which the ramp begins
		t_stop - total duration of the waveform
		baseline - current value before the ramp
		time_step - interval between increments in the ramp current
		t_start - time at which the waveform begins (used to construct waveforms
		          containing multiple ramps).
        """
        if onset > t_start:
            times = np.hstack((np.array((t_start, onset)),  # flat part
	                   np.arange(onset + time_step, t_stop + time_step, time_step)))  # ramp part
        else:
            times = np.arange(t_start, t_stop + time_step, time_step)
        amps = baseline + gradient*(times - onset) * (times > onset)
        return times, amps

    def inject_ramp_current(self, t_stop, gradient=0.000015, onset=30.0, baseline=0.0, t_start=0.0):
        times, amps = self.ramp(gradient, onset, t_stop, baseline=0.0, t_start=0.0)

        everything = copy.copy(self.attrs)

        everything.update({'ramp':amps})
        everything.update({'start':onset})
        everything.update({'stop':t_stop})

        if 'current_inj' in everything.keys():
            everything.pop('current_inj',None)

        self.attrs['celltype'] = round(self.attrs['celltype'])
        if np.bool_(self.attrs['celltype'] <= 3):
            everything.pop('celltype',None)
            v = get_vm_matlab_one_two_three(**everything)
        else:



            if np.bool_(self.attrs['celltype'] == 4):
                v = get_vm_matlab_four(**everything)
            if np.bool_(self.attrs['celltype'] == 5):
                v = get_vm_matlab_five(**everything)
            if np.bool_(self.attrs['celltype'] == 6):
                v = get_vm_matlab_six(**everything)
            if np.bool_(self.attrs['celltype'] == 7):
                v = get_vm_matlab_seven(**everything)


        self.attrs

        self.vM = AnalogSignal(v,
                            units=pq.mV,
                            sampling_period=0.125*pq.ms)

        return self.vM

    def stepify(times, values):
        """
        Generate an explicitly-stepped version of a time series.
        """
        new_times = np.empty((2*times.size - 1,))
        new_values = np.empty_like(new_times)
        new_times[::2] = times
        new_times[1::2] = times[1:]
        new_values[::2] = values
        new_values[1::2] = values[:-1]
        return new_times, new_values


    @cython.boundscheck(False)
    @cython.wraparound(False)
    #@timer
    def inject_square_current(self, current):
        """
        Inputs: current : a dictionary with exactly three items, whose keys are: 'amplitude', 'delay', 'duration'
        Example: current = {'amplitude':float*pq.pA, 'delay':float*pq.ms, 'duration':float*pq.ms}}
        where \'pq\' is a physical unit representation, implemented by casting float values to the quanitities \'type\'.
        Description: A parameterized means of applying current injection into defined
        Currently only single section neuronal models are supported, the neurite section is understood to be simply the soma.

        """

        attrs = self.attrs
        #print(attrs)
        if attrs is None:
            attrs = self.default_attrs
            #print('gets here')

        self.attrs = attrs
        if 'delay' in current.keys() and 'duration' in current.keys():
            square = True
            c = current
        if isinstance(c['amplitude'],type(pq)):
            amplitude = float(c['amplitude'].simplified)
            duration = float(c['duration'])
            delay = float(c['delay'])
        else:
            amplitude = float(c['amplitude'])
            duration = float(c['duration'])
            delay = float(c['delay'])
        #print(amplitude,duration,delay)
        tMax = delay + duration #+ 200.0#*pq.ms

        self.set_stop_time(tMax*pq.ms)
        tMax = self.tstop
        N = int(tMax/0.25)
        Iext = np.zeros(N)
        delay_ind = int((delay/tMax)*N)
        duration_ind = int((duration/tMax)*N)

        Iext[0:delay_ind-1] = 0.0
        Iext[delay_ind:delay_ind+duration_ind-1] = amplitude
        Iext[delay_ind+duration_ind::] = 0.0
        self.Iext = None
        self.Iext = Iext


        everything = copy.copy(self.attrs)
        everything.update({'N':len(Iext)})

        #everything.update({'Iext':Iext})
        everything.update({'start':delay_ind})
        everything.update({'stop':delay_ind+duration_ind})
        everything.update({'amp':amplitude})

        if 'current_inj' in everything.keys():
            everything.pop('current_inj',None)
        #import pdb; pdb.set_trace()

        self.attrs['celltype'] = int(round(self.attrs['celltype']))
        assert type(self.attrs['celltype']) is type(int())

        if np.bool_(self.attrs['celltype'] <= 3):
            everything.pop('celltype',None)
            v = get_vm_matlab_one_two_three(**everything)
        else:


            if np.bool_(self.attrs['celltype'] == 4):
                v = get_vm_matlab_four(**everything)
            if np.bool_(self.attrs['celltype'] == 5):
                v = get_vm_matlab_five(**everything)
            if np.bool_(self.attrs['celltype'] == 6):
                v = get_vm_matlab_six(**everything)
            if np.bool_(self.attrs['celltype'] == 7):
                v = get_vm_matlab_seven(**everything)


        #self.model.attrs.update(attrs)
        if 'v' not in locals():
            print(self.attrs['celltype'])
        self.vM = AnalogSignal(v,
                            units=pq.mV,
                            sampling_period=0.25*pq.ms)

    def _backend_run(self):
        results = {}
        if len(self.attrs) > 1:
            v = get_vm(**self.attrs)
        else:
            v = get_vm(self.attrs)

        self.vM = AnalogSignal(v,
                               units = voltage_units,
                               sampling_period = 0.25*pq.ms)
        results['vm'] = self.vM.magnitude
        results['t'] = self.vM.times
        results['run_number'] = results.get('run_number',0) + 1
        return results
