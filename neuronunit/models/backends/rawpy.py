from neuronunit.tests.base import ALLEN_STIM, ALLEN_ONSET, on_indexs, DT, ALLEN_STIM, ALLEN_STOP, ALLEN_FINISH
from quantities import mV, ms, s, V
import sciunit
from neo import AnalogSignal
import neuronunit.capabilities as cap
import numpy as np
from .base import *
import quantities as qt
from quantities import mV, ms, s
import matplotlib as mpl
import asciiplotlib as apl
import numpy

#mpl.use('Agg')

#import matplotlib.pyplot as plt
# @jit(cache=True) I suspect this causes a memory leak
from numba import jit
# @cuda.jit(device=True, cache=True)

@jit
def get_vm(C=89.7960714285714, a=0.01, b=15, c=-60, d=10, k=1.6, vPeak=(86.364525297619-65.2261863636364), vr=-65.2261863636364, vt=-50, dt=0.030, Iext=[]):
    '''
    dt determined by
    Apply izhikevich equation as model
    This function can't get too pythonic (functional), it needs to be a simple loop for
    numba/jit to understand it.
    '''
    N = len(Iext)
    v = np.zeros(N)
    u = np.zeros(N)
    v[0] = vr
    for m in range(0,N-1):
        vT = v[m]+ (dt/2) * (k*(v[m] - vr)*(v[m] - vt)-u[m] + Iext[m])/C;
        v[m+1] = vT + (dt/2)  * (k*(v[m] - vr)*(v[m] - vt)-u[m] + Iext[m])/C;
        u[m+1] = u[m] + dt * a*(b*(v[m]-vr)-u[m]);
        u[m+1] = u[m] + dt * a*(b*(v[m+1]-vr)-u[m]);
        if v[m+1]>= vPeak:# % a spike is fired!
            v[m] = vPeak;# % padding the spike amplitude
            v[m+1] = c;# % membrane voltage reset
            u[m+1] = u[m+1] + d;# % recovery variable update

    for m in range(0,N):
        v[m] = v[m]/1000.0

    return v


class RAWBackend(Backend):

    def init_backend(self, attrs=None, cell_name='alice',
                     current_src_name='hannah', DTC=None,
                     debug = False):
        backend = 'RAW'
        super(RAWBackend,self).init_backend()
        self.model._backend.use_memory_cache = False
        self.current_src_name = current_src_name
        self.cell_name = cell_name
        self.vM = None
        self.attrs = attrs
        self.debug = debug
        self.temp_attrs = None

        if type(attrs) is not type(None):
            self.set_attrs(**attrs)
            self.sim_attrs = attrs

        if type(DTC) is not type(None):
            if type(DTC.attrs) is not type(None):

                self.set_attrs(**DTC.attrs)


            if hasattr(DTC,'current_src_name'):
                self._current_src_name = DTC.current_src_name

            if hasattr(DTC,'cell_name'):
                self.cell_name = DTC.cell_name



    def set_stop_time(self, stop_time = 650*pq.ms):
        """Sets the simulation duration
        stopTimeMs: duration in milliseconds
        """
        self.tstop = float(stop_time.rescale(pq.ms))


    def get_membrane_potential(self,**run_params):
        """Must return a neo.core.AnalogSignal.
        And must destroy the hoc vectors that comprise it.
        """
        if type(self.vM) is not type(None):
            pass
        else:
            v = get_vm(**self.attrs)
            self.vM = AnalogSignal(v,
                                   units = mV,
                                   sampling_period = self.attrs['dt'] * ms)
        return self.vM

    def set_attrs(self, **attrs):
        self.attrs = attrs
        self.model.attrs.update(attrs)


    def inject_square_current(self, current):#, section = None, debug=False):
        """Inputs: current : a dictionary with exactly three items, whose keys are: 'amplitude', 'delay', 'duration'
        Example: current = {'amplitude':float*pq.pA, 'delay':float*pq.ms, 'duration':float*pq.ms}}
        where \'pq\' is a physical unit representation, implemented by casting float values to the quanitities \'type\'.
        Description: A parameterized means of applying current injection into defined
        Currently only single section neuronal models are supported, the neurite section is understood to be simply the soma.

        """
        attrs = copy.copy(self.model.attrs)

        if 'injected_square_current' in current.keys():
            c = current['injected_square_current']
        else:
            c = current
        amplitude = float(c['amplitude'])
        duration = float(c['duration'])#/dt#/dt.rescale('ms')
        delay = float(c['delay'])#/dt#.rescale('ms')
        tMax = delay + duration + 200.0#/dt#*pq.ms
        self.set_stop_time(tMax*pq.ms)
        tMax = self.tstop
        attrs['dt'] = DT
        N = int(tMax/attrs['dt'])
        Iext = np.zeros(N)
        delay_ind = int((delay/tMax)*N)
        duration_ind = int((duration/tMax)*N)
        try:
            Iext = [ 0.0 ] * int(ALLEN_ONSET) + [ amplitude ] * int(ALLEN_STOP) + [ 0.0 ] * int(ALLEN_FINISH)
        except:        
            Iext[0:delay_ind-1] = 0.0
            Iext[delay_ind:delay_ind+duration_ind-1] = amplitude
            Iext[delay_ind+duration_ind::] = 0.0
        attrs['Iext'] = Iext
        v = get_vm(**attrs)
        self.vM = AnalogSignal(v,
                     units = mV,
                     sampling_period = attrs['dt'] * ms)
        t = [float(f) for f in self.vM.times]
        v = [float(f) for f in self.vM.magnitude]
        #print(len(v),len(t),'this is a short vector')
        try:
            fig = apl.figure()
            fig.plot(t, v, label=str('spikes: '), width=100, height=20)
            fig.show()
        except:
            pass
        self.attrs = attrs
        self.model.attrs.update(attrs)
        return self.vM

    def _backend_run(self):
        results = {}
        v = get_vm(**self.attrs)
        self.vM = AnalogSignal(v,
                               units = mV,
                               sampling_period = self.attrs['dt'] * ms)
        results['vm'] = self.vM
        results['t'] = self.vM.times
        results['run_number'] = results.get('run_number',0) + 1
        return results
