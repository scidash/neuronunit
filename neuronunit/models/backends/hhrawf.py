

from scipy.integrate import odeint
from numba import jit
import numpy as np

# Set random seed (for reproducibility)
#np.random.seed(1000)

import io
import math
import pdb
from numba import jit
from .base import *
import quantities as qt
from quantities import mV, ms, s
from sciunit.utils import redirect_stdout
from elephant.spike_train_generation import threshold_detection
'''
try:
    import asciiplotlib as apl
    fig = apl.figure()
    fig.plot([1,0], [0,1])
    ascii_plot = True
    import gc

except:
    ascii_plot = False
SLOW_ZOOM = True
'''
ascii_plot = False

@jit
def Id(t,delay,duration,tmax,amplitude):
    #print(delay+duration,tmax)

    if 0.0 < t < delay:
        return 0.0
    elif delay < t < delay+duration:
        return amplitude#(100.0)
    elif delay+duration < t < tmax:
        return 0.0
    return 0.0

# Compute derivatives
# Average potassium channel conductance per unit area (mS/cm^2)
# Average sodoum channel conductance per unit area (mS/cm^2)
# Average leak channel conductance per unit area (mS/cm^2)
# Membrane capacitance per unit area (uF/cm^2)
# Potassium potential (mV)
# Sodium potential (mV)
# Leak potential (mV)

@jit
def alpha_m(V):
    """Channel gating kinetics. Functions of membrane voltage"""
    return 0.1*(V+40.0)/(1.0 - np.exp(-(V+40.0) / 10.0))
@jit
def beta_m(V):
    """Channel gating kinetics. Functions of membrane voltage"""
    return 4.0*np.exp(-(V+65.0) / 18.0)
@jit
def alpha_h(V):
    """Channel gating kinetics. Functions of membrane voltage"""
    return 0.07*np.exp(-(V+65.0) / 20.0)
@jit
def beta_h(V):
    """Channel gating kinetics. Functions of membrane voltage"""
    return 1.0/(1.0 + np.exp(-(V+35.0) / 10.0))
@jit
def alpha_n(V):
    """Channel gating kinetics. Functions of membrane voltage"""
    return 0.01*(V+55.0)/(1.0 - np.exp(-(V+55.0) / 10.0))
@jit
def beta_n(V):
    """Channel gating kinetics. Functions of membrane voltage"""
    return 0.125*np.exp(-(V+65) / 80.0)


@jit
def dALLdt(X, t, attrs):
    """
    Integrate

    |  :param X:
    |  :param t:
    |  :return: calculate membrane potential & activation variables
    """
    defaults = { 'g_K' : 36.0, 'g_Na' : 120.0, 'g_L' : 0.3, \
             'C_m' : 1.0, 'E_L' : -54.387, 'E_K' : -77.0, 'E_Na' : 50.0, 'vr':-65.0 }
    #print("defaults",defaults)
    #print("actual:",attrs)
    delay,duration,T,amplitude = copy.copy(attrs['I'])
    #for k,v in attrs.items():
    #    if k in defaults.keys():
    #        pass
    #    pass
            #print('difference in k: ',k,': ',defaults[k]-v)
            #attrs[k] = defaults[k]
    #    return
    #attrs = defaults
    V, m, h, n = X

    C_m = attrs['C_m']
    E_L = attrs['E_L']
    E_K = attrs['E_K']
    E_Na = attrs['E_Na']
        # dVm/dt
    g_K = attrs['g_K'] #/ Cm) * np.power(n, 4.0)
    g_Na = attrs['g_Na'] #/ Cm) * np.power(m, 3.0) * h
    g_L = attrs['g_L'] #/ Cm
    delay,duration,T,amplitude = attrs['I']
    #Id(t,delay,duration,tmax,amplitude)
    Iext = Id(t,delay,duration,np.max(T),amplitude)
    #print(Iext)


    I_Na = g_Na * m**3.0 * h * (V - E_Na)
    I_K = g_K  * n**4.0 * (V - E_K)
    #  Leak
    I_L = g_L * (V - E_L)

    dVdt = (Iext - I_Na - I_K - I_L) / C_m
    dmdt = alpha_m(V)*(1.0-m) - beta_m(V)*m
    dhdt = alpha_h(V)*(1.0-h) - beta_h(V)*h
    dndt = alpha_n(V)*(1.0-n) - beta_n(V)*n
    return dVdt, dmdt, dhdt, dndt


def get_vm(attrs):
    '''
    dt determined by
    Apply Hodgkin Huxley equation corresponding to point as model
    This function can't get too pythonic (functional), it needs to be a simple loop for
    numba/jit to understand it.
    '''
    # State (Vm, n, m, h)
    # saturation value
    #vr = attrs['vr']
    m = 0.05#*1000.0
    h = 0.60#*1000.0
    n = 0.32#*1000.0

    Y = [attrs['vr'], m, h, n]
    # Solve ODE system
    T = attrs['T']
    dt = attrs['dt']
    Vy = odeint(dALLdt, Y, T, args=(attrs,))

    volts = [ v[0] for v in Vy ]
    vm = AnalogSignal(volts,
                 units = mV,
                 sampling_period = dt * ms)

    if ascii_plot:
        t = [float(f) for f in vm.times]
        v = [float(f) for f in vm.magnitude]
        fig = apl.figure()
        fig.plot(t, v, label=str('hhraw: ')+str(vm.units), width=100, height=20)
        fig.show()
        gc.collect()
        fig = None

    fig  = None
    return vm


class HHBackend(Backend):

    def init_backend(self, attrs = None, cell_name = 'alice', current_src_name = 'hannah', DTC = None):
        backend = 'HH'
        super(HHBackend,self).init_backend()
        self.model._backend.use_memory_cache = False
        self.current_src_name = current_src_name
        self.cell_name = cell_name
        self.vM = None
        self.attrs = attrs

        self.temp_attrs = None
        self.name = str(backend)

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


    def get_membrane_potential(self):
        """Must return a neo.core.AnalogSignal.
        And must destroy the hoc vectors that comprise it.
        """
        return self.vM

    def get_spike_count(self):
        thresh = threshold_detection(self.vM,0.0*pq.mV)
        print(len(thresh),' spikes')
        return len(thresh)

    def set_attrs(self, **attrs):
        self.model.attrs.update(attrs)

    def _backend_run(self):
        results = {}
        results['vm'] = self.vM
        results['t'] = self.vM.times
        results['run_number'] = results.get('run_number',0) + 1
        return results


    def inject_square_current(self, current):#, section = None, debug=False):
        """Inputs: current : a dictionary with exactly three items, whose keys are: 'amplitude', 'delay', 'duration'
        Example: current = {'amplitude':float*pq.pA, 'delay':float*pq.ms, 'duration':float*pq.ms}}
        where \'pq\' is a physical unit representation, implemented by casting float values to the quanitities \'type\'.
        Description: A parameterized means of applying current injection into defined
        Currently only single section neuronal models are supported, the neurite section is understood to be simply the soma.

        """

        if 'injected_square_current' in current.keys():
            c = current['injected_square_current']
        else:
            c = current
        amplitude = float(c['amplitude'])#s.simplified*1000000000000000.0)
        duration = float(c['duration'])#/dt#/dt.rescale('ms')
        delay = float(c['delay'])#/dt#.rescale('ms')
        tmax = delay + duration + 200.0#/dt#*pq.ms
        self.set_stop_time(tmax*pq.ms)
        tmax = self.tstop
        tmin = 0.0
        DT = 0.01
        T = np.linspace(tmin, tmax, int(tmax/DT))
        dt = T[1]-T[0]

        attrs = copy.copy(self.model.attrs)
        attrs['I'] = (delay,duration,tmax,amplitude)
        #Iext = Id(t,delay,duration,T,amplitude)

        #Iext_ = [Id(t,delay,duration,np.max(T),amplitude) for t in T]
        #import pdb
        #pdb.set_trace()
        attrs['dt'] = dt
        attrs['T'] = T

        self.vM  = get_vm(attrs)
        return self.vM
