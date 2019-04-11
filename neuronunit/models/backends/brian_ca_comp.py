import copy
import pdb
import numpy as np
from .base import *
import quantities as qt
from quantities import mV, ms, s
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')
import brian2 as b2
from neurodynex.adex_model import AdEx
from neurodynex.tools import plot_tools, input_factory

from brian2 import NeuronGroup, StateMonitor, SpikeMonitor, morpho, neuron

# Parameters
C = 281 * pF
gL = 30 * nS
taum = C / gL
EL = -70.6 * mV
VT = -50.4 * mV
DeltaT = 2 * mV
Vcut = VT + 5 * DeltaT

# Pick an electrophysiological behaviour
tauw, a, b, Vr = 144*ms, 4*nS, 0.0805*nA, -70.6*mV # Regular spiking (as in the paper)
#tauw,a,b,Vr=20*ms,4*nS,0.5*nA,VT+5*mV # Bursting
#tauw,a,b,Vr=144*ms,2*C/(144*ms),0*nA,-70.6*mV # Fast spiking

['ADAPTATION_TIME_CONSTANT_tau_w',
'ADAPTATION_VOLTAGE_COUPLING_a',
'FIRING_THRESHOLD_v_spike',
'MEMBRANE_RESISTANCE_R',
'MEMBRANE_TIME_SCALE_tau_m',
'RHEOBASE_THRESHOLD_v_rh',
'SHARPNESS_delta_T',
'SPIKE_TRIGGERED_ADAPTATION_INCREMENT_b',
'V_RESET']

'''

eqs = """
dvm/dt = (gL*(EL - vm) + gL*DeltaT*exp((vm - VT)/DeltaT) + I - w)/C : volt
dw/dt = (a*(vm - EL) - w)/tauw : amp
I : amp
"""

neuron = NeuronGroup(1, model=eqs, threshold='vm>Vcut',
                     reset="vm=Vr; w+=b", method='euler')
neuron.vm = EL
trace = StateMonitor(neuron, 'vm', record=0)
spikes = SpikeMonitor(neuron)

run(20 * ms)
neuron.I = 1*nA
run(100 * ms)
neuron.I = 0*nA
run(20 * ms)

# We draw nicer spikes
vm = trace[0].vm[:]
for t in spikes.t:
    i = int(t / defaultclock.dt)
    vm[i] = 20*mV

plot(trace.t / ms, vm / mV)
xlabel('time (ms)')
ylabel('membrane potential (mV)')
show()
'''

class BRIANCAMULTIBackend(Backend):


    def init_backend(self, attrs = None, cell_name= 'camulti', current_src_name = 'hannah', DTC = None, dt=0.01, cell_type=None):
        backend = 'BRIAN2'
        super(BRIANCAMULTIBackend,self).init_backend()
        self.current_src_name = current_src_name
        self.cell_name = cell_name
        self.dt = dt
        self.setup = setup
        self.model_path = None
        self.related_data = {}
        self.lookup = {}
        self.attrs = {}
        #self.neuron = neuron
        self.model._backend.use_memory_cache = False
        #self.model.unpicklable += ['h','ns','_backend']
        #neuron.setup(timestep=dt, min_delay=1.0)

        if type(DTC) is not type(None):
            if type(DTC.attrs) is not type(None):

                self.set_attrs(**DTC.attrs)
                assert len(self.model.attrs.keys()) > 0

            if hasattr(DTC,'current_src_name'):
                self._current_src_name = DTC.current_src_name

            if hasattr(DTC,'cell_name'):
                self.cell_name = DTC.cell_name

            def __init__(self):

        defaultclock.dt = 0.01*ms


        VT = -52*mV
        El = -76.5*mV  # from code, text says: -69.85*mV

        E_Na = 50*mV
        E_K = -100*mV
        C_d = 7.954  # dendritic correction factor

        T = 34*kelvin + zero_celsius # 34 degC (current-clamp experiments)
        tadj_HH = 3.0**((34-36)/10.0)  # temperature adjustment for Na & K (original recordings at 36 degC)
        tadj_m_T = 2.5**((34-24)/10.0)
        tadj_h_T = 2.5**((34-24)/10.0)

        shift_I_T = -1*mV

        gamma = F/(R*T)  # R=gas constant, F=Faraday constant
        Z_Ca = 2  # Valence of Calcium ions
        Ca_i = 240*nM  # intracellular Calcium concentration
        Ca_o = 2*mM  # extracellular Calcium concentration

        eqs = Equations('''
        Im = gl*(El-v) - I_Na - I_K - I_T: amp/meter**2
        I_inj : amp (point current)
        gl : siemens/meter**2
        # HH-type currents for spike initiation
        g_Na : siemens/meter**2
        g_K : siemens/meter**2
        I_Na = g_Na * m**3 * h * (v-E_Na) : amp/meter**2
        I_K = g_K * n**4 * (v-E_K) : amp/meter**2
        v2 = v - VT : volt  # shifted membrane potential (Traub convention)
        dm/dt = (0.32*(mV**-1)*(13.*mV-v2)/
                (exp((13.*mV-v2)/(4.*mV))-1.)*(1-m)-0.28*(mV**-1)*(v2-40.*mV)/
                (exp((v2-40.*mV)/(5.*mV))-1.)*m) / ms * tadj_HH: 1
        dn/dt = (0.032*(mV**-1)*(15.*mV-v2)/
                (exp((15.*mV-v2)/(5.*mV))-1.)*(1.-n)-.5*exp((10.*mV-v2)/(40.*mV))*n) / ms * tadj_HH: 1
        dh/dt = (0.128*exp((17.*mV-v2)/(18.*mV))*(1.-h)-4./(1+exp((40.*mV-v2)/(5.*mV)))*h) / ms * tadj_HH: 1
        # Low-threshold Calcium current (I_T)  -- nonlinear function of voltage
        I_T = P_Ca * m_T**2*h_T * G_Ca : amp/meter**2
        P_Ca : meter/second  # maximum Permeability to Calcium
        G_Ca = Z_Ca**2*F*v*gamma*(Ca_i - Ca_o*exp(-Z_Ca*gamma*v))/(1 - exp(-Z_Ca*gamma*v)) : coulomb/meter**3
        dm_T/dt = -(m_T - m_T_inf)/tau_m_T : 1
        dh_T/dt = -(h_T - h_T_inf)/tau_h_T : 1
        m_T_inf = 1/(1 + exp(-(v/mV + 56)/6.2)) : 1
        h_T_inf = 1/(1 + exp((v/mV + 80)/4)) : 1
        tau_m_T = (0.612 + 1.0/(exp(-(v/mV + 131)/16.7) + exp((v/mV + 15.8)/18.2))) * ms / tadj_m_T: second
        tau_h_T = (int(v<-81*mV) * exp((v/mV + 466)/66.6) +
                   int(v>=-81*mV) * (28 + exp(-(v/mV + 21)/10.5))) * ms / tadj_h_T: second
        ''')
        model = SpatialNeuron(morphology=morpho, model=eqs, method="exponential_euler",
                               refractory="m > 0.4", threshold="m > 0.5",
                               Cm=1*uF/cm**2, Ri=35.4*ohm*cm)
        self.model = model




    def get_membrane_potential(self):
        """Must return a neo.core.AnalogSignal.
        And must destroy the hoc vectors that comprise it.
        """

        #data = self.state_monitor
        #volts = data.filter(name="v")[0]

        vm = AnalogSignal(self.vm,
             units = mV,
             sampling_period = self.dt *ms)

        return vm

    def _local_run(self):
        '''
        '''
        if self.M.v is None:
            run(self.tstop*ms)#, report='text')
        else:
            self.vm = AnalogSignal(copy.copy(self.M.v),units = mV,sampling_period = self.dt *s)
        results = {}
        results['vm'] = self.vm
        results['t'] = self.vm.times
        results['run_number'] = results.get('run_number',0) + 1
        if self.debug == True:
            plt.clf()
            plt.plot(vm.times,vm)
            plt.savefig('debug_ad_exp.png')
        return results

    def load_model(self):
        neuron.setup(timestep=0.01, min_delay=1.0)
        #self.eif = neuron.Population(1,EIF_cond_exp_isfa_ista())

    def set_attrs(self, **attrs):
        self.model.attrs.update(attrs)
        assert type(self.model.attrs) is not type(None)
        self.neuron = SpatialNeuron(morphology=morpho, model=eqs, method="exponential_euler",
                               refractory="m > 0.4", threshold="m > 0.5",
                               Cm=1*uF/cm**2, Ri=35.4*ohm*cm)
        self.neuron.v = 0*mV
        self.neuron.h = 1
        self.neuron.m = 0
        self.neuron.n = .5
        self.neuron.I = 0*amp
        self.neuron.gNa = gNa0
        self.M = StateMonitor(neuron, 'v', record=True)
        spikes = SpikeMonitor(neuron)

        return selfl


    def set_stop_time(self, stop_time = 650*pq.ms):
        """Sets the simulation duration
        stopTimeMs: duration in milliseconds
        """
        self.tstop = float(stop_time.rescale(pq.ms))


    def inject_square_current(self, current):

        #plot_tools.plot_voltage_and_current_traces(state_monitor, current)
        #print("nr of spikes: {}".format(spike_monitor.count[0]))


        self.set_attrs(**attrs)
        c = copy.copy(current)
        if 'injected_square_current' in c.keys():
            c = current['injected_square_current']

        stop = float(c['delay'])+float(c['duration'])
        duration = float(c['duration'])
        start = float(c['delay'])
        amplitude = float(c['amplitude']/1000.0)#*1000.0#*10000.0
        #current = input_factory.get_step_current(start, duration, 1. * b2.ms, amplitude * b2.pA)
        #state_monitor, spike_monitor = AdEx.simulate_AdEx_neuron(I_stim=current, simulation_time=stop * b2.ms)
        #run(50*ms, report='text')
        neuron.I[0] = 0*uA # current injection at one end

        run(c['delay'])
        neuron.I[0] = amplitude*uA # current injection at one end
        run(c['duration'])
        neuron.I[0] = 0*uA # current injection at one end
        run(stop*ms)

        #neuron.amplitude = 0*amp
        #self.results = self._local_run()
        self.vm = self.results['vm']
