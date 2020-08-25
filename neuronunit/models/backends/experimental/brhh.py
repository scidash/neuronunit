from __future__ import print_function
from brian2 import *

try:
    import julia
    j = julia.Julia()
    print('no')
    j.eval('using Pkg; Pkg.add("https://github.com/russelljjarvis/SpikingNeuralNetworks.jl")')
    x1 = j.include("hh_neuron.jl")
    print(x1)
    x2 = j.include("hh_net.jl")

except:
    print('raises exception')
import brian2 as b2
from neo import AnalogSignal
import neuronunit.capabilities.spike_functions as sf
import neuronunit.capabilities as cap
cap.ReceivesCurrent
cap.ProducesActionPotentials
from types import MethodType



from brian2.units.constants import (zero_celsius, faraday_constant as F,
    gas_constant as R)


try:
    import asciiplotlib as apl
    fig = apl.figure()
    fig.plot([0,1], [0,1], label=str('spikes: ')+str(self.n_spikes), width=100, height=20)
    fig.show()
    ascii_plot = True
except:
    ascii_plot = False
import numpy

from scipy.interpolate import interp1d


def HHbrian(Backend):
    def get_spike_count(self):
        return int(self.spike_monitor.count[0])
    def init_backend(self, attrs=None, cell_name='thembi',
                     current_src_name='spanner', DTC=None,
                     debug = False):
        backend = 'adexp'
        super(ADEXPBackend,self).init_backend()
        self.name = str(backend)

        #self.threshold = -20.0*qt.mV
        self.debug = None
        self.model._backend.use_memory_cache = False
        self.current_src_name = current_src_name
        self.cell_name = cell_name
        self.vM = None
        self.attrs = attrs
        self.debug = debug
        self.temp_attrs = None
        self.n_spikes = None
        self.spike_monitor = None
        self.peak_v = 0.02
        self.verbose = False
        self.model.get_spike_count = self.get_spike_count

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
        model.init_backend = MethodType(init_backend,self.model)
        model.get_spike_count = MethodType(get_spike_count,self.model)
        model.get_APs = MethodType(get_APs,model)
        model.get_spike_train = MethodType(get_spike_train,model)
        model.set_attrs = MethodType(set_attrs, model) # Bind to the score.
        model.inject_square_current = MethodType(inject_square_current, model) # Bind to the score.
        model.set_attrs = MethodType(set_attrs, model) # Bind to the score.
        model.get_membrane_potential = MethodType(get_membrane_potential,model)
        model.load_model = MethodType(load_model, model) # Bind to the score.
        model._local_run = MethodType(_local_run,model)
        model.init_backend(model)

        return model



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
    def _local_run(self):
        '''
        pyNN lazy array demands a minimum population size of 3. Why is that.
        '''
        results = {}
        DURATION = 1000.0

        #ctx_cells.celltype.recordable
        M = StateMonitor(neuron, 'v', record=True)
        spikes = SpikeMonitor(neuron)

        #run(50*ms, report='text')
        neuron.I[0] = 1*uA # current injection at one end
        #run(3*ms)


            #self.neuron.record_gsyn(self.hhcell, "Results/HH_cond_exp_%s.gsyn" % str(neuron))
        self.neuron.run(DURATION*ms)
        data = self.hhcell.get_data().segments[0]
        volts = data.filter(name="v")[0]#/10.0
        vm = self.M.v/mV
        vm = AnalogSignal(volts,
                     units = mV,
                     sampling_period = self.dt *ms)
        results['vm'] = vm
        results['t'] = self.M.t/ms # self.times
        results['run_number'] = results.get('run_number',0) + 1
        return results




    def set_attrs(self,**attrs):
        '''
        example params:
            neuron.v = 0*mV
            neuron.h = 1
            neuron.m = 0
            neuron.n = .5
            neuron.I = 0*amp
            neuron.gNa = gNa0
        '''
        self.init_backend()
        self.model.attrs.update(attrs)
        for k, v in attrs.items():
            exec('self.neuron.'+str(k)+'='+str(v))
        assert type(self.model.attrs) is not type(None)
        return self


    def inject_square_current(self,current):
        attrs = copy.copy(self.model.attrs)
        self.init_backend()
        self.set_attrs(**attrs)
        c = copy.copy(current)
        if 'injected_square_current' in c.keys():
            c = current['injected_square_current']

        stop = float(c['delay'])+float(c['duration'])
        duration = float(c['duration'])
        start = float(c['delay'])
        amplitude = float(c['amplitude'])*1000.0 #convert pico amp to micro Amp
        self.neuron.I[0] = amplitude*uA # current injection at one end

        run(stop*ms)

        self.vm = self.results['vm']




    def get_APs(self,vm):
        #

        vm = self.get_membrane_potential()
        waveforms = sf.get_spike_waveforms(vm,threshold=-45.0*mV)
        return waveforms

    def get_spike_train(self,**run_params):

        vm = self.get_membrane_potential()

        spike_train = threshold_detection(vm,threshold=-45.0*mV)

        return spike_train# self.M(self.neuron)

    def get_spike_count(self,**run_params):
        vm = self.get_membrane_potential()
        return len(threshold_detection(vm,threshold=-45.0*mV))
    #model.add_attribute(init_backend)
