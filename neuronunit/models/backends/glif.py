#from neuronunit.tests.base import ALLEN_ONSET, DT, ALLEN_STOP, ALLEN_FINISH
from quantities import mV, ms, s, V
import sciunit
from neo import AnalogSignal
import neuronunit.capabilities as cap
import numpy as np
from neuronunit.models.backends import parse_glif
from neuronunit.models.backends.base import Backend
import quantities as qt
import quantities as pq

from quantities import mV, ms, s
import pickle
import copy
import re
import matplotlib as mpl
#mpl.use('Agg')

#try:
import asciiplotlib as apl
#except:
#    pass
import matplotlib.pyplot as plt
import allensdk.core.json_utilities as json_utilities
from allensdk.model.glif.glif_neuron import GlifNeuron
from allensdk.api.queries.cell_types_api import CellTypesApi
# from neuronunit.models.reduced import ReducedModel

try:
    from allensdk.api.queries.glif_api import GlifApi
    from allensdk.core.cell_types_cache import CellTypesCache
    import allensdk.core.json_utilities as json_utilities
    import sciunit
except:
    import os
    os.system('pip install allensdk')
    from allensdk.api.queries.glif_api import GlifApi
    from allensdk.core.cell_types_cache import CellTypesCache
    import allensdk.core.json_utilities as json_utilities

    #os.system('pip install git+https://github.com/scidash/sciunit@dev')

#ls = ls1[0]
#DT = sampling_period = 1.0/ls['sampling_rate']*pq.s
#on_indexs = np.where(ls==np.max(ls))

#ALLEN_STIM = ls
#ALLEN_ONSET = start = np.min(on_indexs)*1.0/ls['sampling_rate']*pq.s
#ALLEN_STOP = stop = np.max(on_indexs)*1.0/ls['sampling_rate']*pq.s
#ALLEN_FINISH = len(ls)*(1.0/ls['sampling_rate'])*pq.s


class GLIFBackend(Backend):
    def init_backend(self, attrs = None, cell_name = 'alice', current_src_name = 'hannah', DTC = None, debug = False):
        backend = 'GLIF'
        super(GLIFBackend,self).init_backend()

        self.model._backend.use_memory_cache = False
        self.current_src_name = current_src_name
        self.cell_name = cell_name
        self.vM = None
        #self.allen_id = None
        self.allen_id = 566302806

        self.attrs = attrs
        self.nc = None

        self.temp_attrs = None
        self.debug = debug

        if self.allen_id == None:
            try:
                self.nc = pickle.load(open(str('allen_id.p'),'rb'))
            except:
                self.allen_id = 566302806
                glif_api = GlifApi()

                self.nc = glif_api.get_neuron_configs([self.allen_id])[self.allen_id]
                pickle.dump(copy.copy(self.nc),open(str('allen_id.p'),'wb'))


        else:

            try:
                self.nc = pickle.load(open(str('allen_id.p'),'rb'))
            except:
                glif_api = GlifApi()
                #allen_id =

                self.allen_id = 566302806
                self.glif = glif_api.get_neuronal_models_by_id([self.allen_id])[0]
                #import pdb
                #pdb.set_trace()
                self.nc = glif_api.get_neuron_configs([self.allen_id])[self.allen_id]
                pickle.dump(self.nc,open(str('allen_id.p'),'wb'))


        try:
            self.glif = GlifNeuron.from_dict(self.nc)
        except:
            #import pdb; pdb.set_trace()
            self.nc['init_AScurrents'] = [0,0]
            self.glif = GlifNeuron.from_dict(self.nc)

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

        #print(self.internal_params)
    def as_lems_model(self, backend=None):
        glif_package = []
        glif_package.append(self.metad)
        glif_package.append(self.nc)
        glif_package.append(self.get_sweeps)
        lems_file_path = parse_glif.generate_lems(glif_package)
        return ReducedModel(lems_file_path, backend=backend)

    def get_sweeps(self,specimen_id = None):
        if specimen_id == None:
            self.sweeps = ctc.get_ephys_sweeps(self.glif[self.allen_id], \
            file_name='%d_ephys_sweeps.json' % self.allen_id)

    def get_sweep(self, n,specimen_id = None):
        if specimen_id == None:
            self.sweeps = ctc.get_ephys_sweeps(self.glif[self.allen_id], \
            file_name='%d_ephys_sweeps.json' % self.allen_id)
        sweep_info = self.sweeps[n]
        sweep_number = sweep_info['sweep_number']
        sweep = ds.get_sweep(sweep_number)
        return sweep



    def get_stimulus(self, n):
        sweep = self.get_sweep(n)
        return sweep['stimulus']

    def apply_stimulus(self, n):
        self.stimulus = self.get_stimulus(n)

    def get_spike_train(self):
        spike_times = self.results['interpolated_spike_times']

        ##print(self.results['interpolated_spike_times'])
        return np.array(self.results['grid_spike_times'])
    def get_spike_count(self):
        #print('npsikes: ',len(self.results['interpolated_spike_times']))
        self.results['interpolated_spike_times']
        return len(self.results['grid_spike_times'])
        #return np.array(spike_times)
    def get_membrane_potential(self):
        """Must return a neo.core.AnalogSignal.
        And must destroy the hoc vectors that comprise it.
        """
        threshold = self.results['threshold']
        interpolated_spike_times = self.results['interpolated_spike_times']

        interpolated_spike_thresholds = self.results['interpolated_spike_threshold']
        grid_spike_indices = self.results['spike_time_steps']
        grid_spike_times = self.results['grid_spike_times']
        after_spike_currents = self.results['AScurrents']

        vm = self.results['voltage']
        if len(grid_spike_times) > 0:
            #print(len(self.results['interpolated_spike_voltage']), 'yes')
            isv = self.results['interpolated_spike_voltage'].tolist()[0]
            vm = list(map(lambda x: isv if np.isnan(x) else x, vm))
        dt =  self.glif.dt
        vm = [v*10.0 for v in vm]
        self.vM = AnalogSignal(vm,units = mV,sampling_period =  dt * ms)
        return self.vM


    def set_attrs(self, **attrs):
        self.model.attrs.update(attrs)
        #self.nc.update(attrs)
        for k,v in attrs.items():
            self.nc[k] = v
        #self.nc['init_AScurrents'] = [0,0]
        self.glif = GlifNeuron.from_dict(self.nc)
        #print(self.nc)
        self.glif.init_voltage = -0.065

        return self.glif


    def set_stop_time(self, stop_time = 650*pq.ms):
        """Sets the simulation duration
        stopTimeMs: duration in milliseconds
        """
        self.tstop = float(stop_time.rescale(pq.ms))

    def inject_square_current(self, current):
        if 'injected_square_current' in current.keys():
            c = current['injected_square_current']
        else:
            c = current
        stop = float(c['delay'])+float(c['duration'])+200.0
        start = float(c['delay'])
        duration = float(c['duration'])
        amplitude = float(c['amplitude'])/100000000000.0
        #self.glif.dt = DT
        dt = 0.030
        self.glif.dt = dt

        #amplitude = float(c['amplitude'])
        duration = float(c['duration'])#/dt#/dt.rescale('ms')
        delay = float(c['delay'])#/dt#.rescale('ms')
        if 'sim_length' in c.keys():
            sim_length = c['sim_length']
        tMax = delay + duration + 200.0#/dt#*pq.ms
        self.set_stop_time(tMax*pq.ms)
        tMax = self.tstop
        #attrs['dt'] = DT
        N = int(tMax/dt)
        Iext = np.zeros(N)
        delay_ind = int((delay/tMax)*N)
        duration_ind = int((duration/tMax)*N)

        Iext[0:delay_ind-1] = 0.0
        Iext[delay_ind:delay_ind+duration_ind-1] = amplitude
        Iext[delay_ind+duration_ind::] = 0.0


        #stim = [ 0.0 ] * int(start) + [ amplitude ] * int(stop) + [ 0.0 ] * int(duration)
        #print(np.max(stim),'max current')
        self.results = self.glif.run(Iext)
        #import pdb; pdb.set_trace()
        vm = self.results['voltage']
        if len(self.results['interpolated_spike_voltage']) > 0:
            #print('npsikes: ',len(self.results['interpolated_spike_times']), 'called by rheobase?')
            isv = self.results['interpolated_spike_voltage'].tolist()[0]
            self.spikes = self.results['interpolated_spike_voltage']
            vm = list(map(lambda x: isv if np.isnan(x) else x, vm))
        vm = [v*10.0 for v in vm]
        self.vM = AnalogSignal(vm,units = V,sampling_period =  dt * s)
        t = [float(f) for f in self.vM.times]
        v = [float(f) for f in self.vM.magnitude]
        #try:
        #plt.clf()
        #plt.plot(t,v)
        #plt.show()
        try:
            fig = apl.figure()
            fig.plot(t, v, label=str('spikes: ')+str(len(self.results['grid_spike_times'])), width=100, height=20)
            fig.show()
        except:
            pass
        return self.vM


    def inject_square_current_allen(self, current):
        if 'injected_square_current' in current.keys():
            c = current['injected_square_current']
        else:
            c = current
        stop = float(c['delay'])+float(c['duration'])
        start = float(c['delay'])
        duration = float(c['duration'])
        amplitude = float(c['amplitude'])/100000000000.0
        import pickle

        ls1 = pickle.load(open('../models/backends/generic_current_injection.p','rb'))
        ls = ls1[0]['stimulus']
        DT = sampling_period = 1.0/ls1[0]['sampling_rate']#*pq.s
        on_indexs = np.where(ls==np.max(ls))

        ALLEN_STIM = ls
        ALLEN_ONSET = start = np.min(on_indexs)*DT
        ALLEN_STOP = stop = np.max(on_indexs)*DT
        ALLEN_FINISH = len(ls)*DT

        ls = ls1[0]['stimulus']


        old_max = np.max(ls)
        on_indexs = np.where(ls==np.max(ls))

        ls[on_indexs] = amplitude

        assert np.max(ls)!= old_max

        self.stim = ls
        self.glif.dt = sampling_period
        dt =  self.glif.dt
        #print(np.max(self.stim),'max current')
        self.results = self.glif.run(self.stim)
        vm = self.results['voltage']
        if len(self.results['interpolated_spike_voltage']) > 0:
            #print('npsikes: ',len(self.results['interpolated_spike_times']), 'called by rheobase?')
            isv = self.results['interpolated_spike_voltage'].tolist()[0]
            self.spikes = self.results['interpolated_spike_voltage']
            vm = list(map(lambda x: isv if np.isnan(x) else x, vm))

        self.vM = AnalogSignal(vm,units = V,sampling_period =  dt * s)
        t = [float(f) for f in self.vM.times]
        v = [float(f) for f in self.vM.magnitude]
        #try:
        #plt.clf()
        #plt.plot(t,v)
        #plt.show()
        try:
            fig = apl.figure()
            fig.plot(t, v, label=str('spikes: ')+str(len(self.results['grid_spike_times'])), width=100, height=20)
            fig.show()
        except:
            pass
        return self.vM


    def _backend_run(self):
        results = None
        results = {}
        results['vm'] = self.vM
        results['t'] = self.vM.times
        results['run_number'] = results.get('run_number',0) + 1
        return results
