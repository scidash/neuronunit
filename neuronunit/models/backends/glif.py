#from neuronunit.tests.base import ALLEN_ONSET, DT, ALLEN_STOP, ALLEN_FINISH
import matplotlib.pyplot as plt

from quantities import mV, ms, s, V
import sciunit
from neo import AnalogSignal
import neuronunit.capabilities as cap
import numpy as np
#from neuronunit.models.backends.base import Backend
import quantities as qt
import quantities as pq

from quantities import mV, ms, s
import pickle
import copy
import re
import matplotlib as mpl
#mpl.use('Agg')

try:
   import asciiplotlib as apl
except:
   pass
import matplotlib.pyplot as plt
import allensdk.core.json_utilities as json_utilities
from allensdk.model.glif.glif_neuron import GlifNeuron
import logging

logger = logging.getLogger("allensdk")
logger.setLevel(logging.DEBUG)
del logger


logger = logging.getLogger("allensdk.model.glif.glif_neuron")
logger.setLevel(logging.DEBUG)
del logger


logger = logging.getLogger("allensdk.model")
logger.setLevel(logging.DEBUG)
del logger

logger = logging.getLogger("GlifNeuron")
logger.setLevel(logging.DEBUG)
del logger

logging.disable(logging.CRITICAL)
import numpy as np

from allensdk.api.queries.cell_types_api import CellTypesApi
#
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

from allensdk.api.queries.glif_api import GlifApi
from allensdk.core.cell_types_cache import CellTypesCache
import allensdk.core.json_utilities as json_utilities
#from allensdk.model.glif.glif_neuron import GlifNeuron
#logger = logging.getLogger("GlifNeuron")

import pickle
from .base import *
class GLIFBackend(Backend):

    name = 'GLIF'

    def init_backend(self, attrs=None,DTC=None,
                     debug = False):

        super(GLIFBackend,self).init_backend()
        self.attrs = attrs
        self.model.attrs = attrs
        if type(DTC) is not type(None):
            print('gets here')

            if type(DTC.attrs) is not type(None):
                print('gets here')

                self.set_attrs(DTC.attrs)
                #self.attrs = DTC.attrs
        self.model._backend.use_memory_cache = False
        #self.current_src_name = current_src_name
        #self.cell_name = cell_name
        self.vM = None
        #self.allen_id = None
        self.allen_id = None#566302806

        #self.attrs = attrs
        self.nc = None


        #self.debug = debug

        if self.allen_id == None:
            try:
                with open(str('allen_id.p'),'rb') as f:
                    self.nc = pickle.load(f)
            except:
                self.allen_id = 566302806
                glif_api = GlifApi()

                self.nc = glif_api.get_neuron_configs([self.allen_id])[self.allen_id]
                with open(str('allen_id.p'),'wb') as f:
                    pickle.dump(self.nc,f)
        try:
            ref_neuron_config = json_utilities.read('neuron_config.json')#['491546962']
        except:
            glif_api = GlifApi()
            glif_api.get_neuronal_models_by_id(['491546962'])

            ref_neuron_config = glif_api.get_neuron_configs(['491546962'])
            #ref_neuron_config = json_utilities.read('neuron_config.json')#['491546962']
        #print(ref_neuron_config)
        self.default_attrs = ref_neuron_config
        #self.default_attrs['dt'] = 0.01
        self.default_attrs['El_reference'] = -0.065
        self.default_attrs['init_AScurrents'] = [0.0,0.0]
        self.default_attrs['asc_tau_array'] = [0.3333333333333333,0.01]
        self.default_attrs['init_AScurrents'] = [0.0,0.0]
        self.default_attrs['threshold_reset_method'] = {'params': {'a_spike': 0.005021924962510285,'b_spike': 506.8413774098697}, 'name': 'three_components'}

        if attrs is not None:
            self.default_attrs.update(attrs)
        for k,v in ref_neuron_config.items():
            if 'method' in str(k):
                self.default_attrs[k] = v
            if 'reset_threshold_three_components' in str(k):
                self.default_attrs[k] = v

        #print(self.internal_params)
    def as_lems_model(self, backend=None):
        glif_package = []
        glif_package.append(self.metad)
        glif_package.append(self.nc)
        glif_package.append(self.get_sweeps)
        from neuronunit.models.backends import parse_glif

        lems_file_path = parse_glif.generate_lems(glif_package)
        from neuronunit.models.reduced import ReducedModel

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
        #assert  np.array(self.results['grid_spike_times'])
        return len(spike_times)

    def get_spike_count(self):
        #print('npsikes: ',len(self.results['interpolated_spike_times']))
        #self.results['interpolated_spike_times']

        return len(self.results['interpolated_spike_times'])
        #return np.array(spike_times)
    def get_membrane_potential(self):
        """Must return a neo.core.AnalogSignal.
        And must destroy the hoc vectors that comprise it.
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
        #vm = [v-0.0650 for v in vm]

        vm = [v/1000.0 for v in vm]

        vm = [v-0.0650 for v in vm]

        self.vM = AnalogSignal(vm,units = mV,sampling_period =  dt * s)
        """
        return self.vM


    def set_attrs(self,attrs):
        #print(attrs)
        #self.model.attrs.update(attrs)
        ref_neuron_config = json_utilities.read('neuron_config.json')#['491546962']
        self.default_attrs = ref_neuron_config
        #print(ref_neuron_config.keys())
        #print(ref_neuron_config)

        #self.default_attrs['dt'] = 0.01
        #self.default_attrs['El'] = attrs['El_reference']
        # print(attrs)
        self.default_attrs['init_AScurrents'] = [0.0,0.0]
        self.default_attrs['asc_tau_array'] = [0.3333333333333333,0.01]
        self.default_attrs['init_AScurrents'] = [0.0,0.0]
        self.default_attrs['threshold_reset_method'] = {'params': {'a_spike': 0.005021924962510285,'b_spike': 506.8413774098697}, 'name': 'three_components'}

        if attrs is not None:
            #print(attrs)
            self.default_attrs.update(**attrs)
        for k,v in ref_neuron_config.items():
            if 'method' in str(k):
                self.default_attrs[k] = v
            if 'reset_threshold_three_components' in str(k):
                self.default_attrs[k] = v
        #th_inf 0.029668462366002762
        #voltage_dynamics_method {'params': {}, 'name': 'linear_forward_euler'}
        #init_AScurrents [0.0, 0.0]
        #voltage_reset_method {'params': {'a': 0.04159588047528238, 'b': 0.026907648687408525}, 'name': 'v_before'}
        #threshold_dynamics_method {'params': {'b_voltage': 65.32170683581002, 'a_spike': 0.005021924962510285, 'b_spike': 506.8413774098697, 'a_voltage': 8.542046076702041}, 'name': 'three_components_exact'}
        #R_input 115085996.18771248

        #print(self.default_attrs,self.default_attrs['El'])
        self.glif = GlifNeuron.from_dict(self.default_attrs)


        if self.attrs is None:
            self.attrs = self.default_attrs
        else:
            self.attrs.update(self.default_attrs)
        assert self.attrs is not None

        #self.glif = GlifNeuron.from_dict(attrs)
        if self.attrs is not None:
            self.glif.C =  self.default_attrs['C']#,
            #self.glif.G =  self.attrs['G']#,
            self.glif.R_input =  self.default_attrs['R_input']#,
            #self.glif.init_voltage = self.default_attrs['El_reference']

    def set_stop_time(self, stop_time = 650*pq.ms):
        """Sets the simulation duration
        stopTimeMs: duration in milliseconds
        """
        self.tstop = float(stop_time.rescale(pq.ms))

    def inject_square_current(self, current):
        #if self.attrs is not None:
        self.set_attrs(self.attrs)
        if 'injected_square_current' in current.keys():
            c = current['injected_square_current']
        else:
            c = current
        tMax = (float(c['delay'])+float(c['duration'])+200.0)#/#1000.0
        delay = start = float(c['delay'])#/1000.0)
        duration = float(c['duration'])#/100.0)
        ##
        # currently in pico amps
        ##
        # make a short square pulse. stimulus units should be in Amps.

        amplitude = float(c['amplitude'])*10e-12


        #stimulus = [ 0.0 ] * 100 + [ 10e-9 ] * 100 + [ 0.0 ] * 100

        # important! set the neuron's dt value for your stimulus in seconds
        #neuron.dt = 5e-6

        #self.glif.dt = DT
        dt = float(5e-3*qt.s)
        self.glif.dt = dt

        if 'sim_length' in c.keys():
            sim_length = c['sim_length']
        #tMax = stop #delay + duration + 200.0#/dt#*pq.ms
        self.set_stop_time(tMax*pq.ms)
        tMax = self.tstop
        #attrs['dt'] = DT
        N = int(tMax/dt)
        Iext = np.zeros(N)
        delay_ind = int((delay/tMax)*N)
        duration_ind = int((duration/tMax)*N)

        Iext[0:delay_ind-1] = 0.0
        Iext[delay_ind:delay_ind+duration_ind-1] = amplitude #amplitude
        #Iext[delay_ind+N::] = 0.0
        Iext = list(Iext)
        Iext.extend([0 for i in range(0,int(N/2))])
        stimulus = Iext

        if amplitude>0:
            stimulus = [ 0.0 ] * 20 + [ amplitude ] * 200 + [ 0 ] * 40
        else:
            stimulus = [ 0.0 ] * 20 + [ amplitude ] * 100 + [ 0 ] * 40
        assert stimulus[-1] == 0

        #stimulus = [ 0.0 ] * int(delay/1000*2) + [ amplitude ] * int(duration/1000*2) + [ 0.0 ] * 200

        #self.set_attrs(self.attrs)
        #print(self.attrs)

        #print(self.glif.to_dict())
        self.results = self.glif.run(stimulus)
        vm = self.results['voltage']
        if len(self.results['interpolated_spike_voltage']) > 0:
            isv = self.results['interpolated_spike_voltage'].tolist()[0]
            self.spikes = self.results['interpolated_spike_voltage']
            #interpolated_spike_voltages = output['interpolated_spike_voltage']

            vm = list(map(lambda x: isv if np.isnan(x) else float(-0.065), vm))
            vm = [v+self.glif.init_voltage for v in vm]
            if np.max(vm)<=0:
                vm = [v+np.mean([np.abs(np.min(vm)),np.abs(np.max(vm))]) for v in vm]
                #assert np.max(vm)>0
        #vm = [v*1000.0 for v in vm]

        self.vM = AnalogSignal(vm,units =pq.V,sampling_period = float(5e-3)*pq.s)
        t = [float(f) for f in self.vM.times]
        v = [float(f) for f in self.vM.magnitude]
        fig = apl.figure()
        fig.plot(t, v, label=str('spikes: ')+str(len(self.results['grid_spike_times'])), width=100, height=20)
        fig.show()

        if len(self.results['interpolated_spike_voltage']) > 0:
            pass
            '''
            #import numpy as np
            output = self.results
            voltage = output['voltage']
            threshold = output['threshold']
            interpolated_spike_times = output['interpolated_spike_times']
            spike_times = output['interpolated_spike_times']
            interpolated_spike_voltages = output['interpolated_spike_voltage']
            interpolated_spike_thresholds = output['interpolated_spike_threshold']
            grid_spike_indices = output['spike_time_steps']
            grid_spike_times = output['grid_spike_times']
            after_spike_currents = output['AScurrents']

            # create a time array for plotting
            time = np.arange(len(stimulus))*self.glif.dt

            plt.figure(figsize=(10, 10))

            # plot stimulus
            plt.subplot(3,1,1)
            plt.plot(time, stimulus)
            plt.xlabel('time (s)')
            plt.ylabel('current (A)')
            plt.title('Stimulus')

            # plot model output
            plt.subplot(3,1,2)
            plt.plot(time,  voltage, label='voltage')
            plt.plot(time,  threshold, label='threshold')

            if grid_spike_indices is not None:
                plt.plot(interpolated_spike_times, interpolated_spike_voltages, 'x',
                        label='interpolated spike')

                plt.plot((grid_spike_indices-1)*self.glif.dt, voltage[grid_spike_indices-1], '.',
                        label='last step before spike')

            plt.xlabel('time (s)')
            plt.ylabel('voltage (V)')
            plt.legend(loc=3)
            plt.title('Model Response')

            # plot after spike currents
            plt.subplot(3,1,3)
            for ii in range(np.shape(after_spike_currents)[1]):
                plt.plot(time, after_spike_currents[:,ii])
            plt.xlabel('time (s)')
            plt.ylabel('current (A)')
            plt.title('After Spike Currents')

            plt.tight_layout()
            plt.savefig(str(self.attrs.values)+'glif_debug.png')
            '''
            #plt.show()

        #self.vM = AnalogSignal(vm,units = mV,sampling_period = (1.0/1.3)*self.glif.dt*qt.s)
        #print(np.std(self.vM),np.mean(self.vM))
        # neuronal_model_id = 566302806

        return self.vM

        '''

        except:
            pass
        try:
            fig = apl.figure()
            fig.plot(t, v, label=str('spikes: ')+str(len(self.results['grid_spike_times'])), width=100, height=20)
            fig.show()
        except:
            pass
        '''



    def do_once():
        glif_api = GlifApi()
        nm = glif_api.get_neuronal_models_by_id([neuronal_model_id])[0]

        # download the model configuration file
        nc = glif_api.get_neuron_configs([neuronal_model_id])[neuronal_model_id]
        neuron_config = glif_api.get_neuron_configs([neuronal_model_id])
        json_utilities.write('neuron_config.json', neuron_config)

        # download information about the cell
        ctc = CellTypesCache()
        #ctc.get_ephys_data(nm['specimen_id'], file_name='stimulus.nwb')
        #ctc.get_ephys_sweeps(nm['specimen_id'], file_name='ephys_sweeps.json')
        import allensdk.core.json_utilities as json_utilities
        #from allensdk.model.glif.glif_neuron import GlifNeuron
        #logger = logging.getLogger("GlifNeuron")

    # initialize the neuron
    def check_defaults(self,current):
        neuron_config = json_utilities.read('neuron_config.json')#['491546962']
        neuron = GlifNeuron.from_dict(neuron_config)

        # make a short square pulse. stimulus units should be in Amps.
        #stimulus = [ 0.0 ] * 100 + [ 10e-9 ] * 100 + [ 0.0 ] * 100
        # important! set the neuron's dt value for your stimulus in seconds
        neuron.dt = 5e-3
        # simulate the neuron
        if 'injected_square_current' in current.keys():
            c = current['injected_square_current']
        else:
            c = current
        tMax = (float(c['delay'])+float(c['duration'])+200.0)/1000.0
        delay = start = float(c['delay']/1000.0)
        duration = float(c['duration']/100.0)
        amplitude = float(c['amplitude'])#*10e-9
        #self.glif.dt = DT
        dt = 0.030
        self.glif.dt = dt

        if 'sim_length' in c.keys():
            sim_length = c['sim_length']
        #tMax = stop #delay + duration + 200.0#/dt#*pq.ms
        self.set_stop_time(tMax*pq.ms)
        tMax = self.tstop
        #attrs['dt'] = DT
        N = int(tMax/dt)
        Iext = np.zeros(N)
        delay_ind = int((delay/tMax)*N)
        duration_ind = int((duration/tMax)*N)

        Iext[0:delay_ind-1] = 0.0
        Iext[delay_ind:delay_ind+duration_ind-1] = amplitude #amplitude
        #Iext[delay_ind+N::] = 0.0
        Iext = list(Iext)
        Iext.extend([0 for i in range(0,int(N/2))])


        output = neuron.run(Iext)
        vm = output['voltage']
        threshold = output['threshold']
        spike_times = output['interpolated_spike_times']
        #vm = self.results['voltage']
        if len(output['interpolated_spike_voltage']) > 0:
            isv = output['interpolated_spike_voltage'].tolist()[0]
            spikes = output['interpolated_spike_voltage']
            vm = list(map(lambda x: isv if np.isnan(x) else x, vm))
        #vm = [v/10.0 for v in vm]
        vM = AnalogSignal(vm,units = V,sampling_period =  dt * s)
        t = [float(f) for f in vM.times]
        v = [float(f) for f in vM.magnitude]

        t = [float(f) for f in self.vM.times]
        v = [float(f) for f in self.vM.magnitude]
        #try:
        #    fig = apl.figure()
        #    fig.plot(t, v, label=str('spikes: ')+str(len(self.results['grid_spike_times'])), width=100, height=20)
        #    fig.show()
        #except:
        #    pass

        return self.vM


    def inject_square_current_allen(self, current):
        if 'injected_square_current' in current.keys():
            c = current['injected_square_current']
        else:
            c = current
        stop = float(c['delay'])+float(c['duration'])
        start = float(c['delay'])
        duration = float(c['duration'])
        amplitude = float(c['amplitude'])#/100 000 000 000.0


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

        self.vM = AnalogSignal(vm,units = mV,sampling_period =  dt * s)
        t = [float(f) for f in self.vM.times]
        v = [float(f) for f in self.vM.magnitude]
        fig = apl.figure()
        fig.plot([float(t)*1000.0 for t in vm_used.times],[float(v) for v in vm_used_mag],label=str(dtc.attrs), width=100, height=20)
        fig.show()

        return self.vM


    def _backend_run(self):
        results = None
        results = {}
        results['vm'] = self.vM
        results['t'] = self.vM.times
        results['run_number'] = results.get('run_number',0) + 1
        return results
