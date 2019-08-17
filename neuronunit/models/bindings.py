import io
import math
import pdb
import copy
from types import MethodType


import numpy as np
import quantities as pq
import matplotlib.pyplot as plt

from elephant.spike_train_generation import threshold_detection
from neo import AnalogSignal
try:
    from pyNN.neuron import HH_cond_exp
    from pyNN.neuron import EIF_cond_exp_isfa_ista
    import pyNN.neuron as pn
    from pyNN.neuron import setup as setup
    from pyNN.neuron import DCSource
    pyNN_NEURON = True
except (ImportError, AttributeError):
    print("Error loading pyNN.neuron")
    pyNN_NEURON = False

from sciunit.utils import redirect_stdout
import neuronunit.capabilities.spike_functions as sf
<<<<<<< HEAD
from types import MethodType

=======
>>>>>>> 51529ae8e9a02874e8b1d050bb812f2aec8d41d9
import neuronunit.capabilities as cap


def bind_NU_interface(model):

    def load_model(self):
        neuron = None
        from pyNN import neuron
        self.hhcell = neuron.create(EIF_cond_exp_isfa_ista())
        pn.setup(timestep=self.dt, min_delay=1.0)


    def init_backend(self, attrs = None, cell_name= 'HH_cond_exp', current_src_name = 'hannah', DTC = None, dt=0.01):
        backend = 'HHpyNN'
        self.current_src_name = current_src_name
        self.cell_name = cell_name
        self.adexp = True

        self.DCSource = DCSource
        self.setup = setup
        #self.model_path = None
        self.related_data = {}
        self.lookup = {}
        self.attrs = {}
        self.neuron = pn
        self.model._backend = self
        self.backend = self
        self.model.attrs = {}

        #self.orig_lems_file_path = 'satisfying'
        self.model._backend.use_memory_cache = False
        #self.model.unpicklable += ['h','ns','_backend']
        self.dt = dt
        if type(DTC) is not type(None):
            if type(DTC.attrs) is not type(None):

                self.set_attrs(**DTC.attrs)
                assert len(self.model.attrs.keys()) > 0

            if hasattr(DTC,'current_src_name'):
                self._current_src_name = DTC.current_src_name

            if hasattr(DTC,'cell_name'):
                self.cell_name = DTC.cell_name



    def _local_run(self):
        '''
        pyNN lazy array demands a minimum population size of 3. Why is that.
        '''
        results = {}
        DURATION = 1000.0




        if self.celltype == 'HH_cond_exp':

            self.hhcell.record('spikes','v')

        else:
            self.neuron.record_v(self.hhcell, "Results/HH_cond_exp_%s.v" % str(pn))

            #self.neuron.record_gsyn(self.hhcell, "Results/HH_cond_exp_%s.gsyn" % str(neuron))
        self.neuron.run(DURATION)
        data = self.hhcell.get_data().segments[0]
        volts = data.filter(name="v")[0]#/10.0
        #data_block = all_cells.get_data()

        vm = AnalogSignal(volts,
                     units = pq.mV,
                     sampling_period = self.dt*pq.ms)
        results['vm'] = vm
        results['t'] = vm.times # self.times
        results['run_number'] = results.get('run_number',0) + 1
        return results


    def get_membrane_potential(self):
        """Must return a neo.core.AnalogSignal.
        And must destroy the hoc vectors that comprise it.
        """

        data = self.hhcell.get_data().segments[0]
        volts = data.filter(name="v")[0]


        vm = AnalogSignal(volts,
             units = pq.mV,
             sampling_period = self.dt*pq.ms)

        return vm



    def set_attrs(self,**attrs):
        self.init_backend()
        self.model.attrs.update(attrs)
        assert type(self.model.attrs) is not type(None)
        self.hhcell[0].set_parameters(**attrs)
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
        amplitude = float(c['amplitude'])
        electrode = self.neuron.DCSource(start=start, stop=stop, amplitude=amplitude)


        electrode.inject_into(self.hhcell)
        self.results = self._local_run()
        self.vm = self.results['vm']

    def get_APs(self,vm):
        vm = self.get_membrane_potential()
        waveforms = sf.get_spike_waveforms(vm,threshold=-45.0*pq.mV)
        return waveforms

    def get_spike_train(self,**run_params):
        vm = self.get_membrane_potential()

        spike_train = threshold_detection(vm,threshold=-45.0*pq.mV)

        return spike_train

    def get_spike_count(self,**run_params):
        vm = self.get_membrane_potential()
        return len(threshold_detection(vm,threshold=-45.0*pq.mV))

    model.init_backend = MethodType(init_backend,model)
    model.get_spike_count = MethodType(get_spike_count,model)
    model.get_APs = MethodType(get_APs,model)
    model.get_spike_train = MethodType(get_spike_train,model)
    model.set_attrs = MethodType(set_attrs, model) # Bind to the score.
    model.inject_square_current = MethodType(inject_square_current, model) # Bind to the score.
    model.set_attrs = MethodType(set_attrs, model) # Bind to the score.
    model.get_membrane_potential = MethodType(get_membrane_potential,model)
    model.load_model = MethodType(load_model, model) # Bind to the score.
    model._local_run = MethodType(_local_run,model)
    model.init_backend(model)
    #model.load_model() #= MethodType(load_model, model) # Bind to the score.

    return model

if pyNN_NEURON:
    HH_cond_exp = bind_NU_interface(HH_cond_exp)
