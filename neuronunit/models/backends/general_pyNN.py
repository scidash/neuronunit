import matplotlib as mpl
mpl.use('Agg')
import copy
import pdb
import numpy as np
from .base import *
import quantities as qt
from quantities import mV, ms, s
import matplotlib.pyplot as plt
#backend('agg')
#from pyNN.neuron import *
from pyNN.neuron import HH_cond_exp
from pyNN.neuron import EIF_cond_exp_isfa_ista
from pyNN.neuron import Izhikevich

from pyNN import neuron
from pyNN.neuron import simulator as sim

from pyNN.neuron import setup as setup
from pyNN.neuron import DCSource


sim.logging.debug=False
sim.logger.setLevel(level=40)

class PYNNBackend(Backend):


    def init_backend(self, attrs = None, cell_name= 'adexp', current_src_name = 'hannah', DTC = None, dt=0.01, cell_type=None):
        backend = 'PYNN'
        super(PYNNBackend,self).init_backend()
        self.current_src_name = current_src_name
        self.cell_name = cell_name
        if type(cell_type) is None:
            self.cell_type = str('adexp')
            self.adexp = True
        else:
            self.adexp = False
        self.dt = dt
        self.DCSource = DCSource
        self.setup = setup
        self.model_path = None
        self.related_data = {}
        self.lookup = {}
        self.attrs = {}
        #self.neuron = neuron
        self.model._backend.use_memory_cache = False
        self.model.unpicklable += ['h','ns','_backend']
        neuron.setup(timestep=dt, min_delay=1.0)

        if type(DTC) is not type(None):
            if type(DTC.attrs) is not type(None):

                self.set_attrs(**DTC.attrs)
                assert len(self.model.attrs.keys()) > 0

            if hasattr(DTC,'current_src_name'):
                self._current_src_name = DTC.current_src_name

            if hasattr(DTC,'cell_name'):
                self.cell_name = DTC.cell_name


    def get_membrane_potential(self):
        """Must return a neo.core.AnalogSignal.
        And must destroy the hoc vectors that comprise it.
        """

        data = self.eif.get_data().segments[0]
        volts = data.filter(name="v")[0]


        vm = AnalogSignal(volts,
             units = mV,
             sampling_period = self.dt *ms)

        return vm

    def _local_run(self):
        '''
        pyNN lazy array demands a minimum population size of 3. Why is that.
        '''

        self.eif[0].set_parameters(**self.model.attrs)
        DURATION = 1000.0
        self.eif.record('v')
        neuron.run(DURATION)
        volts = self.eif.get_v().segments[0].analogsignals
        volts = [ v for v in volts[-1] ]

        vm = AnalogSignal(copy.copy(volts),units = mV,sampling_period = self.dt *s)
        results = {}
        results['vm'] = vm
        results['t'] = vm.times
        results['run_number'] = results.get('run_number',0) + 1
        plt.clf()
        #plt.title('')
        plt.plot(vm.times,vm)
        plt.savefig('debug_ad_exp.png')
        return results

    def load_model(self):
        neuron.setup(timestep=0.01, min_delay=1.0)
        self.eif = neuron.Population(1,EIF_cond_exp_isfa_ista())

    def set_attrs(self, **attrs):
        self.model.attrs.update(attrs)
        assert type(self.model.attrs) is not type(None)
        self.eif[0].set_parameters(**attrs)
        return self


    def set_stop_time(self, stop_time = 650*pq.ms):
        """Sets the simulation duration
        stopTimeMs: duration in milliseconds
        """
        self.tstop = float(stop_time.rescale(pq.ms))


    def inject_square_current(self, current):
        self.eif = None
        self.eif = neuron.Population(1,EIF_cond_exp_isfa_ista())
        attrs = copy.copy(self.model.attrs)
        self.init_backend()
        neuron.setup(timestep=0.01, min_delay=1.0)

        self.set_attrs(**attrs)
        c = copy.copy(current)
        if 'injected_square_current' in c.keys():
            c = current['injected_square_current']

        stop = float(c['delay'])+float(c['duration'])
        duration = float(c['duration'])
        start = float(c['delay'])
        amplitude = float(c['amplitude']/1.0)#*1000.0#*10000.0

        electrode = neuron.DCSource(start=start, stop=stop, amplitude=amplitude)
        electrode.inject_into(self.eif)

        self.results = self._local_run()
        self.vm = self.results['vm']
        #print(self.vm)
