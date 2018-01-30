"""Simulator backends for NeuronUnit models"""
import sys
import os
import platform
import re
import copy
import tempfile
import pickle
import importlib
import shelve
import subprocess

import neuronunit.capabilities as cap
from quantities import ms, mV, nA
from pyneuroml import pynml
from quantities import ms, mV
from neo.core import AnalogSignal
import neuronunit.capabilities.spike_functions as sf
import sciunit
from sciunit.utils import dict_hash, import_module_from_path
try:
    import neuron
    from neuron import h
    NEURON_SUPPORT = True
except:
    NEURON_SUPPORT = False


from neuronunit.models.backends import Backend

class pyNNBackend(Backend):

    backend = 'pyNN'
    try:
        import pyNN, lazyarray
        from pyNN import neuron
    except:
        import os
        os.system('pip install lazyarray pyNN')
        from pyNN import neuron



    def init_backend(self, attrs=None, simulator='neuron'):
        from pyNN import neuron
        from pyNN.neuron import simulator as sim
        from pyNN.neuron import setup as setup
        from pyNN.neuron import Izhikevich
        from pyNN.neuron import Population
        from pyNN.neuron import DCSource
        self.Izhikevich = Izhikevich
        self.Population = Population
        self.DCSource = DCSource
        self.setup = setup
        self.neuron = neuron
        self.model_path = None
        self.related_data = {}
        self.lookup = {}
        self.attrs = {}
        super(pyNNBackend,self).init_backend()#*args, **kwargs)


    def get_membrane_potential(self):
        """Must return a neo.core.AnalogSignal.
        And must destroy the hoc vectors that comprise it.
        """
        dt = float(copy.copy(self.neuron.dt))
        data = self.population.get_data().segments[0]
        return data.filter(name="v")[0]

    def _local_run(self):
        '''
        pyNN lazy array demands a minimum population size of 3. Why is that.
        '''
        import numpy as np
        results={}
        self.population.record('v')
        self.population.record('spikes')
        self.population[0:2].record(('v', 'spikes','u'))
        self.neuron.run(650.0)
        data = self.population.get_data().segments[0]
        results['vm'] = vm = data.filter(name="v")[0]
        sample_freq = 650.0/len(vm)
        results['t'] = np.arange(0,len(vm),650.0/len(vm))
        results['run_number'] = results.get('run_number',0) + 1
        return results


    def load_model(self):
        self.Iz = None
        self.population = None
        self.setup(timestep=0.01, min_delay=1.0)
        self.Iz = self.Izhikevich(a=0.02, b=0.2, c=-65, d=6,
                                i_offset=[0.014, 0.0, 0.0])



    def set_attrs(self, **attrs):

        self.model.attrs.update(attrs)
        assert type(self.model.attrs) is not type(None)
        #This assumes that a,b,c and d are in the attributes wich may be wrong.
        self.Iz = None
        self.population = None
        attrs_ = {x:attrs[x] for x in ['a','b','c','d']}
        self.Iz = self.Izhikevich(i_offset=[0.014, 0.0, 0.0], **attrs_)
        self.population = self.Population(3, self.Iz)

        return self

    def inject_square_current(self, current):
        attrs = self.model.attrs
        attrs_ = {x:attrs[x] for x in ['a','b','c','d']}
        self.Iz = None
        self.population = None
        self.Iz = self.Izhikevich(i_offset=[0.014, 0.0, 0.0], **attrs_)
        self.population = self.Population(3, self.Iz)

        c = copy.copy(current)
        if 'injected_square_current' in c.keys():
            c = current['injected_square_current']

        c['delay'] = re.sub('\ ms$', '', str(c['delay'])) # take delay
        c['duration'] = re.sub('\ ms$', '', str(c['duration']))
        c['amplitude'] = re.sub('\ pA$', '', str(c['amplitude']))
        stop = float(c['delay'])+float(c['duration'])
        start = float(c['delay'])
        amplitude = float(c['amplitude'])/1000.0
        print('amplitude',amplitude)
        electrode = self.neuron.DCSource(start=start, stop=stop, amplitude=amplitude)
        electrode.inject_into([self.population[0]])

'''
class brianBackend(Backend):
    """Used for generation of code for PyNN, with simulation using NEURON"""

    backend = 'brian'
    try:
        from brian2.library.IF import Izhikevich, ms
        eqs=Izhikevich(a=0.02/ms,b=0.2/ms)
        print(eqs)

    except:
        import os
        os.system('pip install brian2')
        #from brian2.library.IF import Izhikevich, ms
        #eqs=Izhikevich(a=0.02/ms,b=0.2/ms)
        #print(eqs)
'''
