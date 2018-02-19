from .base import *

class pyNNBackend(Backend):

    backend = 'pyNN'
    try:
        import pyNN, lazyarray
        from pyNN import neuron
    except:
        import os
        os.system('pip install lazyarray pyNN')
        from pyNN import neuron



    def init_backend(self, attrs=None, simulator='neuron', DTC = None):
        from pyNN import neuron
        self.neuron = neuron
        from pyNN.neuron import simulator as sim
        from pyNN.neuron import setup as setup
        from pyNN.neuron import Izhikevich
        from pyNN.neuron import Population
        from pyNN.neuron import DCSource
        self.Izhikevich = Izhikevich
        self.Population = Population
        self.DCSource = DCSource
        self.setup = setup
        #self.neuron = neuron
        self.model_path = None
        self.related_data = {}
        self.lookup = {}
        self.attrs = {}
        super(pyNNBackend,self).init_backend()#*args, **kwargs)
        if DTC is not None:
            #print(DTC.attrs)
            self.set_attrs(**DTC.attrs)
            #print(DTC.attrs)
            #print(self.model.attrs)

        backend = 'pyNN'


            #print(self._backend.attrs)


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
        #self.population.record('v')
        #self.population.record('spikes')
        # For ome reason you need to record from all three neurons in a population
        # In order to get the membrane potential from only the stimulated neuron.

        self.population[0:2].record(('v', 'spikes','u'))
        '''
        self.Iz.record('v')
        self.Iz.record('spikes')
        # For ome reason you need to record from all three neurons in a population
        # In order to get the membrane potential from only the stimulated neuron.

        self.Iz.record(('v', 'spikes','u'))
        '''
        #self.neuron.run(650.0)
        DURATION = 1000.0
        self.neuron.run(DURATION)

        data = self.population.get_data().segments[0]
        vm = data.filter(name="v")[0]#/10.0
        results['vm'] = vm
        #print(vm)
        sample_freq = DURATION/len(vm)
        results['t'] = np.arange(0,len(vm),DURATION/len(vm))
        results['run_number'] = results.get('run_number',0) + 1
        return results


    def load_model(self):
        self.Iz = None
        self.population = None
        self.setup(timestep=0.01, min_delay=1.0)
        import pyNN
        #i_offset=[0.014, 0.0, 0.0]
        pop = self.neuron.Population(3, pyNN.neuron.Izhikevich(a=0.02, b=0.2, c=-65, d=6, i_offset=[0.014, -65.0, 0.0]))#,v=-65))
        self.population = pop



    def set_attrs(self, **attrs):
        #attrs = copy.copy(self.model.attrs)
        self.init_backend()
        #self.set_attrs(**attrs)
        self.model.attrs.update(attrs)
        assert type(self.model.attrs) is not type(None)
        attrs['i_offset']=None
        attrs_ = {x:attrs[x] for x in ['a','b','c','d','i_offset']}
        attrs_['i_offset']=0.014#[0.014,-attrs_['v0'],0.0]
        #self.population[0].initialize()
        self.population[0].set_parameters(**attrs_)

        print(self.population[0].get_parameters())
        self.neuron.h.psection()
        return self

    def inject_square_current(self, current):
        import copy
        attrs = copy.copy(self.model.attrs)
        self.init_backend()
        self.set_attrs(**attrs)
        c = copy.copy(current)
        if 'injected_square_current' in c.keys():
            c = current['injected_square_current']

        c['delay'] = re.sub('\ ms$', '', str(c['delay'])) # take delay
        c['duration'] = re.sub('\ ms$', '', str(c['duration']))
        c['amplitude'] = re.sub('\ pA$', '', str(c['amplitude']))
        stop = float(c['delay'])+float(c['duration'])
        start = float(c['delay'])
        amplitude = float(c['amplitude'])/1000.0
        #print('amplitude',amplitude)
        electrode = self.neuron.DCSource(start=start, stop=stop, amplitude=amplitude)
        print(self.population[0])
        print(type(self.population[0]))
        print(self.population[0].get_parameters())

        electrode.inject_into(self.population[0:1])
