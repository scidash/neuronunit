"""NeuronUnit model class for reduced neuron models"""

import numpy as np
from neo.core import AnalogSignal
import quantities as pq

import neuronunit.capabilities as cap
import neuronunit.models as mod
import neuronunit.capabilities.spike_functions as sf

class ReducedModel(mod.LEMSModel,
                   cap.ReceivesCurrent,
                   cap.ProducesActionPotentials):
    """Base class for reduced models, using LEMS"""

    def __init__(self, LEMS_file_path, name=None, backend=None, attrs=None):
        """
        LEMS_file_path: Path to LEMS file (an xml file).
        name: Optional model name.
        """
        mod.LEMSModel.__init__(self,LEMS_file_path,name=name,
                                          backend=backend,attrs=attrs)
        #super(ReducedModel,self).__init__(LEMS_file_path,name=name,
        #                                  backend=backend,attrs=attrs)
        self.run_number=0
        self.previous=0
        self.lookup = {}
        self.rheobase_memory = None
    def get_membrane_potential(self, rerun=None, **run_params):
        if rerun is None:
            rerun = self.rerun
        self.run(rerun=rerun, **run_params)
        for rkey in self.results.keys():
            if 'v' in rkey or 'vm' in rkey:
                v = np.array(self.results[rkey])
        t = np.array(self.results['t'])
        dt = (t[1]-t[0])*pq.s # Time per sample in seconds.
        vm = AnalogSignal(v,units=pq.V,sampling_rate=1.0/dt)
        return vm

    def get_APs(self, rerun=False, **run_params):
        vm = self.get_membrane_potential(rerun=rerun, **run_params)
        waveforms = sf.get_spike_waveforms(vm)
        return waveforms

    def get_spike_train(self, rerun=False, **run_params):
        vm = self.get_membrane_potential(rerun=rerun, **run_params)
        spike_train = sf.get_spike_train(vm)
        return spike_train

    #This method must be overwritten in the child class or Derived class 
    # NEURONbackend but I don't understand how to do that.
    #def inject_square_current(self,current):
    #    self.run_params['injected_square_current'] = current
