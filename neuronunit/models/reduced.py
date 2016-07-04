import os
import numpy as np
import sciunit
import neuronunit.capabilities as cap
from pyneuroml import pynml
from neo.core import AnalogSignal
import quantities as pq
import neuronunit.models as mod
import neuronunit.capabilities.spike_functions as sf

class ReducedModel(mod.LEMSModel,
                   cap.ReceivesCurrent,
                   cap.ProducesMembranePotential,
                   cap.ProducesActionPotentials):
    """Base class for reduced models, using LEMS"""

    def __init__(self, LEMS_file_path, name=None, attrs={}):
        """
        LEMS_file_path: Path to LEMS file (an xml file).
        name: Optional model name.
        """
        super(ReducedModel,self).__init__(LEMS_file_path, name=name, attrs=attrs)

    def get_membrane_potential(self, rerun=None, **run_params):
        if rerun is None:
            rerun = self.rerun
        self.run(rerun=rerun, **run_params)
        v = np.array(self.results['v'])
        t = np.array(self.results['t'])
        dt = (t[1]-t[0])*pq.s # Time per sample in milliseconds.  
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

    def inject_square_current(self,current):
        self.run_params['injected_square_current'] = current