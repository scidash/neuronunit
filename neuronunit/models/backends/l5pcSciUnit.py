import pickle
from neo.core import AnalogSignal
import sciunit
from sciunit.models import RunnableModel
import sciunit.capabilities as scap
import neuronunit.capabilities as cap
import neuronunit.capabilities.spike_functions as sf

from collections import OrderedDict
from neuronunit.models.backends import l5pc_evaluator

import json
import os
path = os.path.dirname(os.path.abspath(__file__))

PARAMS = json.load(open(str(path)+'/config/params.json'))
import time
import matplotlib.pyplot as plt


from quantities import mV, ms, s, V
import sciunit
from neo import AnalogSignal
import neuronunit.capabilities as cap
import numpy as np
#from .base import Backend
from .base import *
import quantities as qt
from bluepyopt.parameters import Parameter

def timer(func):
    def inner(*args, **kwargs):
        t1 = time.time()
        f = func(*args, **kwargs)
        t2 = time.time()
        print('time taken on block {0} '.format(t2-t1))
        return f
    return inner

class L5PCBackend(Backend):
    name = 'L5PC'
    def init_backend(self, attrs={}, DTC=None):
        self.model._backend.use_memory_cache = False
        self.evaluator = l5pc_evaluator.create()
        self.evaluator.fitness_protocols.pop('bAP',None)
        self.evaluator.fitness_protocols.pop('Step3',None)
        self.evaluator.fitness_protocols.pop('Step2',None)
        self.evaluator.NU = None
        self.evaluator.NU = True
        self.run_params = {}
        self.test_params = pickle.load(open(str(path)+'/test_params.p','rb'))


        l5_pc_keys = ['gNaTs2_tbar_NaTs2_t.apical', 'gSKv3_1bar_SKv3_1.apical', 'gImbar_Im.apical', 'gNaTa_tbar_NaTa_t.axonal', 'gNap_Et2bar_Nap_Et2.axonal', 'gK_Pstbar_K_Pst.axonal', 'gK_Tstbar_K_Tst.axonal', 'gSK_E2bar_SK_E2.axonal', 'gSKv3_1bar_SKv3_1.axonal', 'gCa_HVAbar_Ca_HVA.axonal', 'gCa_LVAstbar_Ca_LVAst.axonal', 'gamma_CaDynamics_E2.axonal', 'decay_CaDynamics_E2.axonal', 'gNaTs2_tbar_NaTs2_t.somatic', 'gSKv3_1bar_SKv3_1.somatic', 'gSK_E2bar_SK_E2.somatic', 'gCa_HVAbar_Ca_HVA.somatic', 'gCa_LVAstbar_Ca_LVAst.somatic', 'gamma_CaDynamics_E2.somatic', 'decay_CaDynamics_E2.somatic']
        l5_pc_values = [0.0009012730575340265, 0.024287352056036934, 0.0008315987398062784, 1.7100532387472567, 0.7671786030824507, 0.47339571930108143, 0.0025715065622581644, 0.024862299158354962, 0.7754822886266044, 0.0005560440082771592, 0.0020639185209852568, 0.013376906273759268, 207.56154268835758, 0.5154365543590191, 0.2565961138691978, 0.0024100296151316754, 0.0007416593834676707, 0.006240529502225737, 0.028595343511797353, 226.7501580822364]

        L5PC = OrderedDict()
        for k,v in zip(l5_pc_keys,l5_pc_values):
            L5PC[k] = v

        self.default_attrs = L5PC
        self.attrs = attrs
        lop={}
        for k,v in self.attrs.items():
            p = Parameter(name=k,bounds=v,frozen=False)
            lop[k] = p
        self.params = lop

        super(L5PCBackend,self).init_backend()

        if type(DTC) is not type(None):
            if type(DTC.attrs) is not type(None):
                self.attrs = attrs
            if type(DTC.attrs) is type(None):
                self.attrs = self.default_attrs

    '''
    def params_by_names(self, param_names):
        """Get parameter objects by name"""

        return [self.params[param_name] for param_name in param_names]

    def freeze(self, param_dict):
        """Set params"""

        for param_name, param_value in param_dict.items():
            self.params[param_name].freeze(param_value)

    def unfreeze(self, param_names):
        """Unset params"""

        for param_name in param_names:
            self.params[param_name].unfreeze()

    def instantiate(self, sim=None):
        """
        Instantiate model in simulator
        As if called from a genetic algorithm.
        """
        #self.icell.gid = self.gid
        if self.params is not None:
            self.attrs = self.params

        dtc = self.model_to_dtc()
        for k,v in self.params.items():
            v = float(v.value)
            dtc.attrs[k] = v
            self.attrs[k] = v
        return dtc

    def destroy(self, sim=None):  # pylint: disable=W0613
        """Destroy instantiated model in simulator"""

        # Make sure the icell's destroy() method is called
        # without it a circular reference exists between CellRef and the object
        # this prevents the icells from being garbage collected, and
        # cell objects pile up in the simulator
        self.icell.destroy()

        # The line below is some M. Hines magic
        # DON'T remove it, because it will make sure garbage collection

        del self.icell# = None
        for param in self.params.values():
            param.destroy(sim=sim)
            print('destroyed param')

    def check_nonfrozen_params(self, param_names):  # pylint: disable=W0613
        """Check if all nonfrozen params are set"""
        for param_name, param in self.params.items():
            if not param.frozen:
                raise Exception(
                    'CellModel: Nonfrozen param %s needs to be '
                    'set before simulation' %
                    param_name)
    '''

    def set_attrs(self,attrs=None):
        #print('these are parameters that can be modified.')
        not_fronzen = {k:v for k,v in self.evaluator.cell_model.params.items() if not v.frozen}
        for k,v in attrs.items():
           self.test_params[i] = v
        #print(self.test_params)
        #self.attrs = attrs
        '''
        lop={}
        for k,v in self.attrs.items():
            p = Parameter(name=k,bounds=v,frozen=False)
            lop[k] = p
        self.params = lop
        '''

    def model_to_dtc(self,attrs=None):
        """
        Args:
            self
        Returns: 
            dtc
            DTC is a simulator indipendent data transport container object.
        """
            
        dtc = DataTC(backend=self.backend)
        if hasattr(self,'tests'):
            if type(self.tests) is not type(None):
                dtc.tests = self.tests

        if type(attrs) is not type(None):
            if len(attrs):
                dtc.attrs = attrs
                self.attrs = attrs
            assert self._backend is not None
            return dtc
        else:
            if type(self.attrs) is not type(None):
                if len(self.attrs):
                    try:
                        dynamic_attrs = {str(k):float(v) for k,v in self.attrs.items()}
                    except:
                        dynamic_attrs = {str(k):float(v.value) for k,v in self.attrs.items()}

        if self._backend is None:
            super(VeryReducedModel, self).__init__(name=self.name,backend=self.backend)#,attrs=dtc.attrs)
            assert self._backend is not None
        frozen_attrs = self._backend.default_attrs
        if 'dynamic_attrs' in locals():
            frozen_attrs.update(dynamic_attrs)
        all_attrs = frozen_attrs
        dtc.attrs = all_attrs
        assert dtc.attrs is not None
        return dtc


    def inject_square_current(self,current):
        '''
        self.evaluator = l5pc_evaluator.create()
        self.evaluator.fitness_protocols.pop('bAP',None)
        self.evaluator.fitness_protocols.pop('Step3',None)
        self.evaluator.fitness_protocols.pop('Step2',None)
        '''
        protocol = self.evaluator.fitness_protocols['Step1']
        if 'injected_square_current' in current.keys():
            current = current['injected_square_current']
        protocol.stimuli[0].step_amplitude = float(current['amplitude'])/1000.0
        protocol.stimuli[0].step_delay = float(current['delay'])#/(1000.0*1000.0*1000.0)#*1000.0
        protocol.stimuli[0].step_duration = float(current['duration'])#/(1000.0*1000.0*1000.0)#*1000.0
        

        #self.init_backend(attrs=self.attrs)
        feature_outputs = self.evaluator.evaluate(self.test_params)

        #self.destroy()
        #self.init_backend(attrs=self.attrs)# DTC=self.model_to_dtc())
        try:
            self.vm = feature_outputs['neo_Step1.soma.v']
        except:
            import pdb
            pdb.set_trace()
        
        self.vM = self.vm
        #plt.plot(self.vm.times,self.vm)
        #self.evaluator = None
        #del self.evaluator
        return feature_outputs['neo_Step1.soma.v']
    
    def get_spike_count(self):
        train = sf.get_spike_train(self.vm)
        print(len(train))
        return len(train)

    def get_membrane_potential(self):
        """Return the Vm passed into the class constructor."""
        
        return self.vm

    def get_APs(self):
        """Return the APs, if any, contained in the static waveform."""
        vm = self.vm 
        waveforms = sf.get_spike_waveforms(vm)
        return waveforms
