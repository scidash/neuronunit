"""NeuronUnit model class for reduced neuron models."""

import sciunit
import neuronunit.capabilities as cap
from sciunit.models.runnable import RunnableModel


import numpy as np
from neo.core import AnalogSignal
import quantities as pq
from neuronunit.optimisation.data_transport_container import DataTC
import copy

import neuronunit.capabilities.spike_functions as sf
class VeryReducedModel(RunnableModel,
                       cap.ReceivesSquareCurrent,
                       cap.ProducesActionPotentials,
                       cap.ProducesMembranePotential):
    """Base class for reduced models, not using LEMS,
    and not requiring file paths this is to wrap pyNN models, Brian models,
    and other self contained models+model descriptions"""

    def __init__(self,name='',backend=None, attrs={}):
        """Instantiate a reduced model.

        LEMS_file_path: Path to LEMS file (an xml file).
        name: Optional model name.
        """
        #sciunit.Model()

        super(VeryReducedModel, self).__init__(name=name,backend=backend, attrs=attrs)
        self.backend = backend
        self.attrs = {}
        self.run_number = 0
        self.tstop = None
        self.rheobse = None

    def model_test_eval(self,tests):
        """
        Take a model and some tests
        Evaluate a test suite over them.
        """
        from sciunit import TestSuite
        if type(tests) is TestSuite:
            not_suite = TSD({t.name:t for t in tests.tests})
        OM = OptMan(tests, backend = self._backend)
        dtc = DataTC()
        dtc.attrs = self.attrs
        assert set(self._backend.attrs.keys()) in set(self.attrs.keys())
        dtc.backend = self._backend
        dtc.tests = copy.copy(not_suite)
        dtc = dtc_to_rheo(dtc)
        if dtc.rheobase is not None:
            dtc.tests = dtc.format_test()
            dtc = list(map(OM.elephant_evaluation,[dtc]))
        model = dtc.dtc_to_model()
        model.SM = dtc.SM
        model.obs_preds = dtc.obs_preds
        return dtc[0], model

    def model_to_dtc(self):
        dtc = DataTC()
        dtc.attrs = self.attrs
        try:
            dtc.backend = self.get_backend()
        except:
            dtc.backend = self.backend
        if hasattr(self,'rheobase'):
            dtc.rheobase = self.rheobase
        return dtc

    def inject_square_current(self, current):
        #pass
        vm = self._backend.inject_square_current(current)
        return vm

    def get_membrane_potential(self,**run_params):
        vm = self._backend.get_membrane_potential()
        return vm

    def get_APs(self, **run_params):
        vm = self.get_membrane_potential(**run_params)
        waveforms = sf.get_spike_waveforms(vm)#,width=10*ms)
        return waveforms

    def get_spike_train(self, **run_params):
        vm = self.get_membrane_potential(**run_params)
        spike_train = sf.get_spike_train(vm)
        return spike_train

    def get_spike_count(self, **run_params):
        train = self.get_spike_train(**run_params)
        return len(train)

    def set_attrs(self,attrs):
        self.attrs.update(attrs)
