"""NeuronUnit model class for reduced neuron models."""

import numpy as np
from neo.core import AnalogSignal
import quantities as pq

import neuronunit.capabilities as cap

from .static import ExternalModel
import neuronunit.capabilities.spike_functions as sf
from neuronunit.optimisation.model_parameters import path_params

from .lems import LEMSModel
class ReducedModel(LEMSModel,
                   cap.ReceivesSquareCurrent,
                   cap.ProducesActionPotentials,
                   ):
    """Base class for reduced models, using LEMS"""

    def __init__(self, LEMS_file_path, name=None, backend=None, attrs=None):
        """Instantiate a reduced model.

        LEMS_file_path: Path to LEMS file (an xml file).
        name: Optional model name.
        """

        super(ReducedModel, self).__init__(LEMS_file_path, name=name,
                                           backend=backend, attrs=attrs)
        self.run_number = 0
        self.tstop = None

    def get_membrane_potential(self, **run_params):
        self.run(**run_params)
        for rkey in self.results.keys():
            if 'v' in rkey or 'vm' in rkey:
                v = np.array(self.results[rkey])
        t = np.array(self.results['t'])
        dt = (t[1]-t[0])*pq.s  # Time per sample in seconds.
        vm = AnalogSignal(v, units=pq.V, sampling_rate=1.0/dt)
        return vm

    def get_APs(self, **run_params):
        try:
            vm = self._backend.get_membrane_potential(**run_params)
        except:
            vm = self.get_membrane_potential(**run_params)
        if hasattr(self._backend,'name'):

            self._backend.threshold = np.max(vm)-np.max(vm)/250.0
            waveforms = sf.get_spike_waveforms(vm,self._backend.threshold)
        else:
            waveforms = sf.get_spike_waveforms(vm)
        return waveforms

    def get_spike_train(self, **run_params):
        vm = self.get_membrane_potential(**run_params)
        #spike_train = sf.get_spike_train(vm)
        #if str('ADEXP') in self._backend.name:
        if hasattr(self._backend,'name'):
            self._backend.threshold = np.max(vm)-np.max(vm)/250.0
            spike_train = sf.get_spike_train(vm,self._backend.threshold)
        else:
            spike_train = sf.get_spike_train(vm)

        return spike_train

    def inject_square_current(self, current):
        assert isinstance(current, dict)
        assert all(x in current for x in
                   ['amplitude', 'delay', 'duration'])
        self.set_run_params(injected_square_current=current)
        self._backend.inject_square_current(current)


class VeryReducedModel(ReducedModel,
                   cap.ReceivesCurrent,
                   cap.ProducesActionPotentials,
                   ):
    """Base class for reduced models, using LEMS"""

    def __init__(self, name=None, backend=None, attrs=None):
        """Instantiate a reduced model.
        LEMS_file_path: Path to LEMS file (an xml file).
        name: Optional model name.
        """
        LEMS_MODEL_PATH = path_params['model_path']
        #model = ReducedModel(LEMS_MODEL_PATH,name = str('vanilla'),backend = str(backend))
        super(VeryReducedModel,self).__init__(LEMS_MODEL_PATH,name=name,backend=backend)
        #self.name=name,
        #self.backend=backend,
        self.attrs=attrs
        #self.run_number = 0
        #self.tstop = None
        #self.attrs = attrs if attrs else {}
        #    self.unpicklable = []
        #self._backend = backend

    def set_attrs(self,attrs):
        self._backend.set_attrs(**attrs)


    def get_backend(self):
        return self._backend

    '''
    def run(self, rerun=None, **run_params):
        if rerun is None:
            rerun = self.rerun
        self.set_run_params(**run_params)
        for key,value in self.run_defaults.items():
            if key not in self.run_params:
                self.set_run_params(**{key:value})
        #if (not rerun) and hasattr(self,'last_run_params') and \
        #   self.run_params == self.last_run_params:
        #    print("Same run_params; skipping...")
        #    return

        self.results = self._backend.local_run()
        self.last_run_params = deepcopy(self.run_params)
        #self.rerun = False
        # Reset run parameters so the next test has to pass its own
        # run parameters and not use the same ones
        self.run_params = {}

    def set_run_params(self, **params):
        self._backend.set_run_params(**params)


    def set_backend(self, backend):
        if isinstance(backend,str):
            name = backend
            args = []
            kwargs = {}
        elif isinstance(backend,(tuple,list)):
            name = ''
            args = []
            kwargs = {}
            for i in range(len(backend)):
                if i==0:
                    name = backend[i]
                else:
                    if isinstance(backend[i],dict):
                        kwargs.update(backend[i])
                    else:
                        args += backend[i]
        else:
            raise TypeError("Backend must be string, tuple, or list")
        if name in available_backends:
            self.backend = name
            self._backend = available_backends[name]()
        elif name is None:
            # The base class should not be called.
            raise Exception(("A backend (e.g. 'jNeuroML' or 'NEURON') "
                             "must be selected"))
        else:
            #print(name,available_backends)
            #import pdb; pdb.set_trace()
            raise Exception("Backend %s not found in backends.py" \
                            % name)
        self._backend.model = self
        self._backend.init_backend(*args, **kwargs)
    # Methods to override using inheritance.
    def get_membrane_potential(self, **run_params):
        pass
    def get_APs(self, **run_params):
        pass
    def get_spike_train(self, **run_params):
        pass
    def inject_square_current(self, current):
        pass
    '''
