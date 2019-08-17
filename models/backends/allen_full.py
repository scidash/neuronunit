
from allensdk.model.biophys_sim.config import Config
from allensdk.model.biophysical.utils import create_utils
from allensdk.core.nwb_data_set import NwbDataSet
import allensdk.ephys.extract_cell_features as extract_cell_features
from shutil import copy
import numpy
import logging
import time
import os
import multiprocessing as mp
from functools import partial

import asciiplotlib as apl
import io
import math
import pdb
#from numba import jit

import numpy as np
from .base import *
import quantities as qt
from quantities import mV, ms, s, us, ns
import matplotlib as mpl
_runner_log = logging.getLogger('allensdk.model.biophysical.runner')
_lock = None
def _init_lock(lock):
    global _lock
    _lock = lock
_runner_log.info = print


class ALLENBIOBackend(Backend):
    def init_backend(self, attrs=None, cell_name='allen_full_bio',
                     current_src_name='spanner', DTC=None,
                     debug = False):
        backend = 'allenbio'
        super(ALLENBIOBackend,self).init_backend()
        self.name = str(backend)

        #self.threshold = -20.0*qt.mV
        self.debug = None
        self.model._backend.use_memory_cache = False
        self.current_src_name = current_src_name
        self.cell_name = cell_name
        self.vM = None
        self.attrs = attrs
        self.debug = debug
        self.temp_attrs = None
        self.n_spikes = None
        self.spike_monitor = None


        self.model.get_spike_count = self.get_spike_count


        if type(attrs) is not type(None):
            self.set_attrs(**attrs)
            self.sim_attrs = attrs

        if type(DTC) is not type(None):
            if type(DTC.attrs) is not type(None):
                self.set_attrs(**DTC.attrs)
            if hasattr(DTC,'current_src_name'):
                self._current_src_name = DTC.current_src_name
            if hasattr(DTC,'cell_name'):
                self.cell_name = DTC.cell_name


    def get_membrane_potential(self):
        recorded_data = utils.get_recorded_data(vec)



        #vm = [ float(i) for i in vm ]
        dt = recorded_data["t"][1] - recorded_data["t"][0]
        self.vM = AnalogSignal(recorded_data["v"],units = mV,sampling_period = dt * pq.ms)

        fig = apl.figure()
        fig.plot(t, v, label=str('spikes: ')+str(self.n_spikes), width=100, height=20)
        fig.show()

    def inject_square_current(self, current):#, section = None, debug=False):
        """Inputs: current : a dictionary with exactly three items, whose keys are: 'amplitude', 'delay', 'duration'
        Example: current = {'amplitude':float*pq.pA, 'delay':float*pq.ms, 'duration':float*pq.ms}}
        where \'pq\' is a physical unit representation, implemented by casting float values to the quanitities \'type\'.
        Description: A parameterized means of applying current injection into defined
        Currently only single section neuronal models are supported, the neurite section is understood to be simply the soma.

        """

        attrs = copy.copy(self.model.attrs)
        self.set_attrs(**attrs)

        utils = create_utils(description)

        h = utils.h

        # configure model
        manifest = description.manifest
        morphology_path = description.manifest.get_path('MORPHOLOGY')

        try:
            utils.generate_morphology(morphology_path)
        except:
            utils.generate_morphology(morphology_path.encode('ascii', 'ignore'))

        utils.load_cell_parameters()
        # configure stimulus and recording
        stimulus_path = description.manifest.get_path('stimulus_path')
        run_params = description.data['runs'][0]
        if sweeps is None:
            sweeps = run_params['sweeps']
        try:
            sweeps_by_type = run_params['sweeps_by_type']
        except:
            pass
        output_path = manifest.get_path("output_path")

        sweep = sweeps
        _runner_log.info("Loading sweep: %d" % (sweep))


        if 'injected_square_current' in current.keys():
            c = current['injected_square_current'];
        else:
            c = current
        amplitude = float(c['amplitude'])#*1000.0
        duration = int(c['duration'])#/dt#/dt.rescale('ms')
        delay = int(c['delay'])#/dt#.rescale('ms')
        pre_current = int(duration)+100
        stim_curr = [ 0.0 ] * int(start) + [ amplitude ] * int(duration) + [ 0.0 ] * int(stop)
        utils.setup_iclamp()

        # configure NEURON

        utils.setup_iclamp(stimulus_path, sweep=sweep)

        _runner_log.info("Simulating sweep: %d" % (sweep))
        vec = utils.record_values()
        tstart = time.time()
        h.finitialize()
        h.run()
        tstop = time.time()
        _runner_log.info("Time: %f" % (tstop - tstart))

        # write to an NWB File
        _runner_log.info("Writing sweep: %d" % (sweep))
        recorded_data = utils.get_recorded_data(vec)



        #vm = [ float(i) for i in vm ]
        dt = recorded_data["t"][1] - recorded_data["t"][0]
        self.vM = AnalogSignal(recorded_data["v"],units = mV,sampling_period = dt * pq.ms)

        fig = apl.figure()
        fig.plot(t, v, label=str('spikes: ')+str(self.n_spikes), width=100, height=20)
        fig.show()
        return self.vM

    def set_attrs():
        seclist = utils.description.data['genome']
        for seclist in utils.description.data['genome']:
            for key,value in seclist.items():
                print(key,value)
        for p in utils.description.data['passive']:
            for k,v in p.items():
                print(k,v)

    def get_raw_mp(self):
        t = [float(f) for f in self.vM.times]
        v = [float(f) for f in self.vM.magnitude]

    def _backend_run(self):
        results = None
        results = {}

        results['vm'] = self.vM

        results['t'] = self.vM.times
        results['run_number'] = results.get('run_number',0) + 1
        return results

def run(description, sweeps=None, procs=6,swn = 0):
    '''Main function for simulating sweeps in a biophysical experiment.

    Parameters
    ----------
    description : Config
        All information needed to run the experiment.
    procs : int
        number of sweeps to simulate simultaneously.
    sweeps : list
        list of experiment sweep numbers to simulate.  If None, simulate all sweeps.
    '''

    prepare_nwb_output(description.manifest.get_path('stimulus_path'))


    if sweeps is None:
        stimulus_path = description.manifest.get_path('stimulus_path')
        run_params = description.data['runs'][0]
        sweeps = run_params['sweeps']

    (vm,times,wraped) = run_sync(description,sweeps[swn])
    return (vm,times,wraped)

from neuronunit.tests.allen_tests import allen_format 

def examination(vm,time,wraped):
    model = pickle.load(open('typical_model.p','rb'))
    model.vm30 = wraped
    model.static = None
    model.static = True
    size=len(model.vm30)
    short = vm[0:size]
    model.vm30 = short
    #(a,b) = allen_format(np.array([float(v) for v in short]),np.array([float(t) for t in model.vm30.times])
    #scores = [ judge(model,t) for t in test_collection ]
    (a,b) = allen_format(np.array([float(v) for v in vm]),np.array([float(t) for t in time]))
                         
    return (a,b)
                         
def run_sync(description, sweeps=None):
    '''Single-process main function for simulating sweeps in a biophysical experiment.

    Parameters
    ----------
    description : Config
        All information needed to run the experiment.
    sweeps : list
        list of experiment sweep numbers to simulate.  If None, simulate all sweeps.
    '''

    # configure NEURON
    utils = create_utils(description)

    h = utils.h

    # configure model
    manifest = description.manifest
    morphology_path = description.manifest.get_path('MORPHOLOGY')

    try:
        utils.generate_morphology(morphology_path)
    except:
        utils.generate_morphology(morphology_path.encode('ascii', 'ignore'))

    utils.load_cell_parameters()
    # configure stimulus and recording
    stimulus_path = description.manifest.get_path('stimulus_path')
    run_params = description.data['runs'][0]
    if sweeps is None:
        sweeps = run_params['sweeps']
    try:
        sweeps_by_type = run_params['sweeps_by_type']
    except:
        pass
    output_path = manifest.get_path("output_path")

    sweep = sweeps
    _runner_log.info("Loading sweep: %d" % (sweep))

    utils.setup_iclamp(stimulus_path, sweep=sweep)

    _runner_log.info("Simulating sweep: %d" % (sweep))
    vec = utils.record_values()
    tstart = time.time()
    h.finitialize()
    h.run()
    tstop = time.time()
    _runner_log.info("Time: %f" % (tstop - tstart))

    # write to an NWB File
    _runner_log.info("Writing sweep: %d" % (sweep))
    recorded_data = utils.get_recorded_data(vec)
    dt = recorded_data["t"][1] - recorded_data["t"][0]
    wraped = AnalogSignal(recorded_data["v"],units = mV,sampling_period = dt * pq.ms)
    print('gets here \n\n\n\n')
    try:                     
        (a,b) = examination(recorded_data["v"],recorded_data["t"],wraped)
        print('success \n\n\n\n')
        print(a,b)
        import pdb
        pdb.set_trace()
    except:
        print('fail')
        #pass
    return (recorded_data["v"],recorded_data["t"],wraped)



def prepare_nwb_output(nwb_stimulus_path):
    '''Copy the stimulus file, zero out the recorded voltages and spike times.

    Parameters
    ----------
    nwb_stimulus_path : string
        NWB file name
    nwb_result_path : string
        NWB file name
    '''
    import os
    nwb_result_path = os.getcwd()
    output_dir = os.path.dirname(nwb_result_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    #copy(nwb_stimulus_path, nwb_result_path)
    '''
    data_set = NwbDataSet(nwb_result_path)
    data_set.fill_sweep_responses(0.0, extend_experiment=True)
    for sweep in data_set.get_sweep_numbers():
        data_set.set_spike_times(sweep, [])
    '''

def save_nwb(output_path, v, sweep, sweep_by_type= None):
    '''Save a single voltage output result into an existing sweep in a NWB file.
    This is intended to overwrite a recorded trace with a simulated voltage.

    Parameters
    ----------
    output_path : string
        file name of a pre-existing NWB file.
    v : numpy array
        voltage
    sweep : integer
        which entry to overwrite in the file.
    '''
    output = NwbDataSet(output_path)
    output.set_sweep(sweep, None, v)
    if sweep_by_type is not None:
        sweep_by_type = {t: [sweep]
                     for t, ss in sweeps_by_type.items() if sweep in ss}
        sweep_features = extract_cell_features.extract_sweep_features(output,
                                                                  sweep_by_type)
    try:
        spikes = sweep_features[sweep]['spikes']
        spike_times = [s['threshold_t'] for s in spikes]
        output.set_spike_times(sweep, spike_times)
    except Exception as e:
        logging.info("sweep %d has no sweep features. %s" % (sweep, e.args))


def load_description(manifest_json_path):
    '''Read configuration file.

    Parameters
    ----------
    manifest_json_path : string
        File containing the experiment configuration.

    Returns
    -------
    Config
        Object with all information needed to run the experiment.
    '''
    description = Config().load(manifest_json_path)

    # fix nonstandard description sections
    fix_sections = ['passive', 'axon_morph,', 'conditions', 'fitting']
    description.fix_unary_sections(fix_sections)

    return description
