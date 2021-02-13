#!/usr/bin/env python
# coding: utf-8
from typing import Any, Dict, List, Optional, Tuple, Type, Union, Text
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import pickle
from neo.core import AnalogSignal
import quantities as qt
from elephant.spike_train_generation import threshold_detection

from allensdk.core.cell_types_cache import CellTypesCache
from allensdk.ephys.extract_cell_features import extract_cell_features
from allensdk.core.nwb_data_set import NwbDataSet


from sciunit import TestSuite

from neuronunit.optimization.model_parameters import MODEL_PARAMS
from neuronunit.optimization.optimization_management import efel_evaluation
from neuronunit.models import StaticModel
from neuronunit.tests.make_allen_tests import AllenTest
from neuronunit.tests.target_spike_current import SpikeCountSearch
from elephant.statistics import isi
from sklearn.linear_model import LinearRegression


def allen_id_to_sweeps(specimen_id: int = -1) -> Union[defaultdict, Any, List]:
    ctc = CellTypesCache(manifest_file="cell_types/manifest.json")

    specimen_id = int(specimen_id)
    data_set = ctc.get_ephys_data(specimen_id)
    sweeps = ctc.get_ephys_sweeps(specimen_id)
    sweep_numbers = defaultdict(list)
    for sweep in sweeps:
        sweep_numbers[sweep["stimulus_name"]].append(sweep["sweep_number"])
    return sweep_numbers, data_set, sweeps


def closest(lst, K):
    lst = np.asarray(lst)
    idx = (np.abs(lst - K)).argmin()
    return idx




def sweeps_build_fi_tests(data_set, sweep_numbers, specimen_id):
    sweep_numbers = sweep_numbers["Square - 2s Suprathreshold"]
    rheobase = -1
    above_threshold_sn = []
    currents = {}
    relation_map = {}
    for sn in sweep_numbers:

        spike_times = data_set.get_spike_times(sn)

        if len(spike_times) > 1:
            isi_ = isi(spike_times)
            rate = 1.0 / np.mean(isi_)
            sweep_data = data_set.get_sweep(sn)

            current_stim = np.max(sweep_data["stimulus"])
            relation_map[rate] = current_stim
    model = LinearRegression()
    x = np.array(list(relation_map.keys())).reshape(-1, 1)
    y = np.array(list(relation_map.values())).reshape(-1, 1)
    if len(x):
        model.fit(x, y)
        m = model.coef_ * (qt.Hz / qt.pA)
        if type(m) is type(list()):
            m = m[0]
        slope = m
        plot = False
        if plot:
            c = model.intercept_
            y = supra_thresh_I * float(m) + c
            plt.figure()  # new figure
            plt.plot(supra_thresh_I, fr)
            plt.plot(supra_thresh_I, y)
            plt.xlabel("Amplitude of Injecting step current (pA)")
            plt.ylabel("Firing rate (Hz)")
            plt.grid()
            plt.show()
    else:
        slope = None
    return slope


def get_rheobase(numbers, sets):
    rheobase_numbers = [
        sweep_number
        for sweep_number in numbers
        if len(sets.get_spike_times(sweep_number)) == 1
    ]
    sweeps = [sets.get_sweep(n) for n in rheobase_numbers]
    temp = [
        (i, np.max(s["stimulus"]))
        for i, s in zip(rheobase_numbers, sweeps)
        if "stimulus" in s.keys()
    ]  # if np.min(s['stimulus'])>0 ]
    temp = sorted(temp, key=lambda x: [1], reverse=True)
    rheobase = temp[0][1]
    index = temp[0][0]
    return rheobase, index


def get_model_parts(data_set, sweep_numbers, specimen_id):
    sweep_numbers = sweep_numbers["Square - 2s Suprathreshold"]
    rheobase = -1
    above_threshold_sn = []
    currents = {}
    # this_cnt_scheme = 0
    for sn in sweep_numbers:
        sweep_data = data_set.get_sweep(sn)

        spike_times = data_set.get_spike_times(sn)

        # stimulus is a numpy array in amps
        stimulus = sweep_data["stimulus"]

        if len(spike_times) == 1:
            if np.max(stimulus) > rheobase and rheobase == -1:
                rheobase = np.max(stimulus)
                stim = rheobase
                currents["rh"] = stim
                sampling_rate = sweep_data["sampling_rate"]
                vmrh = AnalogSignal(
                    [v * 1000 for v in sweep_data["response"]],
                    sampling_rate=sampling_rate * qt.Hz,
                    units=qt.mV,
                )
                vmrh = vmrh[0 : int(len(vmrh) / 2.1)]
        if len(spike_times) >= 1:
            reponse = sweep_data["response"]
            sampling_rate = sweep_data["sampling_rate"]
            vmm = AnalogSignal(
                [v * 1000 for v in sweep_data["response"]],
                sampling_rate=sampling_rate * qt.Hz,
                units=qt.mV,
            )
            vmm = vmm[0 : int(len(vmm) / 2.1)]
            above_threshold_sn.append((np.max(stimulus), sn, vmm))
    if rheobase == -1:
        rheobase = above_threshold_sn[0][0]
        vmrh = above_threshold_sn[0][2]
    myNumber = 3.0 * rheobase
    currents_ = [t[0] for t in above_threshold_sn]
    indexvm30 = closest(currents_, myNumber)
    stim = above_threshold_sn[indexvm30][0]
    currents["30"] = stim
    vm30 = above_threshold_sn[indexvm30][2]
    myNumber = 1.5 * rheobase
    currents_ = [t[0] for t in above_threshold_sn]
    indexvm15 = closest(currents_, myNumber)
    stim = above_threshold_sn[indexvm15][0]
    currents["15"] = stim
    vm15 = above_threshold_sn[indexvm15][2]
    vm15.sn = None
    vm15.sn = above_threshold_sn[0][1]
    vm15.specimen_id = None
    vm15.specimen_id = specimen_id

    del sweep_numbers
    del data_set
    return vm15, vm30, rheobase, currents, vmrh


def get_model_parts_sweep_from_spk_cnt(spk_cnt, data_set, sweep_numbers, specimen_id):
    sweep_numbers = sweep_numbers["Square - 2s Suprathreshold"]
    rheobase = -1
    above_threshold_sn = []

    for sn in sweep_numbers:
        spike_times = data_set.get_spike_times(sn)
        if len(spike_times) >= spk_cnt:
            # stimulus is a numpy array in amps
            sweep_data = data_set.get_sweep(sn)
            stimulus = sweep_data["stimulus"]
            reponse = sweep_data["response"]

            if len(spike_times):
                thresh_ = len(np.where(reponse > 0))

            sampling_rate = sweep_data["sampling_rate"]
            vmm = AnalogSignal(
                [v * 1000 for v in reponse],
                sampling_rate=sampling_rate * qt.Hz,
                units=qt.mV,
            )
            # Reduce the recording to a smaller time window when interesting things
            # happen
            vmm = vmm[0 : int(len(vmm) / 2.1)]
            # store the sweep number in the neo AnalogSignal trace object.
            vmm.sn = None
            vmm.sn = sn
            return vmm, stimulus, sn, spike_times
    return None, None, None, None


def get_model_parts_sweep_from_number(sn, data_set, sweep_numbers, specimen_id):
    sweep_numbers = sweep_numbers["Square - 2s Suprathreshold"]
    rheobase = -1
    above_threshold_sn = []
    currents = {}
    sweep_data = data_set.get_sweep(sn)
    stimulus = sweep_data["stimulus"]
    spike_times = data_set.get_spike_times(sn)
    reponse = sweep_data["response"]

    sampling_rate = sweep_data["sampling_rate"]
    vmm = AnalogSignal(
        [v * 1000 for v in reponse], sampling_rate=sampling_rate * qt.Hz, units=qt.mV
    )
    vmm = vmm[0 : int(len(vmm) / 2.1)]
    vmm.sn = None
    vmm.sn = sn

    return vmm, stimulus, sn, spike_times


def make_suite_known_sweep_from_static_models(
    vm_soma, stimulus, specimen_id, efel_filter_iterable=None
):
    sm = StaticModel(vm=vm_soma)
    sm.backend = "static_model"
    sm.vm_soma = vm_soma
    sm = efel_evaluation(
        sm, current=np.max(stimulus), efel_filter_iterable=efel_filter_iterable
    )
    allen_tests = []
    if sm.efel is not None:
        for k, v in sm.efel.items():
            try:
                # if v is not None:
                at = AllenTest(name=str(k))
                at.set_observation(v)
                at = wrangle_tests(at)
                allen_tests.append(at)
            except:
                pass
    suite = TestSuite(allen_tests, name=str(specimen_id))
    suite.traces = None
    suite.traces = {}
    suite.traces["vm_soma"] = sm.vm_soma

    suite.model = sm
    suite.stim = stimulus
    return suite, specimen_id


def wrangle_tests(t):
    if hasattr(t.observation["mean"], "units"):
        t.observation["mean"] = (
            np.mean(t.observation["mean"]) * t.observation["mean"].units
        )
        t.observation["std"] = (
            np.mean(t.observation["mean"]) * t.observation["mean"].units
        )
    else:
        t.observation["mean"] = np.mean(t.observation["mean"])
        t.observation["std"] = np.mean(t.observation["mean"])
    return t


def make_suite_from_static_models(vm_soma, vm30, rheobase, currents, vmrh, specimen_id):

    if vmrh is not None:
        sm = StaticModel(vm=vmrh)
    else:
        sm = StaticModel(vm=vm30)
    sm.rheobase = rheobase
    sm.vm_soma = vm_soma
    sm = efel_evaluation(sm, current=rheobase)

    sm = rekeyed(sm)
    useable = False
    sm.vmrh = vmrh
    allen_tests = []

    if sm.efel is not None:
        for k, v in sm.efel.items():
            try:
                at = AllenTest(name=str(k))
                at.set_observation(v)
                at = wrangle_tests(at)
                allen_tests.append(at)
            except:
                pass

    suite = TestSuite(allen_tests, name=str(specimen_id))
    suite.traces = None
    suite.traces = {}
    suite.traces["rh_current"] = sm.rheobase
    suite.traces["vmrh"] = sm.vmrh
    ##
    # Not here deliberate misleading naming
    ##
    suite.traces["vm_soma"] = sm.vm_soma
    suite.model = None
    suite.model = sm
    suite.stim = None
    suite.stim = currents
    return suite, specimen_id
