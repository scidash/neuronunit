"""NeuronUnit module for interaction with the Allen Brain Insitute
Cell Types database"""
# import logging
# logger = logging.getLogger(name)
# logging.info("test")
import matplotlib as mpl

try:
    mpl.use("agg")
except:
    pass
import matplotlib.pyplot as plt
import shelve
import requests
import numpy as np
import quantities as pq
from allensdk.api.queries.cell_types_api import CellTypesApi
from allensdk.core.cell_types_cache import CellTypesCache
from allensdk.api.queries.glif_api import GlifApi
import os
import pickle
from allensdk.api.queries.biophysical_api import BiophysicalApi
from neuronunit.optimization.data_transport_container import DataTC

# from allensdk.model.glif.glif_neuron import GlifNeuron

from allensdk.core.cell_types_cache import CellTypesCache
from allensdk.ephys.extract_cell_features import extract_cell_features
from collections import defaultdict
from allensdk.core.nwb_data_set import NwbDataSet

from neuronunit import models
from neo.core import AnalogSignal
import quantities as qt
from types import MethodType

from allensdk.ephys.extract_cell_features import extract_cell_features
from collections import defaultdict
from allensdk.core.cell_types_cache import CellTypesCache

import neo
from elephant.spike_train_generation import threshold_detection
from quantities import mV, ms
from numba import jit
import sciunit
import math
import pdb
from allensdk.ephys.extract_cell_features import extract_cell_features


def is_aibs_up():
    """Check whether the AIBS Cell Types Database API is working."""
    url = (
        "http://api.brain-map.org/api/v2/data/query.xml?criteria=model"
        "::Specimen,rma::criteria,[id$eq320654829],rma::include,"
        "ephys_result(well_known_files(well_known_file_type"
        "[name$eqNWBDownload]))"
    )
    request = requests.get(url)
    return request.status_code == 200


def get_observation(dataset_id, kind, cached=True, quiet=False):
    """Get an observation.

    Get an observation of kind 'kind' from the dataset with id 'dataset_id'.
    optionally using the cached value retrieved previously.
    """

    db = shelve.open("aibs-cache") if cached else {}
    identifier = "%d_%s" % (dataset_id, kind)
    if identifier in db:
        print(
            "Getting %s cached data value for from AIBS dataset %s"
            % (kind.title(), dataset_id)
        )
        value = db[identifier]
    else:
        print(
            "Getting %s data value for from AIBS dataset %s"
            % (kind.title(), dataset_id)
        )
        ct = CellTypesApi()
        cmd = ct.get_cell(dataset_id)  # Cell metadata

        if kind == "rheobase":
            if "ephys_features" in cmd:
                value = cmd["ephys_features"][0]["threshold_i_long_square"]  # newer API
            else:
                value = cmd["ef__threshold_i_long_square"]  # older API

            value = np.round(value, 2)  # Round to nearest hundredth of a pA.
            value *= pq.pA  # Apply units.

        else:
            value = cmd[kind]

        db[identifier] = value

    if cached:
        db.close()
    return {"value": value}


def get_value_dict(experiment_params, sweep_ids, kind):
    """Get a dictionary of data values from the experiment.

    A candidate method for replacing 'get_observation'.
    This fix is necessary due to changes in the allensdk.
    Warning: Together with 'get_sp' this method may not properly
    convey the meaning of 'get_observation'.
    """

    if kind == str("rheobase"):
        sp = get_sp(experiment_params, sweep_ids)
        value = sp["stimulus_absolute_amplitude"]
        value = np.round(value, 2)  # Round to nearest hundredth of a pA.
        value *= pq.pA  # Apply units.
        return {"value": value}


"""Auxiliary helper functions for analysis of spiking."""


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return (array[idx], idx)


def inject_square_current(model, current):
    if type(current) is type({}):
        current = float(current["amplitude"])
    data_set = model.data_set
    numbers = data_set.get_sweep_numbers()
    injections = [np.max(data_set.get_sweep(sn)["stimulus"]) for sn in numbers]
    sns = [sn for sn in numbers]
    (nearest, idx) = find_nearest(injections, current)
    index = np.asarray(numbers)[idx]
    sweep_data = data_set.get_sweep(index)
    temp_vm = sweep_data["response"]
    injection = sweep_data["stimulus"]
    sampling_rate = sweep_data["sampling_rate"]
    vm = AnalogSignal(temp_vm, sampling_rate=sampling_rate * qt.Hz, units=qt.V)
    model._vm = vm
    return model._vm


def get_membrane_potential(model):
    return model._vm


def get_spike_train(vm, threshold=0.0 * mV):
    """
    Inputs:
     vm: a neo.core.AnalogSignal corresponding to a membrane potential trace.
     threshold: the value (in mV) above which vm has to cross for there
                to be a spike.  Scalar float.

    Returns:
     a neo.core.SpikeTrain containing the times of spikes.
    """
    spike_train = threshold_detection(vm, threshold=threshold)
    return spike_train


def get_spike_count(model):
    vm = model.get_membrane_potential()
    train = get_spike_train(vm)
    return len(train)


def appropriate_features():
    for s in sweeps:
        if s["ramp"]:
            print([(k, v) for k, v in s.items()])
        current = {}
        current["amplitude"] = s["stimulus_absolute_amplitude"]
        current["duration"] = s["stimulus_duration"]
        current["delay"] = s["stimulus_start_time"]


def get_features(specimen_id=485909730):
    data_set = ctc.get_ephys_data(specimen_id)
    sweeps = ctc.get_ephys_sweeps(specimen_id)

    # group the sweeps by stimulus
    sweep_numbers = defaultdict(list)
    for sweep in sweeps:
        sweep_numbers[sweep["stimulus_name"]].append(sweep["sweep_number"])

    # calculate features
    cell_features = extract_cell_features(
        data_set,
        sweep_numbers["Ramp"],
        sweep_numbers["Short Square"],
        sweep_numbers["Long Square"],
    )


def get_sweep_params(dataset_id, sweep_id):
    """Get sweep parameters.

    Get those corresponding to the sweep with id 'sweep_id' from
    the dataset with id 'dataset_id'.
    """

    ct = CellTypesApi()
    experiment_params = ct.get_ephys_sweeps(dataset_id)
    sp = None
    for sp in experiment_params:
        if sp["id"] == sweep_id:
            sweep_num = sp["sweep_number"]
            if sweep_num is None:
                msg = "Sweep with ID %d not found in dataset with ID %d."
                raise Exception(msg % (sweep_id, dataset_id))
            break
    return sp


def get_sp(experiment_params, sweep_ids):

    """Get sweep parameters.
    A candidate method for replacing 'get_sweep_params'.
    This fix is necessary due to changes in the allensdk.
    Warning: This method may not properly convey the original meaning
    of 'get_sweep_params'.
    """

    sp = None
    for sp in experiment_params:
        for sweep_id in sweep_ids:
            if sp["id"] == sweep_id:
                sweep_num = sp["sweep_number"]
                if sweep_num is None:
                    raise Exception("Sweep with ID %d not found." % sweep_id)
                break
    return sp
