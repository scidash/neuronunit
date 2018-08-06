"""NeuronUnit module for interaction with the AIBS Cell Types Database."""

import shelve
import requests

import numpy as np
import quantities as pq
from allensdk.api.queries.cell_types_api import CellTypesApi


def is_aibs_up():
    """Check whether the AIBS Cell Types Database API is working."""
    url = ("http://api.brain-map.org/api/v2/data/query.xml?criteria=model"
           "::Specimen,rma::criteria,[id$eq320654829],rma::include,"
           "ephys_result(well_known_files(well_known_file_type"
           "[name$eqNWBDownload]))")
    request = requests.get(url)
    return request.status_code == 200


def get_sweep_params(dataset_id, sweep_id):
    """Get sweep parameters.

    Get those corresponding to the sweep with id 'sweep_id' from
    the dataset with id 'dataset_id'.
    """
    ct = CellTypesApi()
    experiment_params = ct.get_ephys_sweeps(dataset_id)
    sp = None
    for sp in experiment_params:
        if sp['id'] == sweep_id:
            sweep_num = sp['sweep_number']
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
            if sp['id'] == sweep_id:
                sweep_num = sp['sweep_number']
                if sweep_num is None:
                    raise Exception('Sweep with ID %d not found.' % sweep_id)
                break
    return sp


def get_observation(dataset_id, kind, cached=True, quiet=False):
    """Get an observation.

    Get an observation of kind 'kind' from the dataset with id 'dataset_id'.
    optionally using the cached value retrieved previously.
    """
    db = shelve.open('aibs-cache') if cached else {}
    identifier = '%d_%s' % (dataset_id, kind)
    if identifier in db:
        print("Getting %s cached data value for from AIBS dataset %s"
              % (kind.title(), dataset_id))
        value = db[identifier]
    else:
        print("Getting %s data value for from AIBS dataset %s"
              % (kind.title(), dataset_id))
        ct = CellTypesApi()
        cmd = ct.get_cell(dataset_id)  # Cell metadata
        if kind == 'rheobase':
            sweep_id = cmd['ephys_features'][0]['rheobase_sweep_id']
        sp = get_sweep_params(dataset_id, sweep_id)
        if kind == 'rheobase':
            value = sp['stimulus_absolute_amplitude']
            value = np.round(value, 2)  # Round to nearest hundredth of a pA.
            value *= pq.pA  # Apply units.
        db[identifier] = value

    if cached:
        db.close()
    return {'value': value}


def get_value_dict(experiment_params, sweep_ids, kind):
    """Get a dictionary of data values from the experiment.

    A candidate method for replacing 'get_observation'.
    This fix is necessary due to changes in the allensdk.
    Warning: Together with 'get_sp' this method may not properly
    convey the meaning of 'get_observation'.
    """
    if kind == str('rheobase'):
        sp = get_sp(experiment_params, sweep_ids)
        value = sp['stimulus_absolute_amplitude']
        value = np.round(value, 2)  # Round to nearest hundredth of a pA.
        value *= pq.pA  # Apply units.
        return {'value': value}
