"""NeuronUnit module for interaction with the Allen Brain Insitute 
Cell Types database"""

import shelve
import numpy as np
import quantities as pq
from allensdk.api.queries.cell_types_api import CellTypesApi
from allensdk.ephys.extract_cell_features import get_square_stim_characteristics#,\
                                                 #get_sweep_from_nwb
    
def get_sweep_params(dataset_id, sweep_id):
    """
    Gets sweep parameters corresponding to the sweep with id 'sweep_id' from
    the dataset with id 'dataset_id'.
    """

    ct = CellTypesApi()
    experiment_params = ct.get_ephys_sweeps(dataset_id)
    sp = None
    for sp in experiment_params:
        if sp['id']==sweep_id:
            sweep_num = sp['sweep_number']
            if sweep_num is None:
                raise Exception('Sweep with ID %d not found in dataset with ID %d.' % (sweep_id, dataset_id))
            break
    return sp


def get_sp(experiment_params, sweep_ids):
    """
    A candidate method for replacing 'get_sweep_params'.
    This fix is necessary due to changes in the allensdk.
    Warning: This method may not properly convey the original meaning 
    of 'get_sweep_params'.
    """
    
    sp = None
    for sp in experiment_params:
        for sweep_id in sweep_ids:
            if sp['id']==sweep_id:
                sweep_num = sp['sweep_number']
                if sweep_num is None:
                    raise Exception('Sweep with ID %d not found.' % sweep_id)
                break
    return sp


def get_observation(dataset_id, kind, cached=True, quiet=False):
    """
    Gets an observation of kind 'kind' from the dataset with id 'dataset_id', 
    optionally using the cached value retrieved previously.
    """
    
    db = shelve.open('aibs-cache') if cached else {}
    identifier = '%d_%s' % (dataset_id,kind)
    if identifier in db:
        print("Getting %s cached data value for from AIBS dataset %s" \
              % (kind.title(),dataset_id))
        value = db[identifier]
    else:
        print("Getting %s data value for from AIBS dataset %s" \
              % (kind.title(),dataset_id))    
        ct = CellTypesApi()
        cmd = ct.get_cell(dataset_id) # Cell metadata
 
        if kind == 'rheobase':
            sweep_id = cmd['ephys_features'][0]['rheobase_sweep_id']
        sp = get_sweep_params(dataset_id, sweep_id)
        if kind == 'rheobase':
            value = sp['stimulus_absolute_amplitude']
            value = np.round(value,2) # Round to nearest hundredth of a pA.
            value *= pq.pA # Apply units.
        db[identifier] = value
    
    if cached:
        db.close()
    return {'value': value}

       
def get_value_dict(experiment_params, sweep_ids, kind):
    """
    A candidate method for replacing 'get_observation'.
    This fix is necessary due to changes in the allensdk.
    Warning: Together with 'get_sp' this method may not properly 
    convey the meaning of 'get_observation'.
    """

def get_sp(experiment_params,sweep_ids):
    '''
    get sweep parameter. A candidate method for replacing get_sweep_params.
    This fix is necessary due to changes in the allensdk however:
    Warning. This method may not properly convey the meaning of get_sweep_params
    '''
    ct = CellTypesApi()
    for sp in experiment_params:
       for sweep_id in sweep_ids:
          if sp['id']==sweep_id:
              sweep_num = sp['sweep_number']
              break
    if sweep_num is None:
        raise Exception('Sweep with ID %d not found in dataset with ID %d.' % (sweep_id, dataset_id))
    return sp

       
def get_value_dict(experiment_params,sweep_ids,kind,cached=True):
    '''
    A candidate method for replacing get_observation.
    This fix is necessary due to changes in the allensdk however:
    Warning. Togethor with get_sp this method may not properly convey the meaning of get_observation
    '''
    ct = CellTypesApi()
    if kind == str('rheobase'):
        sp = get_sp(experiment_params,sweep_ids)
        value = sp['stimulus_absolute_amplitude']
        value = np.round(value,2) # Round to nearest hundredth of a pA.
        value *= pq.pA # Apply units.  
        return {'value': value}           
    
    
 
    
        

       
                  
    
    
    
