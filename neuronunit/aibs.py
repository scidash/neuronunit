import numpy as np
import quantities as pq
from allensdk.api.queries.cell_types_api import CellTypesApi
from allensdk.ephys.extract_cell_features import get_square_stim_characteristics,\
                                                 get_sweep_from_nwb
    
def get_sweep_params(dataset_id,sweep_id):
    ct = CellTypesApi()
    experiment_params = ct.get_ephys_sweeps(dataset_id)
    for sp in experiment_params:
        if sp['id']==sweep_id:
            sweep_num = sp['sweep_number']
            break
    if sweep_num is None:
        raise Exception('Sweep with ID %d not found in dataset with ID %d.' % (sweep_id, dataset_id))
    return sp
    
def get_observation(dataset_id,kind,cached=True):
    ct = CellTypesApi()
    cmd = ct.get_cell(dataset_id) # Cell metadata
    
    sweep_num = None
    if kind == 'rheobase':
        sweep_id = cmd['ephys_features'][0]['rheobase_sweep_id']
    sp = get_sweep_params(dataset_id, sweep_id)
    if kind == 'rheobase':
        value = sp['stimulus_absolute_amplitude']
        value = np.round(value,2) # Round to nearest hundredth of a pA.
        value *= pq.pA # Apply units.  
    return {'value': value}
    
