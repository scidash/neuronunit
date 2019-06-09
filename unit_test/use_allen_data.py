
import matplotlib as mpl
mpl.use('agg')

from allensdk.core.cell_types_cache import CellTypesCache
from allensdk.ephys.extract_cell_features import extract_cell_features
from collections import defaultdict
from neuronunit.optimisation.optimisation_management import inject_rh_and_dont_plot, add_druckmann_properties_to_cells

from allensdk.core.nwb_data_set import NwbDataSet
import pickle

from neuronunit import aibs

#dm_tests = init_dm_tests(value,1.5*value)

files = [324257146, 485909730]
data = []
for f in files:
    data.append(aibs.get_nwb(f))
import pdb; pdb.set_trace()


def a_cell_for_check(stim):
    cells = pickle.load(open("multi_objective_raw.p","rb"))
    dtc = cells['results']['RAW']['Dentate gyrus basket cell']['pf'][0].dtc
    dtc.attrs['dt'] = 0.0001

    (_,times,vm) = inject_rh_and_dont_plot(dtc)
    return (_,times,vm)
# if you ran the examples above, you will have a NWB file here

ctc = CellTypesCache(manifest_file='cell_types/manifest.json')
file_name = 'cell_types/specimen_485909730/ephys.nwb'
specimen_id = 485909730

try:
    data_set = NwbDataSet(file_name)
except:
    pass
# pick a cell to analyze
#specimen_id = 324257146
    # download the ephys data and sweep metadata


def get_nwb(specimen_id = 485909730):
    data_set = ctc.get_ephys_data(specimen_id)
    data_set = NwbDataSet(file_name)

    sweep_numbers = data_set.get_sweep_numbers()
    for sn in sweep_numbers:
        spike_times = data_set.get_spike_times(sn)
        if sum(spike_times):
            sweep_data = data_set.get_sweep(sn)
            vm = sweep_data['response']
            # Two things need to be done.
            # 1. Apply these stimulations to allen models.
            # 2. Apply the feature extraction to optimized models.
            injection = sweep_data['stimulus']
            # sampling rate is in Hz
            sampling_rate = sweep_data['sampling_rate']

            # start/stop indices that exclude the experimental test pulse (if applicable)
            index_range = sweep_data['index_range']

            #import pdb; pdb.set_trace()
            #meta_d = data_set.get_sweep_metadata(sn)
            #print(vm)
            #print(meta_d)
    return data_set


def get_features(specimen_id = 485909730):
    data_set = ctc.get_ephys_data(specimen_id)
    sweeps = ctc.get_ephys_sweeps(specimen_id)
    #import pdb; pdb.set_trace()
    for s in sweeps:
        if s['ramp']:

            print([(k,v) for k,v in s.items()])
        current = {}
        current['amplitude'] = s['stimulus_absolute_amplitude']
        current['duration'] = s['stimulus_duration']
        current['delay'] = s['stimulus_start_time']


    # group the sweeps by stimulus
    sweep_numbers = defaultdict(list)
    for sweep in sweeps:
        sweep_numbers[sweep['stimulus_name']].append(sweep['sweep_number'])

    # calculate features
    cell_features = extract_cell_features(data_set,
                                          sweep_numbers['Ramp'],
                                          sweep_numbers['Short Square'],
                                          sweep_numbers['Long Square'])
