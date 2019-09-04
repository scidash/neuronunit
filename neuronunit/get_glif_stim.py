import allensdk.core.json_utilities as json_utilities
import pickle
neuronal_model_id = 566302806
'''
# download model metadata
glif_api = GlifApi()
nm = glif_api.get_neuronal_models_by_id([neuronal_model_id])[0]


# download information about the cell
ctc = CellTypesCache()
ctc.get_ephys_data(nm['specimen_id'], file_name='stimulus.nwb')
ctc.get_ephys_sweeps(nm['specimen_id'], file_name='ephys_sweeps.json')
'''
ephys_sweeps = json_utilities.read('ephys_sweeps.json')
ephys_file_name = 'stimulus.nwb'

#neuron = GlifNeuron.from_dict(neuron_config)

sweep_numbers = [ s['sweep_number'] for s in ephys_sweeps if s['stimulus_units'] == 'Amps' ]

#snumber = [ s for s in ephys_sweeps if s['stimulus_units'] == 'Amps' if s['num_spikes']>=1]
stimulus = [ s for s in ephys_sweeps if s['stimulus_units'] == 'Amps' \
 if s['num_spikes'] != None \
 if s['stimulus_name']!='Ramp' and s['stimulus_name']!='Short Square']

amplitudes = [ s['stimulus_absolute_amplitude'] for s in stimulus ]
durations = [ s['stimulus_duration'] for s in stimulus ]

import pdb
pdb.set_trace()
expeceted_spikes = [ s['num_spikes'] for s in stimulus ]
#durations = [ s['stimulus_absolute_amplitude'] for s in stimulus ]
delays = [ s['stimulus_start_time'] for s in stimulus ]
sn = [ s['sweep_number'] for s in stimulus ]
make_stim_waves = {}
for i,j in enumerate(sn):
    make_stim_waves[j] = {}
    make_stim_waves[j]['amplitude'] = amplitudes[i]
    make_stim_waves[j]['delay'] = delays[i]
    make_stim_waves[j]['durations'] = durations[i]
    make_stim_waves[j]['expeceted_spikes'] = expeceted_spikes[i]

pickle.dump(make_stim_waves,open('waves.p','wb'))
print(make_stim_waves)
#sweep_numbers = sweep_numbers[:1] # for the sake of a speedy example, just run the first one
#simulate_neuron(neuron, sweep_numbers, ephys_file_name, ephys_file_name, 0.05)
