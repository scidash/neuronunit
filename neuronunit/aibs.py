"""NeuronUnit module for interaction with the Allen Brain Insitute
Cell Types database"""
#import logging
#logger = logging.getLogger(name)
#logging.info("test")
import matplotlib as mpl
try:
    mpl.use('agg')
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
## Need this import but it fails because of python2 formatted strings.
#from neuronunit.optimisation.optimisation_management import add_dm_properties_to_cells
#from neuronunit.optimisation.optimization_management import mint_generic_model, dtc_to_rheo, split_list
from neuronunit.optimisation.data_transport_container import DataTC
from allensdk.model.glif.glif_neuron import GlifNeuron
import dask.bag as db
import multiprocessing

from allensdk.core.cell_types_cache import CellTypesCache
from allensdk.ephys.extract_cell_features import extract_cell_features
from collections import defaultdict
#from neuronunit.optimisation.optimization_management import inject_rh_and_dont_plot

from allensdk.core.nwb_data_set import NwbDataSet
import pickle
#from neuronunit.optimisation.optimisation_management import init_dm_tests

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
from itertools import repeat
##from sklearn.cross_decomposition import CCA
'''
def a_cell_for_check(stim):
    cells = pickle.load(open("multi_objective_raw.p","rb"))
    dtc = cells['results']['RAW']['Dentate gyrus basket cell']['pf'][0].dtc
    dtc.attrs['dt'] = 0.0001

    (_,times,vm) = inject_rh_and_dont_plot(dtc)
    return (_,times,vm)
'''
# if you ran the examples above, you will have a NWB file here

# pick a cell to analyze
#specimen_id = 324257146
    # download the ephys data and s

#all_models = json.loads(url.read().decode())

import logging
logging.info("test")

def get_all(Model_ID = str('NMLNT001592')):
    if Model_ID == None:
        try:
            url = str("https://www.neuroml-db.org/api/models")
            all_models = requests.get(url)
            all_models = json.loads(all_models.text)
            print(all_models)
            for d in all_models:
                print(d.keys())
            for d in all_models[0]:
                print(d['Model_ID'],d['Directory_Path'])
                url = str('https://www.neuroml-db.org/GetModelZip?modelID=')+str(d['Model_ID'])+str('&version=NeuroML')
                urllib.request.urlretrieve(url,Model_ID)
                os.system(str('unzip *')+str(d['Model_ID'])+str('*'))
                os.system(str('cd *')+str(d['Model_ID'])+str('*'))
                os.system(str('pynml hhneuron.cell.nml -neuron'))
            return data

        except:
            pass
    else:
        d = {}
        d['Model_ID'] = Model_ID
        url = "https://www.neuroml-db.org/GetModelZip?modelID=NMLNT001592&version=NeuroML"
        urllib.request.urlretrieve(url,Model_ID)
        url = "https://www.neuroml-db.org/GetModelZip?modelID=NMLNT001592&version=NeuroML"
        os.system('wget '+str(url))
        os.system(str('unzip ')+str(d['Model_ID'])+str('*'))
        os.system(str('cd *')+str(d['Model_ID'])+str('*'))

        os.system(str('pynml hhneuron.cell.nml -neuron'))


def run_cell():
    from neuron import h
    h.load_file('hhneuron.hoc')
    cell = h.hhneuron
    d = {}
    d['Model_ID'] = str('NT001592')
    with urllib.request.urlopen(str('https://www.neuroml-db.org/api/model?id=')+str(d['Model_ID'])) as url:
        data_on_model = json.loads(url.read().decode())

def is_aibs_up():
    """Check whether the AIBS Cell Types Database API is working."""
    url = ("http://api.brain-map.org/api/v2/data/query.xml?criteria=model"
           "::Specimen,rma::criteria,[id$eq320654829],rma::include,"
           "ephys_result(well_known_files(well_known_file_type"
           "[name$eqNWBDownload]))")
    request = requests.get(url)
    return request.status_code == 200


def find_nearest(array, value):
    #value = float(value)
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return (array[idx], idx)



def inject_square_current(model,current):
    if type(current) is type({}):
        current = float(current['amplitude'])
    data_set = model.data_set
    numbers = data_set.get_sweep_numbers()
    injections = [ np.max(data_set.get_sweep(sn)['stimulus']) for sn in numbers ]
    sns = [ sn for sn in numbers]
    (nearest,idx) = find_nearest(injections,current)
    index = np.asarray(numbers)[idx]
    sweep_data = data_set.get_sweep(index)
    temp_vm = sweep_data['response']
    injection = sweep_data['stimulus']
    sampling_rate = sweep_data['sampling_rate']
    vm = AnalogSignal(temp_vm,sampling_rate=sampling_rate*qt.Hz,units=qt.V)
    model._vm = vm
    return model._vm

def get_membrane_potential(model):
    return model._vm

"""Auxiliary helper functions for analysis of spiking."""

def get_spike_train(vm, threshold=0.0*mV):
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


def dm_map(compound):
    (sm, dm) = compound
    return dm.generate_prediction(sm)


def get_nwb(specimen_id = 324257146):
    file_name = 'cell_types/specimen_'+str(specimen_id)+'/ephys.nwb'
    data_set = NwbDataSet(file_name)

    try:
        sweep_numbers = data_set.get_sweep_numbers()
    except:
        return
    try:
        sweeps = ctc.get_ephys_sweeps(specimen_id)
        for sn in sweep_numbers:
            spike_times = data_set.get_spike_times(sn)
            sweep_data = data_set.get_sweep(sn)
    except:
        ctc = CellTypesCache(manifest_file='cell_types/manifest.json')
        data_set = ctc.get_ephys_data(specimen_id)

    sweeps = ctc.get_ephys_sweeps(specimen_id)

    sweep_numbers = defaultdict(list)
    for sweep in sweeps:
        sweep_numbers[sweep['stimulus_name']].append(sweep['sweep_number'])


    cell_features = extract_cell_features(data_set, sweep_numbers['Ramp'],sweep_numbers['Short Square'],sweep_numbers['Long Square'])

    sweep_numbers = data_set.get_sweep_numbers()
    smallest_multi = 1000
    all_currents = []
    for sn in sweep_numbers:
        spike_times = data_set.get_spike_times(sn)
        sweep_data = data_set.get_sweep(sn)

        if len(spike_times) == 1:
            inj_rheobase = np.max(sweep_data['stimulus'])

        if len(spike_times) < smallest_multi and len(spike_times) > 1:
            smallest_multi = len(spike_times)
            inj_multi_spike = np.max(sweep_data['stimulus'])
            temp_vm = sweep_data['response']
        val = np.max(sweep_data['stimulus'])#*qt.pA
        all_currents.append(val)
    dmrheobase15 = (1.5*inj_rheobase)#cell_features['long_squares']['rheobase_i'])#*qt.pA
    (nearest_allen15,idx_nearest_allen) = find_nearest(all_currents,dmrheobase15)
    dmrheobase30 = (3.0*inj_rheobase)#cell_features['long_squares']['rheobase_i'])#*qt.pA
    (nearest_allen30,idx_nearest_allen) = find_nearest(all_currents,dmrheobase30)

    print(nearest_allen15,nearest_allen30,inj_multi_spike)
    #import pdb
    #pdb.set_trace()
    #print('how close are these two \n\n\n\n\n\n ?', nearest_allen15,inj_multi_spike)
    if inj_multi_spike < nearest_allen15 and inj_rheobase!=nearest_allen15:# != inj_rheobase:
        pass
        #dm_tests = init_dm_tests(inj_rheobase,nearest_allen15)
    else:
        pass
    dm_tests = init_dm_tests(inj_rheobase,inj_multi_spike)

    # Two things need to be done.
    # 1. Apply these stimulations to allen models.
    # 2. Apply the feature extraction to optimized models.
    injection = sweep_data['stimulus']
    # sampling rate is in Hz
    sampling_rate = sweep_data['sampling_rate']
    vm = AnalogSignal(temp_vm,sampling_rate=sampling_rate*qt.Hz,units=qt.V)
    #plt.plot(vm.times,vm.magnitude)
    #plt.savefig('too_small_debug.png')

    sm = models.StaticModel(vm)
    sm.data_set = data_set
    sm.inject_square_current = MethodType(inject_square_current,sm)
    sm.get_membrane_potential = MethodType(get_membrane_potential,sm)
    sm.get_spike_count = MethodType(get_spike_count,sm)
    # these lines are functional
    sm.inject_square_current(inj_rheobase)
    dm_tests[0].generate_prediction(sm)
    #sm.inject_square_current(inj_mutli_spike)
    compound = list(zip(repeat(sm),dm_tests))
    bag = db.from_sequence(compound,npartitions=8)
    preds = list(bag.map(dm_map).compute())
    names = [ d.name for d in dm_tests ]
    preds = list(zip(preds,names))
    spiking_sweeps = cell_features['long_squares']['spiking_sweeps'][0]
    multi_spike_features = cell_features['long_squares']['hero_sweep']
    biophysics = cell_features['long_squares']
    shapes =  cell_features['long_squares']['spiking_sweeps'][0]['spikes'][0]
    #import pdb; pdb.set_trace()
    #print(spiking_sweeps)
    #print(biophysics)
    #cca = CCA(n_components=1)
    #cca.fit(X, Y)

    #X_c, Y_c = cca.transform(X, Y)

    everything = (preds,cell_features)
    return everything



def appropriate_features():
    for s in sweeps:
        if s['ramp']:
            print([(k,v) for k,v in s.items()])
        current = {}
        current['amplitude'] = s['stimulus_absolute_amplitude']
        current['duration'] = s['stimulus_duration']
        current['delay'] = s['stimulus_start_time']

def get_features(specimen_id = 485909730):
    data_set = ctc.get_ephys_data(specimen_id)
    sweeps = ctc.get_ephys_sweeps(specimen_id)




    # group the sweeps by stimulus
    sweep_numbers = defaultdict(list)
    for sweep in sweeps:
        sweep_numbers[sweep['stimulus_name']].append(sweep['sweep_number'])

    # calculate features
    cell_features = extract_cell_features(data_set,
                                          sweep_numbers['Ramp'],
                                          sweep_numbers['Short Square'],
                                          sweep_numbers['Long Square'])

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
            if 'ephys_features' in cmd:
                value = cmd['ephys_features'][0]['threshold_i_long_square'] # newer API
            else:
                value = cmd['ef__threshold_i_long_square'] # older API

            value = np.round(value, 2)  # Round to nearest hundredth of a pA.
            value *= pq.pA  # Apply units.

        else:
            value = cmd[kind]

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



def allen_morph_model(description):
    from allensdk.model.biophysical.utils import Utils
    from allensdk.model.biophys_sim import neuron
    utils = Utils.create_utils(description)
    h = utils.h

    #The next step is to get the path of the morphology file and pass it to NEURON.

    # configure model
    manifest = description.manifest
    morphology_path = description.manifest.get_path('MORPHOLOGY')
    utils.generate_morphology(morphology_path.encode('ascii', 'ignore'))
    utils.load_cell_parameters()



def run_all_cell_bio_configs():
    try:
        with open('all_allen_cells.p','rb') as f:
            cells = pickle.load(f)

    except:
        ctc = CellTypesCache(manifest_file='cell_types/manifest.json')
        cells = ctc.get_cells()
        with open('all_allen_cells.p','wb') as f:
            pickle.dump(cells,f)
    bp = BiophysicalApi()
    from bmtk.simulator.utils import config
    from allensdk.model.biophysical import runner
    from allensdk.model.biophysical.utils import Utils


    #bp.cache_stimulus = False # change to False to not download the large stimulus NWB file
    #neuronal_model_id = 472451419    # get this from the web site as above
    for description in cells[0:5]:
        config = config.from_dict(description)
        '''
                # configure NEURON -- this will infer model type (perisomatic vs. all-active)
        utils = Utils.create_utils(description)
        h = utils.h

        The next step is to get the path of the morphology file and pass it to NEURON.

        # configure model
        manifest = description.manifest
        morphology_path = description.manifest.get_path('MORPHOLOGY')
        utils.generate_morphology(morphology_path.encode('ascii', 'ignore'))
        utils.load_cell_parameters()

        Then read the stimulus and recording configuration and configure NEURON

        # configure stimulus and recording
        stimulus_path = description.manifest.get_path('stimulus_path')
        nwb_out_path = manifest.get_path("output")
        output = NwbDataSet(nwb_out_path)
        run_params = description.data['runs'][0]
        sweeps = run_params['sweeps']
        junction_potential = description.data['fitting'][0]['junction_potential']
        mV = 1.0e-3

        Loop through the stimulus sweeps and write the output.

        # run sweeps
        for sweep in sweeps:
            utils.setup_iclamp(stimulus_path, sweep=sweep)
            vec = utils.record_values()

            h.finitialize()
            h.run()

            # write to an NWB File
            output_data = (numpy.array(vec['v']) - junction_potential) * mV
            output.set_sweep(sweep, None, output_data)


        import pdb; pdb.set_trace()
        '''
        #try:
        bp.cache_data(cell['id'], working_directory='.')
        os.subprocess('nrnivmodl ./modfiles')   # compile the model (only needs to be done once)
        ## Need this import but it fails because of python2 formatted strings.
        print(runner)

        allen_morph_model(description)
        runner.load_description(description)
        runner.run(description, sweeps=None, procs=6)
        #except:
        #    pass

    return cells



def to_map(params):
    '''
    find rheobase for each model
    '''
    dtc = DataTC()
    b = str('GLIF')
    dtc.attrs = params
    dtc.backend = b
    dtc = dtc_to_rheo(dtc)
    return dtc

def run_glif_to_rheobase():
    try:
        with open('gcm.p','rb') as f: model_params = pickle.load(f)
    except:
        os.system('wget https://osf.io/k7ryf/download')
        os.system('mv download gcm.p')
        with open('gcm.p','rb') as f: model_params = pickle.load(f)

    flat_iter = [ mp.pop(list(mp.keys())[0]) for mp in model_params ]

    dtcpop = []
    #cnt = 0
    # Todo make this line interruptable/cachable, as its a big job.
    # count can be pickle loaded to check where left off
    dtcpop = list(map(to_map,flat_iter))
    return dtcpop
def run_glif_to_druckmanns():
    try:
        with open('gcm.p','rb') as f: model_params = pickle.load(f)
    except:
        os.system('wget https://osf.io/k7ryf/download')
        os.system('mv download gcm.p')
        with open('gcm.p','rb') as f: model_params = pickle.load(f)

    flat_iter = [ mp.pop(list(mp.keys())[0]) for mp in model_params ]
    # f1,f2 = split_list(flat_iter)
    # f_1_2 = split_list(f1)
    # flat_iter = flat_iter[0:2]

    dtcpop = []
    cnt = 0
    # Todo make this line interruptable/cachable, as its a big job.
    # count can be pickle loaded to check where left off
    for f in flat_iter:
        dtcpop.append(to_map(f))
        cnt += 1
    dtcpop,dm_properties = add_dm_properties_to_cells(dtcpop)
    return (dtcpop,dm_properties)

def construct_data_frame(arg):
    self.dtcpop,dm_properties = run_glif_to_druckmanns()
    # populate the data frame
    # make dummy tests:
    tests = init_dm_tests(self.dtcpop[0].rheobase,self.dtcpop[0].rheobase*1.5)

    df = pd.DataFrame(columns=tests)
    df.loc[0] = [ d['mean'] for d in dm_properties ]

    df.head()

    for t in tests:
        t.name = t.name.replace(" ", "")
    df1 = pd.DataFrame(columns=tests)
    df1.loc[-1] = [ d['mean'] for d in dm_properties ]

    return (df,df1)



def boot_strap_all_glif_configs():
    '''
    Mass download all the glif model parameters
    '''
    gapi = GlifApi()

    cells = gapi.get_neuronal_models() # this returns a list of cells, each containing a list of models
    models = [ nm for c in cells for nm in c['neuronal_models'] ] # flatten to just a list of models
    model_params = []
    # this will take awhile!
    # returns a dictionary of params, indexed on model id

    try:
        with open('last_index.p','rb') as f:
            index = pickle.load(f)
    except:
        index = 0
    until_done = len(models[index:-1])
    cnt = 0
    while cnt <until_done-1:
        for i,model in enumerate(models[index:-1]):
            until_done = len(models[index:-1])
            try:
                print(i,model)
                # keep trying to download more and more.
                model_params.append(gapi.get_neuron_configs([model['id']])) # download the first five
                print('progress',len(models),i)
                with open('gcm.p','wb') as f:
                    pickle.dump(model_params,f)
                with open('last_index.p','wb') as f:
                    pickle.dump(i,f)
            except:
                with open('last_index.p','rb') as f:
                    index = pickle.load(f)
            cnt+=1


    with open('gcm.p','rb') as f:
        model_params = pickle.load(f)
    flat_iter = [ mp.pop(list(mp.keys())[0]) for mp in model_params ]
    new_flat_iter = [(k,v) for fi in flat_iter for k,v in fi.items() ]
    glif_range = {}
    for k,v in new_flat_iter:
        glif_range[k] = [v,v]
    for k,v in new_flat_iter:
        if isinstance(v,dict) and not isinstance(v,type(None)) and not isinstance(v,list):
            if v<glif_range[k][0]:
                glif_range[k][0] = v
            if v>glif_range[k][1]:
                glif_range[k][1] = v
        else:
            glif_range[k] = v
    with open('glif_range.p','wb') as f:
        pickle.dump(glif_range,f)
    return glif_range


def get_all_glif_configs():
    '''
    Find the boundaries of the GLIF cell parameter space, by exhaustively sampling all GLIF cells
    '''
    try:
        with open('gcm.p','rb') as f:
            model_params = pickle.load(f)
    except:
        #os.sytem('wget https://osf.io/fzxsn/download')
        #with open('gcm.p','rb') as f:
        #        model_params = pickle.load(f)

        flat_iter = [ mp.pop(list(mp.keys())[0]) for mp in model_params ]
        new_flat_iter = [(k,v) for fi in flat_iter for k,v in fi.items() ]

        #flat_iter = list((k,v) for p in model_params for k,v in p.values())
        glif_range = {}
        for k,v in new_flat_iter:
            glif_range[k] = [v,v]
        for k,v in new_flat_iter:
            if type(v) is not type({'dict':1}) and type(v) is not type(None):        #import pdb; pdb.set_trace()
                    if v<glif_range[k][0]:
                        glif_range[k][0] = v
                    if v>glif_range[k][1]:
                        glif_range[k][1] = v
            else:
                glif_range[k] = v
            with open('glif_range.p','wb') as f: pickle.dump(glif_range,f)

    #except:
