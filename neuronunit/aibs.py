"""NeuronUnit module for interaction with the Allen Brain Insitute
Cell Types database"""

import shelve
import requests

import numpy as np
import quantities as pq
from allensdk.api.queries.cell_types_api import CellTypesApi

from allensdk.api.queries.cell_types_api import CellTypesApi
from allensdk.core.cell_types_cache import CellTypesCache
from allensdk.api.queries.glif_api import GlifApi
import os
import pickle
#import allensdk.model.biophysical.runner #manifest.json')
from allensdk.api.queries.biophysical_api import BiophysicalApi
## Need this import but it fails because of python2 formatted strings.
#from allensdk.model.biophysical import runner
from neuronunit.optimisation.optimisation_management import add_dm_properties_to_cells
from neuronunit.optimisation.optimisation_management import mint_generic_model, dtc_to_rheo
from neuronunit.optimisation.data_transport_container import DataTC
from allensdk.model.glif.glif_neuron import GlifNeuron
import dask.bag as db
import multiprocessing


def is_aibs_up():
    url = ("http://api.brain-map.org/api/v2/data/query.xml?criteria=model"
           "::Specimen,rma::criteria,[id$eq320654829],rma::include,ephys_result"
           "(well_known_files(well_known_file_type[name$eqNWBDownload]))")
    request = requests.get(url)
    return request.status_code == 200


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

    if kind == str('rheobase'):
        sp = get_sp(experiment_params,sweep_ids)
        value = sp['stimulus_absolute_amplitude']
        value = np.round(value,2) # Round to nearest hundredth of a pA.
        value *= pq.pA # Apply units.
        return {'value': value}



def allen_morph_model(description):
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

    #bp.cache_stimulus = False # change to False to not download the large stimulus NWB file
    #neuronal_model_id = 472451419    # get this from the web site as above
    for cell in cells:
        try:
            bp.cache_data(cell['id'], working_directory='.')
            os.subprocess('nrnivmodl ./modfiles')   # compile the model (only needs to be done once)
            ## Need this import but it fails because of python2 formatted strings.
            #from allensdk.model.biophysical import runner
            print(runner)

            allen_morph_model(description)
        except:
            pass

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

def run_glif_to_druckmanns():
    try:
        with open('gcm.p','rb') as f: model_params = pickle.load(f)
    except:
        os.system('wget https://osf.io/k7ryf/download')
        os.system('mv download gcm.p')
        with open('gcm.p','rb') as f: model_params = pickle.load(f)

    flat_iter = [ mp.pop(list(mp.keys())[0]) for mp in model_params ]
    dtcpop = []
    cnt = 0
    # Todo make this line interruptable/cachable, as its a big job.
    # count can be pickle loaded to check where left off
    for f in flat_iter:
        dtcpop.append(to_map(f))
        cnt += 1
    # dtcpop = map(to_map,flat_iter)
    self.dtcpop,dm_properties = add_dm_properties_to_cells(dtcpop)
    return (self.dtcpop,dm_properties)

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
