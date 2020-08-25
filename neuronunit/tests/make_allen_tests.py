from neuronunit.tests.base import VmTest
import pickle
import numpy as np
from allensdk.core.cell_types_cache import CellTypesCache


class AllenTest(VmTest):
    def __init__(self,
                 observation={'mean': None, 'std': None},
                 name='generic_allen',
                 prediction={'mean': None, 'std': None},
                 **params):
        #super(self,VmTest).__init__()
        super(AllenTest, self).__init__(observation, name, **params)

    #required_capabilities = (scap.Runnable, ncap.ProducesMembranePotential,)
    name = ''

    # units = pass #
    # ephysprop_name = ''
    aliases = ''
    '''
    def generate_prediction(self):
        if self.GENERIC:
            return {'mean':prediction['mean'],'std':prediction['std']}
        else:
            dtc,ephys = allen_wave_predictions(dtc,thirty=True)
            dtc,ephys = allen_wave_predictions(dtc,thirty=False)
            self.ephys
            #import pdb
            #pdb.set_trace()
    '''

    def compute_params(self):
        self.params['t_max'] = (self.params['delay'] +
                               self.params['duration'] +
                               self.params['padding'])


    def set_observation(self,observation):
        self.observation = {}
        self.observation['mean'] = observation
        self.observation['std'] = observation


    def set_prediction(self,prediction):
        self.prediction = {}
        self.prediction['mean'] = prediction
        self.prediction['std'] = prediction

df = pickle.load(open("onefive_df.pkl","rb"))    
allen_indexs = [ i for i in df.index if str("NML") not in str(i)]
ctc = CellTypesCache(manifest_file='manifest.json')
features = ctc.get_ephys_features() 
'''
for i in allen_indexs:
    for f in features[0].keys():
        for c in df.columns:
            
            at = AllenTest()
            if str(f) in str(c):# and str(f) == str(c):
                print(i,f,c)
                try:
                    temp = np.mean(df.loc[i,c])
                except:
                    temp = df.loc[i,f]
                print(temp)

                at.set_observation(temp)
                print(at)
'''