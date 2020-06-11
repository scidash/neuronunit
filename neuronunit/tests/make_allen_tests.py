from neuronunit.tests 


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
    def generate_prediction(self):
        if self.GENERIC:
            return {'mean':prediction['mean'],'std':prediction['std']}
        else:
            dtc,ephys = allen_wave_predictions(dtc,thirty=True)
            dtc,ephys = allen_wave_predictions(dtc,thirty=False)
            self.ephys
            #import pdb
            #pdb.set_trace()


    def compute_params(self):
        self.params['t_max'] = (self.params['delay'] +
                               self.params['duration'] +
                               self.params['padding'])


    def set_observation(self,prediction):
        self.observation = {}
        self.observation['mean'] = observation
        self.observation['std'] = observation


    def set_prediction(self,prediction):
        self.prediction = {}
        self.prediction['mean'] = prediction
        self.prediction['std'] = prediction
    