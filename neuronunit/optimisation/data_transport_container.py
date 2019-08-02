import numpy as np
#from neuronunit.optimisation.optimisation_management import mint_generic_model
#from sciunit.models.runnable import RunnableModel

class DataTC(object):
    '''
    Data Transport Container

    This Object class serves as a data type for storing rheobase search
    attributes and apriori model parameters,
    with the distinction that unlike the NEURON model this class
    can be cheaply transported across HOSTS/CPUs
    '''
    def __init__(self):
        self.lookup = {}
        self.rheobase = None
        self.previous = 0
        self.run_number = 0
        self.attrs = None
        self.steps = None
        self.name = None
        self.results = None
        self.fitness = None
        self.score = None
        self.scores = None

        self.boolean = False
        self.initiated = False
        self.delta = []
        self.evaluated = False
        self.results = {}
        self.searched = []
        self.searchedd = {}
        self.cached_attrs = {}
        self.backend = None
        self.summed = None
        self.constants = None

    def get_ss(self):
        # get summed score
        if self.scores is not None:
            if len(self.scores) == 1:
                self.summed = list(self.scores.values())[0]
            else:
                self.summed = np.sum(list(self.scores.values()))
        else:
            self.summed = None
        return self.summed

    def add_constant(self):
        if self.constants is not None:
            self.attrs.update(self.constants)
        return #self.attrs
    def dtc_to_model(self):
        from neuronunit.models import VeryReducedModel

        #model = RunnableModel(str(self.backend),backend=self.backend,attrs=self.attrs)
        #model = RunnableModel(str(self.backend),backend=(self.backend, {'DTC':self}))
        model = VeryReducedModel(name='vanilla',backend=(self.backend, {'DTC':self}))#, {'DTC':dtc}))
        # If  test taking data, and objects are present (observations etc).
        # Take the rheobase test and store it in the data transport container.
        if not hasattr(self,'scores'):
            self.scores = None
        if type(self.scores) is type(None):
            self.scores = {}
        model.attrs=self.attrs
        model.scores=self.scores
        # model = mint_generic_model(dtc.backend)
        # model.attrs = self.attrs
        return model
