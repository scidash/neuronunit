
class DataTC(object):
    '''
    Data Transport Vessel

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
        self.boolean = False
        self.initiated = False
        self.delta = []
        self.evaluated = False
        self.results = {}
        self.searched = []
        self.searchedd = {}
        self.cached_attrs = {}
        self.backend = None
