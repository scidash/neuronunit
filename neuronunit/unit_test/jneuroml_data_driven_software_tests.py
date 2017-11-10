"""Tests of NeuronUnit model classes"""




class ReducedModelTestCase(unittest.TestCase):
    """Test instantiation of the reduced model"""

    def setUp(self):
        from neuronunit.models.reduced import ReducedModel
        self.ReducedModel = ReducedModel

    def runTest(self):
        pass # Needed so that Python<3 can access the path attribute.

    @property
    def path(self):
        result = os.path.join(__file__,'..','..','neuronunit',
                              'models','NeuroML2','LEMS_2007One.xml')
        result = os.path.realpath(result)
        return result

    def test_reducedmodel_jneuroml(self):
        import pickle
        with open('opt_run_data','rb') as handle:
            valued = pickle.load(handle)

        dtcpop, pop, pf = valued# = [dtcpop,pop,pf]
        for d in dtcpop:
            model = self.ReducedModel(self.path, backend='jNeuroML')

    #@unittest.skip("Ignoring NEURON until we make it an install requirement")#If(OSX,"NEURON unreliable on OSX")
    def test_reducedmodel_neuron(self):
        model = self.ReducedModel(self.path, backend='NEURON')


if __name__ == '__main__':
    unittest.main()
