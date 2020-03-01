""""Tests of external resources NeuronUnit may access"""


from .base import *


class NeuroElectroTestCase(unittest.TestCase):
    """Testing retrieval of data from NeuroElectro.org"""

    @unittest.skipUnless(neuroelectro.is_neuroelectro_up(), 
                         "NeuroElectro URL is not responsive")
    def test_neuroelectro(self):
        x = neuroelectro.NeuroElectroDataMap()
        x.set_neuron(id=72)
        x.set_ephysprop(id=2)
        x.set_article(pmid=18667618)
        x.get_values()
        x.check()

        x = neuroelectro.NeuroElectroSummary()
        x.set_neuron(id=72)
        x.set_ephysprop(id=2)
        x.get_values()
        x.check()
        

class BlueBrainTestCase(NotebookTools,
                        unittest.TestCase):
     
    path = '.'
    
    @unittest.skipUnless(bbp.is_bbp_up(), 
                         "Blue Brain URL is not responsive")
    def test_bluebrain(self):
        self.do_notebook('bbp')


class AIBSTestCase(unittest.TestCase):

    @unittest.skipUnless(aibs.is_aibs_up(), 
                         "AIBS URL is not responsive")
    def test_aibs(self):
        dataset_id = 354190013  # Internal ID that AIBS uses for a particular Scnn1a-Tg2-Cre 
                                # Primary visual area, layer 5 neuron.
        observation = aibs.get_observation(dataset_id,'rheobase')


if __name__ == '__main__':
    unittest.main()