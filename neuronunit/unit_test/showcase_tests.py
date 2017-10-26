"""Unit tests for the showcase features of NeuronUnit"""

from .base import *

class DocumentationTestCase(NotebookTools,unittest.TestCase):
    """Testing documentation notebooks"""

    path = '../../docs'

    #@unittest.skip("Skipping chapter 3")
    def test_chapter3(self):
        self.do_notebook('chapter3')
        

if __name__ == '__main__':
    unittest.main()
