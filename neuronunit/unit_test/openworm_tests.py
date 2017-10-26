"""Unit tests for the showcase features of NeuronUnit"""

from .base import *

class DocumentationTestCase(NotebookTools,unittest.TestCase):
    """Testing documentation notebooks"""

    path = '../../docs'

    #@unittest.skip("Skipping chapter 2")
    def test_chapter2(self):
        self.do_notebook('chapter2')


if __name__ == '__main__':
    unittest.main()
