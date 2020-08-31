import unittest
from neuronunit.bbp import *
from sciunit.utils import NotebookTools

class BBPTestCase(NotebookTools, unittest.TestCase):
    def test_bbp_notebook(self):
        self.do_notebook("bbp")

    def test_is_bbp_up(self):
        list_curated_data()

if __name__ == '__main__':
    unittest.main()
