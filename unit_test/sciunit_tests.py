"""Tests of SciUnit integration"""

from .base import *

class SciUnitTestCase(NotebookTools, 
                            unittest.TestCase):
    """Testing documentation notebooks"""

    path = '.'

    def test_serialization(self):
        self.do_notebook('serialization_test')
