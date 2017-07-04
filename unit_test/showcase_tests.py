"""Unit tests for the showcase features of NeuronUnit"""

# Run with any of:  
# python showcase_tests.py
# python -m unittest showcase_tests.py
# coverage run --source . showcase_tests.py

import unittest

from sciunit.utils import NotebookTools

# CONVERT_NOTEBOOKS environment variable controls whether notebooks are
# executed as notebooks or converted to regular python files first. 

class DocumentationTestCase(NotebookTools,unittest.TestCase):
    """Testing documentation notebooks"""

    path = 'docs'

    #@unittest.skip("Skipping chapter 3")
    def test_chapter3(self):
        self.do_notebook('chapter3')
        

if __name__ == '__main__':
    unittest.main()
