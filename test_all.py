import unittest
import sys
import os
import warnings

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


class NotebookTest:
    """Base class for testing notebooks"""
    def load_notebook(self,name):
        f = open('docs/%s.ipynb' % name)
        nb = nbformat.read(f, as_version=4)
        return f,nb

    def run_notebook(self,nb):
        if (sys.version_info >= (3, 0)):
            kernel_name = 'python3'
        else:
            kernel_name = 'python2'
        ep = ExecutePreprocessor(timeout=600, kernel_name=kernel_name)
        ep.preprocess(nb, {'metadata': {'path': '.'}})
        
    def execute_notebook(self,name):
        warnings.filterwarnings("ignore", category=DeprecationWarning) 
        warnings.filterwarnings("ignore", category=ImportWarning) 
        f,nb = self.load_notebook(name)
        self.run_notebook(nb)
        f.close()
        self.assertTrue(True)


class DocumentationTestCase(NotebookTest,unittest.TestCase):
    """Testing documentation notebooks"""

    #@unittest.skip("Skipping chapter 1")
    def test_chapter1(self):
        self.execute_notebook('chapter1')

    @unittest.skip("Skipping chapter 2")
    def test_chapter2(self):
        self.execute_notebook('chapter2')

    @unittest.skip("Skipping chapter 3")
    def test_chapter3(self):
        self.execute_notebook('chapter3')

    @unittest.skip("Skipping chapter 4")
    def test_chapter4(self):
        self.execute_notebook('chapter4')


if __name__ == '__main__':
    unittest.main()
        
