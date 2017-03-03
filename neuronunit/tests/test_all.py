import unittest
import sys
import os
import warnings

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

class DocumentationTestCase(unittest.TestCase):
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
        f,nb = self.load_notebook(name)
        self.run_notebook(nb)
        f.close()
        self.assertTrue(True)

    def test_chapter1(self):
        self.execute_notebook('chapter1')



class OptimizationTestCase(unittest.TestCase):

    def load_script(self,name):
        f = open('opt/%s.py' % name)
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
        f,nb = self.load_notebook(name)
        self.run_notebook(nb)
        f.close()
        self.assertTrue(True)

    def test_chapter1(self):
        self.execute_notebook('chapter1')

if __name__ == '__main__':
    unittest.main()
        
