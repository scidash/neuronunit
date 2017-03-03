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

    
    def func2map(file2map):
        import file2map
        #provide an exception to clean execution
        #if exception not raised return clean=True 
        #return clean


    def run_all_files():
        '''
        run all files as different CPU threads, thus saving time on travis
        Since scoop is designed to facilitate nested forking/parallel job dispatch
        This approach should be scalable to larger CPU pools.
        '''
        from scoop import futures
        files_to_exec=['exhaustive_search.py','nsga.py']
        clean_or_dirty=list(futures.map(func2map,files_to_exec))   
        return clean_or_dirty

    '''
    def load_script(self,name):
        f = open('%s.py' % name)
        #nb m nbformat.read(f, as_version=4)
        #return f,nb

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
    '''    

if __name__ == '__main__':
    unittest.main()
        
