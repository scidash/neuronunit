"""Tests of NeuronUnit documentation notebooks"""

from .base import *

class DocumentationTestCase(NotebookTools, 
                            unittest.TestCase):
    """Testing documentation notebooks"""

    path = '../../docs'

    #@unittest.skip("Skipping chapter 1")
    def test_chapter1(self):
        self.do_notebook('chapter1')

    @unittest.skip("Skipping chapter 2")
    def test_chapter2(self):
        self.do_notebook('chapter2')

    @unittest.skip("Skipping chapter 3")
    def test_chapter3(self):
        self.do_notebook('chapter3')

    @unittest.skip("Skipping chapter 4")
    def test_chapter4(self):
        self.do_notebook('chapter4')

    @unittest.skip("Skipping chapter 5")
    def test_chapter5(self):
        self.do_notebook('chapter5')
