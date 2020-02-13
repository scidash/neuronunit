"""Tests of NeuronUnit documentation notebooks"""

from neuronunit.unit_test.base import *

class DocumentationTestCase(NotebookTools, 
                            unittest.TestCase):
    """Testing documentation notebooks"""

    path = '../examples'


    def test_elaborate(self):
        self.do_notebook('../examples/elaborate_backend_tests')
    def test_bbp(self):
        self.do_notebook('../unit_tests/bbp')

    def test_chapter1_opt(self):
        self.do_notebook('../examples/chapter1')
    def test_chapter2_opt(self):
        self.do_notebook('../examples/chapter2_needs_merge')
    def test_chapter3_opt(self):
        self.do_notebook('../examples/chapter3')
    def test_chapter4_opt(self):
        self.do_notebook('../examples/chapter4')
    def test_chapter5_opt(self):
        self.do_notebook('../examples/chapter5')
    def test_chapter6_opt(self):
        self.do_notebook('../examples/chapter6')
    def test_chapter7_opt(self):
        self.do_notebook('../examples/chapter7')
    def test_chapter8_opt(self):
        self.do_notebook('../examples/chapter8')
    def test_chapter9_opt(self):
        self.do_notebook('../examples/chapter9')
    def test_chapter10_opt(self):
        self.do_notebook('../examples/chapter10')
    def test_chapter11_opt(self):
        self.do_notebook('../examples/chapter11')

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
