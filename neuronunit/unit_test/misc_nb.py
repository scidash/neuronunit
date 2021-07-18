"""Tests of NeuronUnit documentation notebooks"""

from .base import *


class DocumentationTestCase(NotebookTools, unittest.TestCase):
    """Testing documentation notebooks"""

    def test_chapter1(self):
        self.do_notebook("relative_diff_unit_test")


if __name__ == "__main__":
    unittest.main()
