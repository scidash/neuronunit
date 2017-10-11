""""Tests of ephys property measurements"""

from .base import *


class EphysPropertiesTestCase(NotebookTools, 
                              unittest.TestCase):
    """Testing sciunit tests of ephys properties"""

    path = '.'

    #@unittest.skip("Skipping get_tau test")
    def test_get_tau(self):
        self.do_notebook('get_tau')


if __name__ == '__main__':
    unittest.main()