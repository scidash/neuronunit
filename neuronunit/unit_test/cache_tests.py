""""Tests of ephys property measurements"""

from .base import *


class BackendCacheTestCase(NotebookTools,
                              unittest.TestCase):
    """Testing reading/writing to the backend cache"""

    path = '.'

    def test_cache_use(self):
        self.do_notebook('cache_use')

    def test_cache_edit(self):
        self.do_notebook('cache_edit')

if __name__ == '__main__':
    unittest.main()
