"""Common imports for many unit tests in this directory"""

import unittest
import sys
import os
import warnings
import pdb
try: # Python 2
    from urllib import urlretrieve
except ImportError: # Python 3
    from urllib.request import urlretrieve

import matplotlib as mpl
mpl.use('Agg') # Avoid any problems with Macs or headless displays.

from sciunit.utils import NotebookTools,import_all_modules
from neuronunit import neuroelectro,bbp,aibs

OSX = sys.platform == 'darwin'