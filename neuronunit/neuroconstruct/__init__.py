"""Neuroconstruct classes for use with neurounit"""

from __future__ import absolute_import
import os,sys

try:
    NC_HOME = os.environ["NC_HOME"]
except KeyError:
    raise Exception("Please add an NC_HOME environment variable corresponding\
                     to the location of the neuroConstruct directory.")

if NC_HOME not in sys.path:
    sys.path.append(NC_HOME)
