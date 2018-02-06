"""Neuroconstruct classes for use with neurounit"""

from __future__ import absolute_import
import os
import sys
import warnings

NC_HOME_DEFAULT = os.path.join(os.environ['HOME'],'neuroConstruct')

try:
    NC_HOME = os.environ["NC_HOME"]
except KeyError:
    warnings.warn(("Please add an NC_HOME environment variable corresponding "
                   "to the location of the neuroConstruct directory. The location "
                   "%s is being used as a default") % NC_HOME_DEFAULT)
    NC_HOME = NC_HOME_DEFAULT

if NC_HOME not in sys.path:
    sys.path.append(NC_HOME)
