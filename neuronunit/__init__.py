"""NeuronUnit.

Testing for neuron and ion channel models
using the SciUnit framework.
"""

from __future__ import print_function
from __future__ import unicode_literals
import os
import platform

try:
    import sciunit
    assert sciunit
except ImportError as e:
    print("NeuronUnit requires SciUnit: http://github.com/scidash/sciunit")
    raise e

IMPLEMENTATION = platform.python_implementation()
JYTHON = IMPLEMENTATION == 'Jython'
CPYTHON = IMPLEMENTATION == 'CPython'
DIR = os.path.dirname(os.path.realpath(__file__))
