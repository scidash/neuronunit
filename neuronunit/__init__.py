"""neurounit: Testing for neuroscience using the sciunit framework."""
import sciunit
import platform
IMPLEMENTATION = platform.python_implementation()
JYTHON = IMPLEMENTATION == 'Jython'
CPYTHON = IMPLEMENTATION == 'CPython'