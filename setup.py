
import os
from setuptools import setup, find_packages


def read_requirements():
    """Parse requirements from requirements.txt."""
    reqs_path = os.path.join('.', 'requirements.txt')
    with open(reqs_path, 'r') as f:
        requirements = [line.rstrip() for line in f]
    return requirements


setup(
    name='neuronunit_opt',
    version='0.1',
    author='Russell Jarvis, Rick Gerkin',
    author_email='rjjarvis@asu.edu',
    packages=['neuronunit',
            'neuronunit.capabilities',
            'neuronunit.neuroconstruct',
            'neuronunit.models',
            'neuronunit.models.backends',
            'neuronunit.tests',
            'neuronunit.optimisation',
            'neuronunit.unit_test'],
    url='http://github.com/russelljjarvis/NeuronunitOpt',
    license='MIT',
    description=("A SciUnit library for optimizing data-driven testing of "
                 "single-neuron physiology models."),
    long_description="",
    test_suite="neuronunit.unit_test.core_tests"
    )
