
import os
from setuptools import setup, find_packages


def read_requirements():
    """Parse requirements from requirements.txt."""
    reqs_path = os.path.join('.', 'requirements.txt')
    with open(reqs_path, 'r') as f:
        requirements = [line.rstrip() for line in f]
    return requirements


setup(
    name='neuronunit',
    version='0.19',
    author='Rick Gerkin',
    author_email='rgerkin@asu.edu',
    packages=['neuronunit',
            'neuronunit.capabilities',
            'neuronunit.neuroconstruct',
            'neuronunit.models',
            'neuronunit.tests',
            'neuronunit.optimisation',
            'neuronunit.unit_test'],
<<<<<<< HEAD
    install_requires = read_requirements(),
=======
>>>>>>> e5c1ba4c20dda6c404003b4c66be72b57b432159
    url='http://github.com/scidash/neuronunit',
    license='MIT',
    description=("A SciUnit library for data-driven testing of "
                 "single-neuron physiology models."),
    long_description="",
    test_suite="neuronunit.unit_test.core_tests",
    install_requires=read_requirements(),
    )
