
import os

import requests
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup, find_packages

from setuptools import setup#, find_packages

def read_requirements():
    '''parses requirements from requirements_light.txt'''
    reqs_path = os.path.join('.', 'requirements_light.txt')
    install_reqs = parse_requirements(reqs_path, session=PipSession())
    reqs = [str(ir.req) for ir in install_reqs]
    return reqs

setup(
    name='neuronunit',
    version='0.19',
    author='Rick Gerkin',
    author_email='rgerkin@asu.edu',
    packages=['neuronunit',
            'neuronunit.capabilities',
            'neuronunit.models',
            'neuronunit.tests',
            'neuronunit.optimisation',
            'neuronunit.unit_test'],
#+    packages=find_packages(),

    #packages=find_packages(),
    url='http://github.com/scidash/neuronunit',
    license='MIT',
    description='A SciUnit library for data-driven testing of single-neuron physiology models.',
    long_description="",
    test_suite="neuronunit.unit_test.core_tests")    
    #install_requires=read_requirements(),
    #)
