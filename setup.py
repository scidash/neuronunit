import os

try:
    from pip.req import parse_requirements
    from pip.download import PipSession
except:
    from pip._internal.req import parse_requirements
    from pip._internal.download import PipSession

from setuptools import setup, find_packages

def read_requirements():
    '''parses requirements from requirements.txt'''
    reqs_path = os.path.join('.', 'requirements.txt')
    install_reqs = parse_requirements(reqs_path, session=PipSession())
    reqs = [str(ir.req) for ir in install_reqs]
    return reqs

setup(
    name='neuronunit',
    version='0.19',
    author='Rick Gerkin',
    author_email='rgerkin@asu.edu',
    packages=find_packages(),
    url='http://github.com/scidash/neuronunit',
    license='MIT',
    description='A SciUnit library for data-driven testing of single-neuron physiology models.',
    long_description="",
    test_suite="neuronunit.unit_test.core_tests",    
    install_requires=read_requirements(),
    dependency_links=['git+https://github.com/scidash/neuronunit@dev#egg=sciunit-9999']
    )
