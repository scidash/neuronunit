import os

# try:
#     from pip.req import parse_requirements
#     from pip.download import PipSession
# except:
#     from pip._internal.req import parse_requirements
#     from pip._internal.download import PipSession

from setuptools import setup, find_packages

import sys


def read_requirements():
    """Parse requirements from requirements.txt."""
    reqs_path = os.path.join('.', 'requirements.txt')
    with open(reqs_path, 'r') as f:
        requirements = [line.rstrip() for line in f]
    #install_requires = []
    #dependency_links = []
    #for i, r in enumerate(requirements):
    #if "#egg=" in r:
    #        name = r.split('#egg=')[1].split('-')[0]
    #        install_requires += ['%s @ %s' % (name, r)]
    #    else:
    #        install_requires += [r]
    return requirements


#install_requires, dependency_links = read_requirements()

setup(
    name='neuronunit',
    version='0.19',
    author='Rick Gerkin',
    author_email='rgerkin@asu.edu',
    packages=find_packages(),
    url='http://github.com/scidash/neuronunit',
    license='MIT',
    description=("A SciUnit library for data-driven testing of "
                 "single-neuron physiology models."),
    long_description="",
    test_suite="neuronunit.unit_test.core_tests",
    install_requires=read_requirements(),
    #dependency_links=dependency_links
    )
