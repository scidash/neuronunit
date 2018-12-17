import os

# try:
#     from pip.req import parse_requirements
#     from pip.download import PipSession
# except:
#     from pip._internal.req import parse_requirements
#     from pip._internal.download import PipSession

from setuptools import setup, find_packages


def read_requirements():
    """Parse requirements from requirements.txt."""
    reqs_path = os.path.join('.', 'requirements.txt')
    with open(reqs_path, 'r') as f:
        requirements = [line.rstrip() for line in f]
    install_requires = []
    dependency_links = []
    for i, r in enumerate(requirements):
        if "#egg=" in r:
            name = r.split('#egg=')[1]
            # e.g. sciunit-9999 to sciunit==9999
            install_requires += [name.replace('-', '==')]
            dependency_links += [r]
        else:
            install_requires += [r]
    return install_requires, dependency_links


install_requires, dependency_links = read_requirements()

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
    install_requires=install_requires,
    dependency_links=dependency_links
    )
