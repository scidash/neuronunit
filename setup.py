try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name='neuronunit',
    version='0.1.8.5',
    author='Rick Gerkin',
    author_email='rgerkin@asu.edu',
        packages=[
            'neuronunit',
            'neuronunit.capabilities',
            'neuronunit.neuroconstruct',
            'neuronunit.neuron',
            'neuronunit.models',
            'neuronunit.tests'],
    url='http://github.com/scidash/neuronunit',
    license='MIT',
    description='A SciUnit library for data-driven testing of single-neuron physiology models.',
    long_description="",
    install_requires=['scipy>=0.17','matplotlib>=1.5','neo==0.4','elephant','sciunit>=0.1.5.5',],
    dependency_links = ['https://github.com/scidash/sciunit/tarball/dev']
)
