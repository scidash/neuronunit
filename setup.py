try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name='neuronunit',
    version='0.19',
    author='Rick Gerkin',
    author_email='rgerkin@asu.edu',
        packages=[
            'neuronunit',
            'neuronunit.capabilities',
            'neuronunit.neuroconstruct',
            'neuronunit.models',
            'neuronunit.tests',
            'neuronunit.optimization',
            'neuronunit.unit_test'],
    url='http://github.com/scidash/neuronunit',
    license='MIT',
    description='A SciUnit library for data-driven testing of single-neuron physiology models.',
    long_description="",
    test_suite="neuronunit.unit_test.core_tests",    
    install_requires=['scipy>=0.17',
                      'matplotlib>=2.0',
                      'neo==0.5.2',
                      'elephant==0.4.1',
                      'igor==0.3',
                      'sciunit==0.19',
                      'allensdk==0.14.2',
                      'pyneuroml==0.3.1.2',
                      ],
    )
