try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name='neuronunit',
    version='0.1.8.9',
    author='Rick Gerkin',
    author_email='rgerkin@asu.edu',
        packages=[
            'neuronunit',
            'neuronunit.capabilities',
            'neuronunit.neuroconstruct',
            'neuronunit.models',
            'neuronunit.tests'],
    url='http://github.com/scidash/neuronunit',
    license='MIT',
    description='A SciUnit library for data-driven testing of single-neuron physiology models.',
    long_description="",
    install_requires=['scipy>=0.17',
                      'matplotlib>=2.0',
                      'neo==9999',
                      'elephant==0.4.1',
                      'igor==0.3',
                      'sciunit==9999',
                      'allensdk==9999',
                      'pyneuroml==9999'],
    dependency_links = ['git+https://github.com/scidash/sciunit@dev#egg=sciunit-9999',
                        'git+https://github.com/rgerkin/AllenSDK@master#egg=allensdk-9999',
                        'git+https://github.com/rgerkin/pyNeuroML@master#egg=pyneuroml-9999',
                        'git+https://github.com/rgerkin/python-neo@master#egg=neo-9999']
)
