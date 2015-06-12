from setuptools import setup

setup(
        name='NeuronUnit',
        packages=[
            'neuronunit',
            'neuronunit.capabilities',
            'neuronunit.nml',
            'neuronunit.neuroconstruct',
            'neuronunit.models',
            'neuronunit.tests'
        ],
        url='https://github.com/scidash/neuronunit',
        description='A SciUnit repository for neuroscience-related tests, capabilities and so on.'
)

