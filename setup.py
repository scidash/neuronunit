from distutils.core import setup

setup(
	name='neuronunit',
	version='0.1.4',
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
	install_requires=['sciunit>=0.1.3.1','numpy','scipy','neo']
)
