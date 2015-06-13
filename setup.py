from distutils.core import setup

setup(
	name='neuronunit',
	version='0.1.2',
	author='Rick Gerkin',
	author_email='rgerkin@asu.edu',
	packages=['neuronunit'],
	url='http://github.com/scidash/neuronunit',
	license='MIT',
	description='A SciUnit library for data-driven testing of single-neuron physiology models.',
	long_description="",
	install_requires=['sciunit','numpy','scipy']
)
