try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
	name='neuronunit',
	version='0.1.5.1',
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
	install_requires=['quantities','sciunit>=0.1.4','numpy','scipy','neo','elephant'],
        dependency_links = ['git+http://github.com/neuralensemble/python-neo.git#egg=neo-0.4.0dev',
                            'git+http://github.com/neuralensemble/elephant.git#egg=elephant-0.1.1']
)
