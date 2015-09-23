
# coding: utf-8

# ![NeuronUnit Logo](https://raw.githubusercontent.com/scidash/assets/master/logos/neuronunit.png)

# ###NeuronUnit is based on SciUnit, a discpline-agnostic framework for data-driven unit testing of scientific models.  
# 
# #Chapter 1
# NeuronUnit can be used to test neuron and ion channel models against the experimental data that you think they should aim to recapitulate.  
# - Have you read the [SciUnit documentation](https://github.com/scidash/sciunit/blob/master/docs/chapter1.ipynb)?  It might be good to start there!<br>
# - A few common Python packages are used below, all of which are available from PyPI.<hr>

# ###Any NeuronUnit test script will have the following actions: 
# Most of these will be abstracted away in SciUnit or NeuronUnit modules that make things easier for the developer:  
# 1. Instantiate a model(s) from a model class, with parameters of interest to build a specific model.    
# 1. Instantiate a test(s) from a test class, with parameters of interest to build a specific test.  
# 2. Check that the model has the capabilities required to take the test.     
# 1. Make the model take the test.  
# 2. Generate a score from that test run.  
# 1. Bind the score to the specific model/test combination and any related data from test execution. 
# 1. Visualize the score (i.e. print or display the result of the test).  

# ###Here, we will break down how this is accomplished in NeuronUnit.  
# Although NeuronUnit contains several model and test classes that make it easy to work with standards in neuron modeling and electrophysiology data reporting, here we will use toy model and test classes constructed on-the-fly so the process of model and test construction is fully transparent.  
# 
# Here is a toy model class:

# In[1]:

import sciunit
from neuronunit.capabilities import ProducesMembranePotential # This class contains a set of methods 
                                                              # for the model to implement.
from neo.core import AnalogSignal # Neo is a package for representing physiology data.  
import numpy as np # NumPy is how most numerical data is represented in Python.  
from quantities import ms # Quantities is a package for representing units.  

class ToyNeuronModel(sciunit.Model, # Each model must subclass SciUnit's base Model class.   
                     ProducesMembranePotential): # Each model will also subclass one of more capabilities.  
    """A toy neuron model that is always at it resting potential"""
    def __init__(self, v_rest, name=None): # Models are instantiated with their parameters.  
        self.v_rest = v_rest
        sciunit.Model.__init__(self, name=name) # SciUnit takes care of the rest.  

    def get_membrane_potential(self):
        """Gets a 'neo.core.AnalogSignal' object encoding the membrane potential trace for this cell model"""
        array = np.ones(10000) * self.v_rest # A trivial membrane potential trace that is constant.  
        dt = 1*ms # Time per sample in milliseconds.  
        vm = AnalogSignal(array,units=mV,sampling_rate=1.0/dt) # Creating the required output.  
        return vm


# <b>Inheriting from a SciUnit `Capability` is a how a SciUnit `Model` lets potential tests know what the `Model` can and cannot do.</b>  
# - It tells tests that the model will be implement any method stubbed out in the definition of that `Capability`.   
# - Tests can then check to make sure a `Model` has the required `Capabilities`, and then can interacts with the `Model` through those `Capabilities` (e.g. running a simulation and getting a membrane potential trace).  
# - The `ToyNeuronModel` class inherits from `sciunit.Model` (as do all NeuronUnit models), and also from  `ProducesMembranePotential`, which is a subclass of `sciunit.Capability`.  
# 
# Let's see what the `ProducesMembranePotential` capability looks like:  

# ```python
# class ProducesMembranePotential(sciunit.Capability):
#     """Indicates that the model produces a somatic membrane potential."""
#     
#     def get_membrane_potential(self):
#         """Must return a neo.core.AnalogSignal."""
#         raise NotImplementedError()
# 
#     def get_median_vm(self):
#         """Returns the median of the membrane potential"""
#         vm = self.get_membrane_potential() # Uses the method above to first get the trace.  
#         return np.median(vm) # Then returns the median of it.  
# ```

# `ProducesMembranePotential` has two methods.  
# - The first, `get_membrane_potential` is **unimplemented** by design.  Since there is no way to know how each model will generate and return a membrane potential time series, the `get_membrane_potential` method in this capability is left unimplemented, while the docstring describes what the model must implement in order to satisfy that capability.  In the `ToyNeuronModel` above, we see that the model implements it by simply creating a long array of resting potential values, and returning it as a `neo.core.AnalogSignal` object.  A more useful model would probably implement it by running a simulation and extracting membrane potential values from memory or disk.   
# - The second, `get_median_vm` is **already implemented**, which means there is no need for the model to do implement it again.  For its implementation to work, however, the implementation of `get_membrane_potential` must be complete.  Pre-implemented capability methods such as these allow the developer to focus on implementing only a few core interactions with the model, and then getting a lot of extra functionality for free.  In the example above, once we know that the membrane potential is being returned as a `neo.core.AnalogSignal`, we can simply take the median using numpy.  We know that the membrane potential isn't being returned as a list or a tuple or some other object on which numpy's median function won't necessarily work.  
# 
# Let's construct a single instance of this simple `model`, by choosing a value for the single membrane potential argument.  This toy `model` will now have a -60 mV membrane potential at all times:  

# In[2]:

from quantities import mV

my_neuron_model = ToyNeuronModel(-60.0*mV, name='my_neuron_model')


# Now we can then construct a simple test to use on this model or any other that expresses the appropriate capabilities:  

# In[3]:

from quantities import Quantity
from sciunit.scores import BooleanScore # One of many score types.  
from sciunit.comparators import compute_zscore # A function for computing a raw z-score.  
from sciunit.converters import RangeToBoolean

class ToyAveragePotentialTest(sciunit.Test):
    """Tests the average membrane potential of a neuron."""
    
    def __init__(self,
                 observation={'mean':None,'std':None},
                 name="Average potential test"):
        """Takes the mean and standard deviation of reference membrane potentials."""

        sciunit.Test.__init__(self,observation,name) # Call the base constructor.  
        self.required_capabilities += (ProducesMembranePotential,) # This test will require a model to 
                                                                   # express these capabilities
        self.comparator = compute_zscore
        self.converter = RangeToBoolean(-2,2)

    description = "A test of the average membrane potential of a cell."
    score_type = BooleanScore # The test will return this kind of score.  

    def validate_observation(self, observation):
        """An optional method that makes sure an observation to be used as reference data has the right form"""
        try:
            assert type(observation['mean']) is Quantity # The mean reported value must be expressed with units.  
            assert type(observation['std']) is Quantity # Ditto for the standard deviation.  
        except Exception as e:
            raise sciunit.ObservationError(("Observation must be of the form "
                                            "{'mean':float*mV,'std':float*mV}")) # If not, raise this exception.  

    def generate_prediction(self, model):
        """Implementation of sciunit.Test.generate_prediction."""
        vm = model.get_median_vm() # If the model has the capability 'ProducesMembranePotential', 
                                   # then it implements this method
        prediction = {'mean':vm} # Once we have our prediction, wrap it in a form that matches the observation.  
        return prediction

    
    def bind_score(self,score,model,observation,prediction):
        score.related_data['mean_vm'] = prediction['mean'] # Binds some related data about the test run.  
        return score


# ###The test constructor takes an *observation* to parameterize the test
# This observation could come from anywhere:  
# - Data from your lab
# - Results you saw in a paper
# - Data from http://neuroelectro.org (see the `neuronunit.neuroelectro` module)
# - Or it could be made up for illustration purposes, as below: 

# In[4]:

from quantities import mV
my_observation = {'mean':-62.1*mV, 
                  'std':3.5*mV}
my_average_potential_test = ToyAveragePotentialTest(my_observation, name='my_average_potential_test')


# A few things happen upon test instantiation, including validation of the observation. Since the observation has the correct format (for this test class), `ToyAveragePotentialTest.validate_observation` will complete successfully and the test will be ready for use.  
# 
# ###Now we judge our toy model using the test we wrote above:

# In[5]:

score = my_average_potential_test.judge(my_neuron_model)


# The `sciunit.Test.judge` method does several things.  
# 1. It checks to makes sure that `my_neuron_model` expresses the capabilities required to take the test.  It doesn't check to see if they are implemented correctly (how could it know?) but it does check to make sure the model at least claims (through inheritance) to express these capabilities.  The required capabilities are none other than those in the test's `required_capabilities` attribute.  Since `ProducesMembranePotential` is the only required capability, and the `ToyNeuronModel` class inherits from this capability class, that check passes.  
# 2. It calls the test's `generate_prediction` method, which uses the model's capabilities to make the model return some quantity of interest, in this case the median membrane potential.  
# 3. It calls the test's `compute_score` method, which compares the observation the test was instantiated with against the prediction returned in the previous step.  This comparison of quantities is cast into a `score` (in this case, a Z Score), bounds to some model output of interest (in this case, the model's mean membrane potential), and that `score` object is returned.  
# 4. The `score` returned is checked to make sure it is of the type promised in the class definition, i.e. that a Z Score is returned if a `ZScore` is listed in the test's `score_type` attribute.  
# 5. The `score` is bound to the `test` that returned it, the `model` that took the `test`, and the `prediction` and `observation` that were used to compute it.  
# 
# If all these checks pass (and there are no runtime errors) then we will get back a `score` object.  Usually this will be the kind of `score` we can use to evaluate `model`/`data` agreement.  If one of the `capabilities` required by the test is not expressed by the `model`, `judge` returns a special `NAScore` score type, which can be thought of as a blank.  It's not an error -- it just means that the model, as written, is not capable of taking that test.  
# 
# If there are runtime errors, they will be raised during test execution; however if the optional `stop_on_error` keyword argument is set to `False` when we call `judge`, then it we will return a special `ErrorScore` score type, encoding the error that was raised.  This is useful when batch testing many model and test combinations, so that the whole script doesn't get halted.  One can always check the scores at the end and then fix and re-run the subset of model/test combinations that returned an `ErrorScore`.
# 
# For any score, we can summarize it like so: 

# In[6]:

score.summarize()


# and we can get more information about the score:  

# In[7]:

score.describe()


# ###On to [Chapter 2](chapter2.ipynb)!
