"""
Interface for creating tests using neuroelectro.org as reference data.  

Example workflow:

x = NeuroElectroDataMap() 
x.set_neuron(nlex_id='nifext_152') # neurolex.org ID for 'Amygdala basolateral nucleus 
								   # pyramidal neuron'.
x.set_ephysprop(id=2) # neuroelectro.org ID for 'Spike width'.  
x.set_article(pmid=18667618) # Pubmed ID for Fajardo et al, 2008 (J. Neurosci.)  
x.get_values() # Gets values for spike width from this paper.  
width = x.val # Spike width reported in that paper. 

t = neurounit.tests.SpikeWidthTest(spike_width=width)
m = sciunit.Model()
m.execute = code_that_runs_your_model
result = sciunit.run(t,m)
print result.score

OR

x = NeuroElectroSummary() 
x.set_neuron(nlex_id='nifext_152') # neurolex.org ID for 'Amygdala basolateral nucleus 
								   # pyramidal neuron'.
x.set_ephysprop(id=2) # neuroelectro.org ID for 'Spike width'.  
x.get_values() # Gets values for spike width from this paper.  
width = x.mean # Mean Spike width reported across all matching papers. 
...

"""

import sciunit
from urllib import urlencode
from urllib2 import urlopen
import json

API_VERSION = 1
API_SUFFIX = '/api/%d/' % API_VERSION
DEVELOPER = False
if DEVELOPER:
	DOMAIN = 'http://localhost:8000'
else:
	DOMAIN = 'http://www.neuroelectro.org'
API_URL = DOMAIN+API_SUFFIX

class Neuron:
	id = None
	nlex_id = None
	name = None

class EphysProp:
	id = None
	nlex_id = None	
	name = None

class Article:
	id = None
	pmid = None

# Abstract test class based on neuroelectro.org data using that site's API.  
class NeuroElectroData(object):
	url = API_URL # Base URL.  
	neuron = Neuron()
	ephysprop = EphysProp() 
	
	@classmethod
	def set_names(cls,neuron_name,ephysprop_name):
		cls.set_neuron(name=neuron_name)
		cls.set_ephysprop(name=ephysprop_name)
	
	def set_neuron(self,**kwargs): # Sets the biological neuron lookup attributes.  .  
		for key,value in kwargs.items():
			self.neuron.__setattr__(key,value)
	
	def set_ephysprop(self,id=None,nlex_id=None,name=None): # Sets the electrophysiological property lookup attributes.  
		for key,value in kwargs.items():
			self.ephysprop.__setattr__(key,value)
	
	def make_url(self,params=None):
		url = self.url+"?"
		# Create the full URL.  
		query = {}
		query['n'] = self.neuron.id # Change this for consistency in the neuroelectro.org API.  
		query['nlex'] = self.neuron.nlex_id # Change this for consistency in the neuroelectro.org API.  
		query['e'] = self.ephysprop.id
		query['e__name'] = self.ephysprop.name
		query = {key:value for key,value in query.items() if value is not None}
		url += urlencode(query)
		return url
	
	def get_json(self,params=None): # Gets JSON data from neuroelectro.org based on the currently set neuron and ephys property.  Use 'params' to constrain the data returned.  
		url = self.make_url(params=params)
		url_result = urlopen(url) # Get the page.  
		html = url_result.read() # Read out the HTML (actually JSON)
		self.json_object = json.loads(html) # Convert into a JSON object.  
	
	def get_values(self,params=None): # Gets values from neuroelectro.org.  We will use 'params' in the future to specify metadata (e.g. temperature) that neuroelectro.org will provide.  
		self.get_json(params=params)
		data = self.json_object['objects'] # All the summary matches in neuroelectro.org for this combination of neuron and ephys property.  
		self.api_data = data[0] # For now, we are just going to take the first match.  If neuron_id and ephysprop_id where both specified, there should be only one anyway.  
		return self.api_data

	def check(self): # See if the data requested from the server were obtained successfully.  
		pass
			
class NeuroElectroDataMap(NeuroElectroData):
	url = API_URL+'nedm/'
	article = Article()
	
	def set_article(self,id=None,pmid=None): # Sets the biological neuron using a NeuroLex ID.  
		self.article.id = id
		self.article.pmid = pmid
	
	def make_url(self,params=None):
		url = super(NeuroElectroDataMapTest, self).make_url(params=params)
		query = {}
		query['a'] = self.article.id
		query['pmid'] self.article.pmid
		query = {key:value for key,value in query.items() if value is not None}
		url += '&'+urlencode(query)
		return url
	
	def get_values(self,params=None):
		data = super(NeuroElectroDataMapTest,self).get_values(params=params)
		self.neuron_name = data['ncm']['n']['name'] # Set the neuron name from the json data.  
		self.ephysprop_name = data['ecm']['e']['name'] # Set the ephys property name from the json data.  
		self.val = data['val']
		self.err = data['err']
		self.n = data['n']
	
	def check(self):
		try:
			mean = self.val
			sd = self.err
		except AttributeError as a:
			print 'The val and err were not found.'
			raise

class NeuroElectroSummary(NeuroElectroData):
	url = API_URL+'nes/'
	
	def get_values(self,params=None):
		data = super(NeuroElectroSummaryTest, self).get_values(params=params)
		self.neuron_name = data['n']['name'] # Set the neuron name from the json data.  
		self.ephysprop_name = data['e']['name'] # Set the ephys property name from the json data.  
		self.mean = data['value_mean']
		self.sd = data['value_sd']
		self.n = data['n']
	
	def check(self):
		try:
			mean = self.mean
			sd = self.sd
		except AttributeError as a:
			print 'The mean and sd were not found.'
			raise

def test_module():
	x = NeuroElectroDataMap()
	x.set_neuron(id=72)
	x.set_ephysprop(id=2)
	x.set_article(pmid=18667618)
	x.get_values()
	x.check()

	x = NeuroElectroSummary()
	x.set_neuron(id=72)
	x.set_ephysprop(id=2)
	x.get_values()
	x.check()
	print "Tests passed."







	
