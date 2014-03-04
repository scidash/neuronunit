"""
Interface for creating tests using neuroelectro.org as reference data.  

Example workflow:

x = NeuroElectroDataMap() 
x.set_neuron(nlex_id='nifext_152') # neurolex.org ID for 'Amygdala basolateral
								   # nucleus pyramidal neuron'.
x.set_ephysprop(id=23) # neuroelectro.org ID for 'Spike width'.  
x.set_article(pmid=18667618) # Pubmed ID for Fajardo et al, 2008 (J. Neurosci.)  
x.get_values() # Gets values for spike width from this paper.  
width = x.val # Spike width reported in that paper. 

t = neurounit.tests.SpikeWidthTest(spike_width=width)
c = sciunit.Candidate() # Instantiation of your model (or other candidate)
c.execute = code_that_runs_your_model
result = sciunit.run(t,m)
print result.score

OR

x = NeuroElectroSummary() 
x.set_neuron(nlex_id='nifext_152') # neurolex.org ID for 'Amygdala basolateral 
								   # nucleus pyramidal neuron'.
x.set_ephysprop(id=2) # neuroelectro.org ID for 'Spike width'.  
x.get_values() # Gets values for spike width from this paper.  
width = x.mean # Mean Spike width reported across all matching papers. 
...

"""

import sciunit
from urllib import urlencode
from urllib2 import urlopen,URLError
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
 
class NeuroElectroData(object):
	"""Abstract class based on neuroelectro.org data using that site's API."""
	def __init__(self,neuron={},ephysprop={}):
		for key,value in neuron.items():
			setattr(self.neuron,key,value)
		for key,value in ephysprop.items():
			setattr(self.ephysprop,key,value)

	url = API_URL # Base URL.  
	neuron = Neuron()
	ephysprop = EphysProp() 
	
	@classmethod
	def set_names(cls,neuron_name,ephysprop_name):
		cls.set_neuron(name=neuron_name)
		cls.set_ephysprop(name=ephysprop_name)
	
	def set_neuron(self,id=None,nlex_id=None,name=None):
		"""Sets the biological neuron lookup attributes.""" 
		for key,value in locals().items():
			if key != 'self':
				setattr(self.neuron,key,value)
	
	def set_ephysprop(self,id=None,nlex_id=None,name=None):
		"""Sets the electrophysiological property lookup attributes."""
		for key,value in locals().items():
			if key != 'self':
				setattr(self.ephysprop,key,value)
	
	def make_url(self,params=None):
		"""Creates the full URL to the neuroelectro API."""  
		url = self.url+"?"
		query = {}
		# Change these for consistency in the neuroelectro.org API.  
		query['n'] = self.neuron.id 
		query['nlex'] = self.neuron.nlex_id
		query['n__name'] = self.neuron.name
		query['e'] = self.ephysprop.id
		query['e__name'] = self.ephysprop.name
		query = {key:value for key,value in query.items() if value is not None}
		url += urlencode(query)
		return url
	
	def get_json(self,params=None):
		"""Gets JSON data from neuroelectro.org based on the currently 
		set neuron and ephys property.  Use 'params' to constrain the 
		data returned."""
		url = self.make_url(params=params)
		print url
		try:
			url_result = urlopen(url,None,3) # Get the page.  
			html = url_result.read() # Read out the HTML (actually JSON)
		except URLError,e:
			html = e.read()
			self.json_object = json.loads(html)
			if 'error_message' in self.json_object:
				if self.json_object['error_message'] == "Neuron matching query does not exist.":
					print "No matching neuron found at NeuroElectro.org."
			else:
				print "NeuroElectro.org appears to be down."
			#print "Using fake data for now."
			#html = '{"objects":[{"n":{"name":"CA1 Pyramidal Cell"},
			#					  "e":{"name":"Spike Width"},\
			#					  "value_mean":0.001,
			#					  "value_sd":0.0003}]}'
			
		else:
			self.json_object = json.loads(html)
		return self.json_object

	def get_values(self,params=None): 
		"""Gets values from neuroelectro.org.  
		We will use 'params' in the future to specify metadata (e.g. temperature) 
		that neuroelectro.org will provide."""  
		print "Getting data values from neuroelectro.org"
		self.get_json(params=params)
		if 'objects' in self.json_object:
			data = self.json_object['objects'] 
		else:
			data = None
		# All the summary matches in neuroelectro.org for this combination 
		# of neuron and ephys property.  
		if data and len(data):
			self.api_data = data[0] 
		else:
			self.api_data = None
		# For now, we are just going to take the first match.  
		# If neuron_id and ephysprop_id where both specified, 
		# there should be only one anyway.  
		return self.api_data

	def check(self): 
		"""See if the data requested from the server were obtained successfully."""  
		pass
			
class NeuroElectroDataMap(NeuroElectroData):
	"""Class for getting single reported values from neuroelectro.org."""
	url = API_URL+'nedm/'
	article = Article()
	
	def set_article(self,id=None,pmid=None):
		"""Sets the biological neuron using a NeuroLex ID.""" 
		self.article.id = id
		self.article.pmid = pmid
	
	def make_url(self,params=None):
		url = super(NeuroElectroDataMap, self).make_url(params=params)
		query = {}
		query['a'] = self.article.id
		query['pmid'] = self.article.pmid
		query = {key:value for key,value in query.items() if value is not None}
		url += '&'+urlencode(query)
		return url
	
	def get_values(self,params=None):
		data = super(NeuroElectroDataMapTest,self).get_values(params=params)
		if data:
			self.neuron_name = data['ncm']['n']['name'] 
			# Set the neuron name from the json data.  
			self.ephysprop_name = data['ecm']['e']['name'] 
			# Set the ephys property name from the json data.  
			self.val = data['val']
			self.sem = data['err']
			self.n = data['n']
			self.check()
		return data
	
	def check(self):
		try:
			val = self.val
			std = self.sem
		except AttributeError as a:
			print 'The attributes "val" and "sem" were not found.'
			raise

class NeuroElectroSummary(NeuroElectroData):
	"""Class for getting summary values (across reports) from 
	neuroelectro.org."""
	
	url = API_URL+'nes/'
	
	def get_values(self,params=None):
		data = super(NeuroElectroSummary, self).get_values(params=params)
		if data:
			self.neuron_name = data['n']['name'] 
			# Set the neuron name from the json data.  
			self.ephysprop_name = data['e']['name']
			# Set the ephys property name from the json data.  
			self.mean = data['value_mean']
			self.std = data['value_sd']
			self.n = data['n']
			self.check()
		return data
		
	def check(self):
		try:
			mean = self.mean
			std = self.std
		except AttributeError as a:
			print 'The attributes "mean" and "sd" were not found.'
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








	
