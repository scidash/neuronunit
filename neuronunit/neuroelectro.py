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

import quantities
import sciunit
try: # Python 2
    from urllib import urlencode
    from urllib2 import urlopen,URLError
except: # Python 3
    from urllib.parse import urlencode
    from urllib.request import urlopen,URLError
import json
from pprint import pprint
from math import sqrt
import numpy as np

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
    def __init__(self,neuron={},ephysprop={},get_values=False):
        for key,value in neuron.items():
            setattr(self.neuron,key,value)
        for key,value in ephysprop.items():
            setattr(self.ephysprop,key,value)
        if get_values:
            self.get_values()

    url = API_URL # Base URL.
    neuron = Neuron()
    ephysprop = EphysProp()

    get_one_match = True # By default only get the first match

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

        if params is not None:
            for key in params.keys():
                if params[key] is not None:
                    query[key] = params[key]

        url += urlencode(query)
        return url

    def get_json(self, params=None, quiet=False):
        """Gets JSON data from neuroelectro.org based on the currently
        set neuron and ephys property.  Use 'params' to constrain the
        data returned."""
        url = self.make_url(params=params)
        if not quiet:
            print(url)
        try:
            url_result = urlopen(url,None,3) # Get the page.
            html = url_result.read() # Read out the HTML (actually JSON)
        except URLError as e:
            html = e.read().decode('utf-8')
            self.json_object = json.loads(html)
            if 'error_message' in self.json_object:
                if self.json_object['error_message'] == "Neuron matching query does not exist.":
                    print("No matching neuron found at NeuroElectro.org.")
            else:
                print("NeuroElectro.org appears to be down.")
            #print "Using fake data for now."
            #html = '{"objects":[{"n":{"name":"CA1 Pyramidal Cell"},
            #					  "e":{"name":"Spike Width"},\
            #					  "value_mean":0.001,
            #					  "value_sd":0.0003}]}'

        else:
            html = html.decode('utf-8')
            self.json_object = json.loads(html)
        return self.json_object

    def get_values(self, params=None, quiet=False):
        """Gets values from neuroelectro.org.
        We will use 'params' in the future to specify metadata (e.g. temperature)
        that neuroelectro.org will provide."""
        if not quiet:
            print("Getting data values from neuroelectro.org")
        self.get_json(params=params, quiet=quiet)
        if 'objects' in self.json_object:
            data = self.json_object['objects']
        else:
            data = None
        # All the summary matches in neuroelectro.org for this combination
        # of neuron and ephys property.
        if data and len(data):
            self.api_data = data[0] if self.get_one_match else data
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

    def get_values(self, params=None, quiet=False):
        data = super(NeuroElectroDataMap,self).get_values(params=params, \
                                                              quiet=quiet)
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
            sem = self.sem
        except AttributeError as a:
            print('The attributes "val" and "sem" were not found.')
            raise

class NeuroElectroSummary(NeuroElectroData):
    """Class for getting summary values (across reports) from
    neuroelectro.org."""

    url = API_URL+'nes/'

    def get_values(self, params=None, quiet=False):
        data = super(NeuroElectroSummary, self).get_values(params=params, \
                                                           quiet=quiet)
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

    def get_observation(self,params=None,show=False):
        values = self.get_values(params=params)
        if show:
            pprint(values)
        observation = {'mean':self.mean, 'std':self.std}
        return observation

    def check(self):
        try:
            mean = self.mean
            std = self.std
        except AttributeError as a:
            print('The attributes "mean" and "sd" were not found.')
            raise

class NeuroElectroPooledSummary(NeuroElectroDataMap):
    """Class for getting summary values by pooling each report's mean and SD from
    neuroelectro.org."""

    def get_values(self, params=None, quiet=False):

        # Get all papers reporting the neuron's property value
        self.get_one_match = False # We want all matches

        if params is None:
            params = {}

        params['limit'] = 999

        data = super(NeuroElectroDataMap, self).get_values(params=params, quiet=quiet)

        if data and len(data) > 0:

            # Ensure data from api matches the requested params
            data = [item for item in data \
                    if (item['ecm']['e']['name'] == self.ephysprop.name.lower() or \
                        item['ecm']['e']['id']   == self.ephysprop.id) \
                    and \
                       (item['ncm']['n']['nlex_id'] == self.neuron.nlex_id or \
                        item['ncm']['n']['id']      == self.neuron.id) \
                   ]

            # Set the neuron name and prop from the first json data object.
            self.neuron_name = data[0]['ncm']['n']['name']
            self.ephysprop_name = data[0]['ecm']['e']['name']

            # Pool each paper by weighing each mean by the paper's N and SD
            stats = self.get_pooled_stats(data, quiet)

            self.mean = stats['mean']
            self.std = stats['std']
            self.n = stats['n']
            self.items = stats['items']

            # Needed by check()
            self.val = stats['mean']
            self.sem = stats['sem']

            self.check()

        else:
            raise RuntimeError("No data was returned by the NeuroElectro API")

        return self.items

    def get_observation(self,params=None,show=False):
        values = self.get_values(params=params)

        if show:
            pprint(values)

        observation = {'mean':self.mean, 'std':self.std, 'n':self.n }

        return observation

    def get_pooled_stats(self, data, quiet = True):

        lines = []
        means = []
        sems = []
        sds = []
        ns = []
        weights = []
        sources = []

        if not quiet:
            print("Raw Values")

        # Collect raw values for each paper from NeuroElectro
        for item in data:

            err_is_sem = item['error_type'] == "sem" # SEM or SD
            err = (item['err_norm'] if item['err_norm'] is not None else item['err'])

            sem = err if err_is_sem else None
            sd = err if not err_is_sem else None
            mean = item['val_norm'] if item['val_norm'] is not None else item['val']
            n = item['n']
            source = item['source']

            means.append(mean)
            sems.append(sem)
            sds.append(sd)
            ns.append(n)
            sources.append(source)

            if not quiet:
                print({ 'mean': mean, 'std': sd, 'sem': sem, 'n': n })

        # Fill in missing values
        self.fill_missing_ns(ns)
        self.fill_missing_sems_sds(sems, sds, ns)

        if not quiet:
            print("---------------------------------------------------")
            print("Filled in Values (computed or median where missing)")

            for i in xrange(len(means)):
                line = { 'mean': means[i], 'sd': sds[i], 'sem': sems[i], 'n': ns[i], "source": sources[i] }
                lines.append(line)
                print(line)

        # Compute the weighted grand_mean
        # grand_mean  = SUM( N[i]*Mean[i] ) / SUM(N[i])
        # grand_sd = SQRT( SUM( (N[i]-1)*SD[i]^2 ) / SUM(N[i]-1) )

        ns_np = np.array(ns)
        means_np = np.array(means)
        sds_np = np.array(sds)
        n_sum = ns_np.sum()

        grand_mean = np.sum(ns_np * means_np) / n_sum
        grand_sd = np.sqrt( np.sum( (ns_np-1)*(sds_np**2) ) / np.sum(ns_np-1) )
        grand_sem = grand_sd / sqrt(n_sum)

        return { 'mean': grand_mean, 'sem': grand_sem, 'std': grand_sd, 'n': n_sum, 'items': lines }

    def fill_missing_ns(self, ns):
        # Fill in the missing N's with median N
        none_free_ns = np.array(ns)[ns != np.array(None)]

        if len(none_free_ns) > 0:
            n_median = int(np.median(none_free_ns))
        else:
            n_median = 1 # If no N's reported at all, weigh all means equally

        for i in xrange(len(ns)):
            if ns[i] is None:
                ns[i] = n_median

    def fill_missing_sems_sds(self, sems, sds, ns):
        # Fill in computable sems/sds
        for i in xrange(len(sems)):

            # Check if sem or sd is computable
            if sems[i] is None and sds[i] is not None:
                sems[i] = sds[i] / sqrt(ns[i])

            if sds[i] is None and sems[i] is not None:
                sds[i] = sems[i] * sqrt(ns[i])

        # Fill in the remaining missing using median sd
        none_free_sds = np.array(sds)[sds != np.array(None)]

        if len(none_free_sds) > 0:
            sd_median = np.median(none_free_sds)

        else: # If no SDs or SEMs reported at all, raise error

            # Perhaps the median SD of all cells for this property could be used
            # however, NE API nes interface does not support summary prop values without specifying the neuron id
            raise NotImplementedError('No StDevs or SEMs reported for "%s" property "%s"'%(self.neuron_name,self.ephysprop_name))

        for i in xrange(len(sds)):
            if sds[i] is None:
                sds[i] = sd_median
                sems[i] = sd_median / sqrt(ns[i])

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
    print("Tests passed.")









