import os
from urllib import request, parse

# Example URL including an extra meaningless query key-value pair
example_url = 'https://neuroml-db.org/model_info?model_id=NMLCL000129&stuff=3'

# Parse the model_id from URL
parsed = parse.urlparse(example_url)
query = parse.parse_qs(parsed.query)
model_id = query['model_id'][0]

# Build the URL to the zip file and download it
zip_url = "https://neuroml-db.org/GetModelZip?modelID=%s&version=NeuroML" % model_id
location = '/tmp/%s.zip' % model_id
request.urlretrieve(zip_url, location)
assert os.path.isfile(location)