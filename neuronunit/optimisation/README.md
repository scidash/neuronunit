# Documentation Companion 

Optimizing inside a Data Driven Optimizer, and Model Validation Framework. By exploring and disgarding falsified Models. 
```python
#!pip install --user tabulate # Install the tabulate package
from tabulate import tabulate
import numpy as np
import pandas as pd
import IPython.display as d

# Some random data
data = np.random.rand(10,4)
# Columns A, B, C, D
columns = [chr(x) for x in range(65,69)]
# Create the dataframe
df = pd.DataFrame(data=data, 
                  columns=columns)
# Optionally give the dataframe's index a name
#df.index.name = "my_index"
# Create the markdown string
md = tabulate(df, headers='keys', tablefmt='pipe')
# Fix the markdown string; it will not render with an empty first table cell, 
# so if the dataframe's index has no name, just place an 'x' there.  
md = md.replace('|    |','| %s |' % (df.index.name if df.index.name else 'x'))
# Create the Markdown object
result = d.Markdown(md)
# Display the markdown object (in a Jupyter code cell)
result
```
