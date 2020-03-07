from lxml import etree


root = etree.parse("Izh2007Cells.net.nml")


for child in root.iter():
     if "http://www.neuroml.org/schema/neuroml2}izhikevich2007Cell" in child.tag:
         parameters = dict(child.attrib)
         for k,v in parameters.items():
             print(v)
             try:
                 print(float(v))
             except:
                 pass

def my_cell(params):
    for child in root.iter():
         print(child.attrib)
         if "http://www.neuroml.org/schema/neuroml2}izhikevich2007Cell" in child.tag:
             parameters = child.attrib

             for k,v in params.items():
                 parameters[k] = v
