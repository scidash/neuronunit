"""Model classes for NeuronUnit."""

import os
import shutil
from urllib.parse import urljoin

import requests
import validators
import quantities as pq
from lxml import etree
from neuroml import nml

import neuronunit.capabilities as cap
from pyneuroml import pynml
from tempfile import TemporaryDirectory
from sciunit.models.runnable import RunnableModel


class LEMSModel(RunnableModel):
    """A generic LEMS model."""

    extra_capability_checks = {
        cap.ReceivesSquareCurrent: 'has_pulse_generator'
    }

    def __init__(self, LEMS_file_path_or_url, name=None,
                 backend=None, attrs=None):
        """Instantiate a LEMS model."""
        # If a URL is provided, download and get the path
        LEMS_file_path = self.url_to_path(LEMS_file_path_or_url)

        if name is None:
            name = os.path.split(LEMS_file_path)[1].split('.')[0]
        self.orig_lems_file_path = os.path.abspath(LEMS_file_path)
        assert os.path.isfile(self.orig_lems_file_path),\
            "'%s' is not a file" % self.orig_lems_file_path
        # Use original path unless create_lems_file is called
        self.lems_file_path = self.orig_lems_file_path

        if self.from_url:
            nml_paths = self.get_nml_paths(original=True, absolute=False)
            for nml_path in nml_paths:
                nml_url = urljoin(self.from_url, nml_path)
                self.url_to_path(nml_url)

        if backend is None:
            backend = 'jNeuroML'
        super(LEMSModel, self).__init__(name, backend=backend, attrs=attrs)
        self.set_default_run_params(**pynml.DEFAULTS)
        self.set_default_run_params(nogui=True)
        self.use_default_run_params()

    from_url = None

    def url_to_path(self, possible_url, base=None):
        """Check for a URL and download the contents.

        If it is not a URL, just consider it a local path to the contents.
        """
        possible_url = str(possible_url)
        if validators.url(possible_url):
            if base is None:
                base = os.getcwd()  # Location to which to download model files
            file_name = os.path.split(possible_url)[1]
            download_path = os.path.join(base, file_name)
            try:
                r = requests.get(possible_url, allow_redirects=True)
                if r.status_code != 200:
                    print("URL %s gave a response code %d" % (possible_url, r.status_code))
                with open(download_path, 'wb') as f:
                    f.write(r.content)
            except requests.ConnectionError:
                print("Could not connect to server at %s" % possible_url)
            
            self.from_url = possible_url
        else:
            download_path = possible_url
            self.from_url = False
        return download_path

    def get_nml_paths(self, lems_tree=None, absolute=True, original=False):
        """Get all NeuroML file paths associated with the model."""
        if not lems_tree:
            lems_tree = etree.parse(self.lems_file_path)
        nml_paths = []
        for atrb in ['file', 'href']:
            for tag in ['Include', 'include']:
                match = "*[contains(@%s, '.nml')][name() = '%s']" % (atrb, tag)
                elements = lems_tree.xpath(match)
                nml_paths += [x.attrib[atrb] for x in elements]
        if absolute:  # Turn into absolute paths
            lems_file_path = self.orig_lems_file_path if original \
                                                      else self.lems_file_path
            nml_paths = [os.path.join(os.path.dirname(lems_file_path), x)
                         for x in nml_paths]
        return nml_paths

    def create_lems_file_copy(self, name=None, use=True):
        """Create a temporary, writable copy of the original LEMS file.

        Used so that e.g. edits can be made to it programatically before
        simulation.
        """
        if name is None:
            name = self.name
        lems_copy_path = os.path.join(self.temp_dir.name,
                                      '%s.xml' % name)
        shutil.copy2(self.orig_lems_file_path, lems_copy_path)
        nml_paths = self.get_nml_paths(original=True)
        for orig_nml_path in nml_paths:
            new_nml_path = os.path.join(self.temp_dir.name,
                                        os.path.basename(orig_nml_path))
            shutil.copy2(orig_nml_path, new_nml_path)
        if self.attrs:
            self.set_lems_attrs(path=lems_copy_path)
        if use:
            self.lems_file_path = lems_copy_path
        return lems_copy_path

    def get_parsed_trees(self):
        """Get a dictionary of parsed XML trees for each model file."""
        lems_tree = etree.parse(self.lems_file_path)
        trees = {self.lems_file_path: lems_tree}
        nml_paths = self.get_nml_paths(lems_tree=lems_tree)
        trees.update({x: nml.nml.parsexml_(x) for x in nml_paths})
        for path, tree in trees.items():
            for elem in tree.getiterator():
                try:
                    # Set the tag name to the local name (i.e. without the namespace)
                    elem.tag = etree.QName(elem).localname
                except:
                    # Probably a comment or someting else that has no QName
                    pass
            # Remove unused namespace declarations
            etree.cleanup_namespaces(tree)
        return trees

    def set_lems_attrs(self, path=None):
        """Set attribite equivalents in the LEMS file and write it to disk."""
        if path is None:
            path = self.lems_file_path
        paths = [path] + self.get_nml_paths()
        for p in paths:
            tree = etree.parse(p)
            for key1, value1 in self.attrs.items():
                nodes = tree.findall(key1)
                for node in nodes:
                    for key2, value2 in value1.items():
                        node.attrib[key2] = value2
            tree.write(p)

    def set_lems_run_params(self, verbose=False):
        """Set run_param equivalents in the LEMS file and write it to disk."""
        trees = self.get_parsed_trees()

        # NeuronUnit->LEMS attribute mapping
        mapping = {'t_stop': 'length', 'dt': 'step'}
        # Edit NML files.
        for file_path, tree in trees.items():
            for key, value in self.run_params.items():
                if key in ['t_stop', 'dt']:
                    simulations = tree.findall('Simulation')
                    for sim in simulations:
                        value_in_ms = float(value.rescale(pq.ms))
                        sim.attrib[mapping[key]] = '%fms' % value_in_ms
                elif key == 'injected_square_current':
                    pulse_generators = tree.findall('pulseGenerator')
                    for pg in pulse_generators:
                        for attr in ['delay', 'duration', 'amplitude']:
                            if attr in value:
                                if verbose:
                                    print('Setting %s to %f' %
                                          (attr, value[attr]))
                                pg.attrib[attr] = '%s' % value[attr]

            tree.write(file_path)

    def has_pulse_generator(self, tree=None):
        """Return True if this model instance contains a pulse generator.

        It must be a NeuroML implementation of a pulse generator attached to an
        explicit input.
        """
        if tree is None:
            trees = self.get_parsed_trees()
            return any([self.has_pulse_generator(tree=tree)
                        for path, tree in trees.items()])
        else:
            try:
                pulse_generators = tree.findall('pulseGenerator')
                pg_ids = [pg.attrib['id'] for pg in pulse_generators]
                all_inputs = []
                explicit_inputs = tree.findall('.//explicitInput')
                all_inputs += [ei.attrib['input'] for ei in explicit_inputs]
                inputLists = tree.findall('.//inputList')
                all_inputs += [il.attrib['component'] for il in inputLists]
                if len(set(pg_ids).intersection(all_inputs)):
                    return True
            except Exception as e:
                raise e
            return False

    @property
    def temp_dir(self):
        if not hasattr(self, '_temp_dir'):
            self._temp_dir = TemporaryDirectory()
        return self._temp_dir

    def get_state_variables(self):
        """
        Parses LEMS xml file and gets simulation's state variables from
        OutputFile element.

        Returns:
            Dict of type "string: string" with the format "id: quantity"

         """
        lems_tree = etree.parse(self.lems_file_path)
        state_variables = {'t': 't'}
        for output_column in lems_tree.iter('OutputColumn'):
            id = output_column.attrib['id']
            quantity = output_column.attrib['quantity']
            state_variables[id] = quantity
        return state_variables
