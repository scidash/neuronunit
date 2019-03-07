import nmldbutils
import csv
import hashlib
import os
import re
import shutil
import string
import urllib2
import xml.etree.ElementTree as ET
from dateutil.parser import parse as parsedate

from database import NMLDB
from tables import *
from config import Config

from tables import Cells, Model_Waveforms, Models, Cells_Similar_Ephyz
from pandas import DataFrame
import numpy as np
import pandas
from matplotlib import pyplot as plt


class ModelManager(object):
    def __init__(self):
        self.config = Config()
        self.server = NMLDB()

        self.valid_actions = [
            'ignore',
            'UPDATE',
            'ADD',
            'DELETE'
        ]

        self.csv_columns = [
            'file_status',
            'action',
            'is_root',
            'model_id',
            'model_name',
            'model_type',
            'file_name',
            'md5',
            'pubmed_id',
            'translators',
            'references',
            'file_updated',
            'neurolex_terms',
            'keywords',
            'channel_protocol',
            'notes',
            'path',
            'children',
        ]

        self.comparison_fields = [
            'model_type',
            'md5',
            'parents',
            'children'
        ]

        self.multi_value_fields = [
            "translators",
            "references",
            "neurolex_terms",
            "keywords",
            "children"
        ]

        self.model_directories = []
        self.is_existing = False
        self.tree_nodes = {}
        self.roots = []
        self.root_ids = []
        self.model_directory_parent = self.config.permanent_models_dir
        self.valid_relationships = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.server.close()

    def validate_model_relationships(self, dirs):
        """
        Compares the parent-child relationships defined in the database to the
        parent-child relationships defined within individual NML files
        (based on the <include> statements).

        Generates a .csv file, which indicates any differences between the tree
        structures of DB and NML files.
        :rtype: None
        """
        # Build node tree from the DB data
        self.parse_directories(dirs)

        # Convert the DB tree to single-folder simulation
        self.to_simulation("temp/sim", clear_contents=True)

        with ModelManager() as sim_version:

            # Build the tree from the simulation files
            sim_version.parse_directories(["temp/sim"])

            # Compare the db tree to the simulation tree - generate comparison CSV
            self.compare_to(sim_version)

        self.to_csv("temp/validation_results.csv")

        if any(node["file_status"] != "same" for node in self.tree_nodes.values()):
            self.open_csv("temp/validation_results.csv")
        else:
            print("Valid: DB records and simulation files are identical")

    def model_to_csv(self, dirs):
        self.parse_directories(dirs)
        self.to_csv()
        self.open_csv()

    def csv_to_db(self, csv_file):
        self.parse_csv(csv_file)
        self.to_db_stored()

    def parse_csv(self, csv_path):
        with open(csv_path, "r") as f:
            rows = list(csv.DictReader(f))

        for row in rows:
            row['parents'] = []
            self.tree_nodes[row['file_name']] = row

        for node in self.tree_nodes.values():
            try:
                # Skip blank lines
                if string.join([str(v) for v in node.values()]).replace("[]", "").replace(" ", "") == "":
                    continue

                if node["action"] not in self.valid_actions:
                    raise Exception("Invalid action: '" + node["action"] + \
                                    "'. Allowed actions are: " + str(self.valid_actions))

                for val in node.values():
                    if node["action"] == 'ignore':
                        raise StopIteration()  # Skip node parsing if action is ignore

                    elif "DB:" in val or "|NML:" in val:
                        raise Exception("Unresolved conflict in CSV file: '" + val + "'")

                if node["model_type"] == "Channel" and node["channel_protocol"] == "":
                    raise Exception("Field channel_protocol is blank for channel: " + node["file_name"])

                for field in self.multi_value_fields:
                    self.parse_multi_value_field(node, field)

                node["file_updated"] = self.parse_date(node["file_updated"])

                node["pubmed_id"] = node["pubmed_id"].lower()

                node["translators"] = [{
                    'last_name': string.split(t, ",")[0],
                    'first_name': string.split(t, ",")[1]
                }
                    for t in node["translators"]]

                node['children'] = [{'file_name': c} for c in node['children']]

                for child in node['children']:
                    child_file = child['file_name']
                    self.tree_nodes[child_file]['parents'].append(node['file_name'])

            except StopIteration:
                continue

        self.get_roots()

    def to_csv(self, path=None):
        if path is None:
            path = self.config.default_out_csv_file

        print("Saving tree to: " + os.path.abspath(path) + "...")

        with open(path, 'wb') as file:
            writer = csv.writer(file, delimiter=',')
            self.write_header(writer)

            for root in self.roots:
                self.write_node(writer, root)

            for node in self.tree_nodes.values():
                if node not in self.roots:
                    self.write_node(writer, node)

    def parse_db_stored(self):
        print("Parsing model tree from DB records...")

        self.server.connect()
        self.tree_nodes = {}
        for root_id in self.root_ids:
            self.fetch_model_tree(root_id)
        self.get_roots()

    def to_db_stored(self):
        db = self.server.connect()

        try:
            with db.atomic() as transaction:
                # ADDS - update or link may depend on record existing
                adds = [n for n in self.tree_nodes.values() if n["action"] == "ADD"]
                for model in adds:
                    self.add_model(model)

                for model in adds:
                    self.add_links(model)

                # UPDATES - may add or remove a link
                updates = [n for n in self.tree_nodes.values() if n["action"] == "UPDATE"]
                for model in updates:
                    self.update_model(model)

                for model in updates:
                    self.add_links(model)

                # DELETEs - will remove any links other nodes
                deletes = [n for n in self.tree_nodes.values() if n["action"] == "DELETE"]
                for model in deletes:
                    self.remove_model(model)

                print('Committing transaction to DB...')

        except:
            print(
                "ERROR DETECTED: DB TRANSACTION ROLLED BACK - NO DB RECORDS SAVED - BUT CHECK FOR ANY ADDED/DELETED FILES")
            raise

    def parse_simulation(self):
        print("Parsing model tree from SIM NML files...")

        # Start with NML Files directly in the directory
        root_files = [file
                      for file in os.listdir(self.model_directories[0])
                      if nmldbutils.is_nml2_file(file)]

        self.tree_nodes = {}

        for root_file in root_files:
            self.parse_file_tree(root_file)

        self.get_roots()

    def parse_date(self, csv_value):
        if csv_value == 'None':
            return None

        else:
            return parsedate(csv_value)

    def get_valid_relationships(self):
        self.server.connect()

        Parent_Types = Model_Types.alias()
        Child_Types = Model_Types.alias()

        allowed = Model_Model_Association_Types \
            .select(Model_Model_Association_Types, Parent_Types, Child_Types) \
            .join(Parent_Types, on=(Parent_Types.ID == Model_Model_Association_Types.Parent_Type)) \
            .switch(Model_Model_Association_Types) \
            .join(Child_Types, on=(Child_Types.ID == Model_Model_Association_Types.Child_Type))

        self.valid_relationships = {}

        for rel in allowed:
            parent = rel.Parent_Type.Name
            child = rel.Child_Type.Name

            if parent not in self.valid_relationships:
                self.valid_relationships[parent] = []

            self.valid_relationships[parent].append(child)

        return self.valid_relationships

    def to_simulation(self, sim_path, clear_contents=False):
        print("Creating SIM files from model tree...")

        if sim_path is None:
            sim_path = self.config.out_sim_directory

        if os.path.exists(sim_path) and clear_contents:
            shutil.rmtree(sim_path)

        if clear_contents:
            os.makedirs(sim_path)

        for node in self.tree_nodes.values():
            shutil.copy2(node["path"], sim_path)

    def compare_to(self, simulation_importer):
        """

        :type simulation_importer: ModelManager
        """

        print("Comparing DB records to NML SIM files...")

        db_tree = self.tree_nodes
        sim_tree = simulation_importer.tree_nodes

        db_files = set(db_tree.keys())
        sim_files = set(sim_tree.keys())

        in_both = db_files.intersection(sim_files)
        in_db_only = db_files - sim_files
        in_sim_only = sim_files - db_files

        def compare_node_value(db_node, sim_node, key):
            db_val = db_node[key]
            sim_val = sim_node[key]

            if type(db_val) == list:

                if key == 'children':
                    db_children = set([c["file_name"] for c in db_val])
                    sim_children = set([c["file_name"] for c in sim_val])

                    if db_children.symmetric_difference(sim_children):
                        return False, {"db": db_val, "sim": sim_val}
                    else:
                        return True, db_val

                else:
                    db_val = set(db_val)
                    sim_val = set(db_val)

            if db_val == sim_val:
                return True, db_val
            else:
                return False, {"db": db_val, "sim": sim_val}

        for file in in_both:
            db_node = db_tree[file]
            sim_node = sim_tree[file]

            different_fields = []

            for key in self.comparison_fields:
                is_same, new_val = compare_node_value(db_node, sim_node, key)

                if not is_same:
                    different_fields.append(key)
                    db_node[key] = new_val

            if different_fields:
                db_node["file_status"] = 'Different:' + string.join(different_fields, ",")
            else:
                db_node["file_status"] = "same"

        for file in in_db_only:
            db_tree[file]["file_status"] = "Not in SIM FILE"

        for file in in_sim_only:
            # Copy the missing node over to db tree
            db_tree[file] = sim_tree[file]

            file_status = "Not in DB"
            matching_db_models = Models.select(Models.Model_ID).where(Models.File_Name == file)

            if matching_db_models:
                file_status += ". Maybe: " + string.join([m.Model_ID for m in matching_db_models], " or ") + "?"

            db_tree[file]["file_status"] = file_status

    def parse_multi_value_field(self, node, field):
        value = node[field]

        if value != "":
            node[field] = string.split(value, "|")
        else:
            node[field] = []

    def parse_directories(self, model_directories):

        for dir in model_directories:
            if nmldbutils.is_nmldb_id(dir):
                dir = os.path.join(self.config.permanent_models_dir, dir)

            self.model_directories.append(os.path.abspath(dir) + "/")

        self.is_existing = self.is_existing_model()

        if self.is_existing:
            self.root_ids = self.get_root_model_ids()
            self.model_directory_parent = re.compile('(.*)/NML').search(self.model_directories[0]).groups(1)[0]
            self.parse_db_stored()
        else:
            self.parse_simulation()

    def get_roots(self):
        # Roots are nodes without parents
        self.roots = [node for node in self.tree_nodes.values() if not node['parents']]

        for node in self.roots:
            node["is_root"] = True

    def remove_model(self, node):
        if not nmldbutils.is_nmldb_id(node["model_id"]):
            raise Exception("When DELET'ing, model ID must not be blank: " + node["file_name"])

        print("REMOVING model " + node["model_id"] + " from DB and models directory...")

        # Remove db records
        self.server.db.execute_sql("CALL delete_model('" + node["model_id"] + "');")

        # Remove files
        model_dir = os.path.join(self.model_directory_parent, node["model_id"])
        print("Deleting directory: " + model_dir + " ...")
        shutil.rmtree(model_dir)

    def update_model(self, node):
        if not nmldbutils.is_nmldb_id(node["model_id"]):
            raise Exception("When UPDATE'ing, model ID must not be blank: " + node["file_name"])

        print("Updating model " + node["model_id"] + " in DB...")

        model = Models.get(Models.Model_ID == node["model_id"])

        # Type and ID cannot be changed
        model.Name = node["model_name"]
        model.File_Name = node["file_name"]
        model.File_MD5_Checksum = node['md5']
        model.Notes = node["notes"]
        model.File_Updated = node["file_updated"]
        model.Publication = self.get_or_create_publication(node["pubmed_id"])
        model.save()

        # Translators, references, neurolexes, keywords
        self.add_metadata(model, node)

        # channel protocol
        if model.Type_id == "CH":
            channel = Channels.get(Channels.Model_ID == model.Model_ID)

            # ok to change - though simulation has to be rerun - need a way to mark simulations to be rerun in general - perhaps at model table level
            channel.Channel_Class = node["channel_protocol"]
            channel.save()

    def add_model(self, node):
        print("Adding model " + node["file_name"] + " to DB...")

        if node["model_id"] != "":
            raise Exception("When ADD'ing, model ID should be blank: " + node["model_id"])

        type_id = Model_Types.get(Model_Types.Name == node['model_type']).ID

        new_model_id = 'NML' + type_id + str(Models.select(peewee.fn.MAX(Models.ID_Helper)).scalar() + 1).zfill(6)

        model = Models.create(
            Model_ID=new_model_id,
            Name=node["model_name"],
            File_Name=node["file_name"],
            File_MD5_Checksum=node['md5'],
            Notes=node["notes"],
            File_Updated=node["file_updated"],
            Publication=self.get_or_create_publication(node["pubmed_id"])
        )

        node["model_id"] = new_model_id

        # Translators, references, neurolexes, keywords
        self.add_metadata(model, node)

        # channel protocol
        if type_id == "CH":
            Channels.create(
                Model_ID=model.Model_ID,
                Type=node["channel_protocol"]
            )

        # Copy model file to DB model directory
        new_dir = self.model_directory_parent + "/" + new_model_id

        if not os.path.exists(new_dir):
            os.mkdir(new_dir)

        shutil.copy2(node["path"], new_dir)

        print("Added " + new_model_id)

    def add_links(self, parent_node):

        # Clear the existing child list first
        Model_Model_Associations \
            .delete() \
            .where(Model_Model_Associations.Parent == parent_node["model_id"]) \
            .execute()

        # (Re)add children links
        for child in parent_node["children"]:
            child_node = self.tree_nodes[child["file_name"]]

            if child_node["model_id"] in ("", None):
                raise Exception("All child files must have an NML-DB ID: " + child["file_name"])

            Model_Model_Associations.create(
                Parent=parent_node["model_id"],
                Child=child_node["model_id"]
            )

    def add_metadata(self, model, node):
        # Nodes contain the complete sets of these, clear the links before adding to avoid dups
        Model_Translators.delete().where(Model_Translators.Model == model).execute()
        Model_References.delete().where(Model_References.Model == model).execute()
        Model_Neurolexes.delete().where(Model_Neurolexes.Model == model).execute()
        Model_Other_Keywords.delete().where(Model_Other_Keywords.Model == model).execute()

        # Translators
        for i, translator in enumerate(node["translators"]):
            Model_Translators.create(
                Translator=self.get_or_create_author(translator["first_name"], translator["last_name"]),
                Model=model,
                Translator_Sequence=i
            )

        # references
        for node_ref in node["references"]:
            Model_References.create(
                Model=model,
                Reference=self.get_or_create_reference(node_ref)
            )

        # neurolexes -
        for node_nlx in node["neurolex_terms"]:
            Model_Neurolexes.create(
                Model=model,
                Neurolex=self.get_neurolex(node_nlx)
            )

        # keywords - create new if cant find exact match
        for kwd in node["keywords"]:
            Model_Other_Keywords.create(
                Model=model,
                Other_Keyword=self.get_or_create_keyword(kwd)
            )

    def get_or_create_keyword(self, keyword_string):
        keyword, created = Other_Keywords.get_or_create(Other_Keyword_Term=keyword_string)
        return keyword

    def get_neurolex(self, term_string):
        # use existing nlx - raise error if cannot match exactly
        try:
            return Neurolexes.get(Neurolexes.NeuroLex_Term == term_string)
        except:
            raise Exception(
                "Could not find Neurolex term '" + term_string + "'. Make sure that a) the neurolex term exists in the database and b) neurolex Term and not the URI are used in the .csv file.")

    def get_or_create_reference(self, ref_url):
        # Create if dont exist, detect from url which resource to use - show error if can't find one
        existing = Refers.get_or_none(Refers.Reference_URI == ref_url)

        if existing is not None:
            return existing
        else:
            resources = Resources \
                .select()

            resource = None
            for r in resources:
                if r.Identifying_URL_Snippet.lower() in ref_url.lower():
                    resource = r
                    break

            if resource is None:
                raise Exception("Could not find a matching record in Resources table for reference URL:" + ref_url)

            ref = Refers()
            ref.Reference_URI = ref_url
            ref.Resource = resource
            ref.save()

            return ref

    def get_or_create_publication(self, pubmed_ref):
        existing = Publications.get_or_none(Publications.Pubmed_Ref == pubmed_ref)

        if existing is not None:
            return existing
        else:
            if pubmed_ref == "":
                return None

            title, year, authors = self.get_pub_info_from_nih(pubmed_ref)

            pub = Publications.create(
                Pubmed_Ref=pubmed_ref,
                Full_Title=title,
                Year=year
            )

            for i, author in enumerate(authors):
                db_author = self.get_or_create_author(author["first_name"], author["last_name"])

                Publication_Authors.create(
                    Publication=pub,
                    Author=db_author,
                    Author_Sequence=i
                )

            return pub

    def get_or_create_author(self, fname, lname):
        author, created = People.get_or_create(Person_Last_Name=lname, Person_First_Name=fname)
        return author

    def get_pub_info_from_nih(self, pubmed_ref):
        pmid = pubmed_ref.lower().replace("pubmed/", "")  # Comes in as e.g. "pubmed/16293591" or "16293591"

        url = 'http://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&retmode=xml&id=' + pmid
        pm_Tree = ET.ElementTree(file=urllib2.urlopen(url))

        title = pm_Tree.findall('.//ArticleTitle')[0].text

        author_list = pm_Tree.findall('.//Author')
        authors = [{
            "last_name": author.find('LastName').text,
            "first_name": author.find('ForeName').text}
            for author in author_list]

        try:
            year = int(pm_Tree.findall('.//PubDate')[0].find('Year').text)
        except:
            try:
                import re
                year = int(re.findall(r'\d+', pm_Tree.findall('.//PubDate')[0].find('MedlineDate').text)[0])
            except:
                raise

        return title, year, authors

    def fetch_model_tree(self, id, parent=None):
        print("Fetching " + id + "...")
        model = self.fetch_model_record(id)

        if model.File_Name in self.tree_nodes:

            # Add only parent info, if any
            if parent is not None:
                self.tree_nodes[model.File_Name]['parents'].append(parent)

            # Skip otherwise
            return

        node = self.get_blank_node()
        self.tree_nodes[model.File_Name] = node

        node["model_id"] = model.Model_ID
        node["model_name"] = model.Name
        node["model_type"] = model.Type.Name
        node["file_name"] = model.File_Name
        node['md5'] = model.File_MD5_Checksum
        node["pubmed_id"] = model.Publication.Pubmed_Ref

        node["translators"] = [
            {"last_name": t.Person_Last_Name,
             "first_name": t.Person_First_Name}
            for t in model.Translators]

        node["references"] = [r.Reference_URI for r in model.References]
        node["neurolex_terms"] = [r.NeuroLex_Term for r in model.Neurolexes]
        node["keywords"] = [r.Other_Keyword_Term for r in model.Keywords]

        if model.Type.Name == "Channel":
            channel = Channels.get(Channels.Model_ID == model.Model_ID)
            node["channel_protocol"] = channel.Channel_Type

        node["notes"] = model.Notes
        node["file_updated"] = model.File_Updated
        node["path"] = self.server_path_to_local_path(model.File)

        if parent is not None:
            node["parents"].append(parent)

        node["children"] = [{
            'file_name': c.File_Name,
            'nml_id': c.Model_ID,
            'server_path': c.File,
            'local_path': self.server_path_to_local_path(c.File)
        }
            for c in model.Children]

        for child in node["children"]:
            self.fetch_model_tree(child['nml_id'], parent=node["file_name"])

    def fetch_model_record(self, id):
        result = Models \
            .select(Models, Model_Types, Publications) \
            .join(Model_Types) \
            .switch(Models) \
            .join(Publications) \
            .where(Models.Model_ID == id) \
            .first()

        if result is None:
            raise Exception("Model with id %s not found" % id)

        result.Translators = People \
            .select() \
            .join(Model_Translators) \
            .where(Model_Translators.Model == id) \
            .order_by(Model_Translators.Translator_Sequence)

        result.References = Refers \
            .select() \
            .join(Model_References) \
            .where(Model_References.Model == id)

        result.Neurolexes = Neurolexes \
            .select() \
            .join(Model_Neurolexes) \
            .where(Model_Neurolexes.Model == id)

        result.Keywords = Other_Keywords \
            .select() \
            .join(Model_Other_Keywords) \
            .where(Model_Other_Keywords.Model == id)

        result.Children = Models \
            .select(Models.Model_ID, Models.File, Models.File_Name) \
            .join(Model_Model_Associations, on=(Model_Model_Associations.Child == Models.Model_ID)) \
            .where(Model_Model_Associations.Parent == id)

        return result

    def server_path_to_local_path(self, server_path):
        return server_path.replace(self.config.server_model_path, self.model_directory_parent)

    def get_type(self, path):

        if path.endswith(".cell.nml"):
            return "Cell"

        if path.endswith(".channel.nml"):
            return "Channel"

        if path.endswith(".synapse.nml"):
            return "Synapse"

        if path.endswith(".net.nml"):
            return "Network"

        if path.endswith(".component.nml"):
            return "Component"

        if not os.path.exists(path):
            return "MISSING"

        root = self.get_xml_root(path)

        for element in root:
            tag = element.tag.lower()

            if 'concentration' in tag:
                return "Concentration"

            if 'gapjunction' in tag or 'synapse' in tag:
                return "Synapse"

            if 'ionchannel' in tag:
                return "Channel"

            if 'cell' in tag:
                return "Cell"

            if 'generator' in tag or "clamp" in tag:
                return "Input"

        return "UNKNOWN"

    @staticmethod
    def get_blank_node():
        return {
            'file_status': 'ok',
            'action': 'ignore',
            'is_root': False,
            'model_id': '',
            'model_name': '',
            'model_type': '',
            'file_name': '',
            'file_updated': '',
            'md5': '',
            'path': '',
            'dir': '',
            'children': [],
            'parents': [],
            'pubmed_id': '',
            'translators': [],
            'references': [],
            'neurolex_terms': [],
            'keywords': [],
            'channel_protocol': '',
            'notes': ''
        }

    def write_header(self, writer):
        writer.writerow(self.csv_columns)

    def write_node(self, writer, node):

        def field_to_string(node, key):
            value = node[key]

            if key in ['references', 'neurolex_terms', 'keywords']:
                return string.join(value, '|')

            elif key == 'translators':
                return string.join([tr["last_name"] + "," + tr["first_name"] for tr in value], '|')

            elif key == 'children':
                if type(value) == dict:
                    return "DB:" + string.join([ch['file_name'] for ch in value["db"]], '|') + \
                           "|NML:" + string.join([ch['file_name'] for ch in value["sim"]], '|')

                return string.join([ch['file_name'] for ch in value], '|')

            else:
                if type(value) == dict:
                    return "DB:" + str(value["db"]) + "|NML:" + str(value["sim"])

                return str(value)

        writer.writerow([unicode(field_to_string(node, key)).encode('utf-8') for key in self.csv_columns])

    def is_existing_model(self):
        return len(self.get_root_model_ids()) > 0

    def get_root_model_ids(self):
        result = []

        for dir in self.model_directories:
            try:
                root_dir = dir.split("/")[-2]

                # Check if parent dir is named as an NML-DB ID
                if nmldbutils.is_nmldb_id(root_dir):
                    result.append(root_dir)
            except:
                pass

        return result

    def parse_file_tree(self, file, parent=None):
        # If file already parsed
        if file in self.tree_nodes:

            # Add only parent info, if any
            if parent is not None:
                self.tree_nodes[file]['parents'].append(parent)

            # Skip otherwise
            return

        # Create node
        node = self.get_blank_node()

        node['file_name'] = file
        node['dir'] = self.model_directories[0]
        node['path'] = self.model_directories[0] + file
        node['md5'] = nmldbutils.get_file_checksum(node['path'])
        node['model_type'] = self.get_type(node["path"])

        if parent is not None:
            node['parents'].append(parent)

        # Add it to the tree
        self.tree_nodes[file] = node

        # Read xml - if file exists
        if os.path.exists(node['path']):

            parent_type = node['model_type']
            root = self.get_xml_root(node['path'])

            # Get list of node children
            for child in root:
                if child.tag == "{http://www.neuroml.org/schema/neuroml2}include":
                    link_attrib = 'href' if 'href' in child.attrib else 'file'
                    if child.attrib[link_attrib]:
                        child_rel_path = child.attrib[link_attrib]
                        child_file = os.path.basename(child_rel_path)

                        child_path = self.model_directories[0] + child_file
                        child_type = self.get_type(child_path)

                        # Don't add children that have invalid relationships
                        if child_type != "MISSING" and child_type not in self.valid_child_types(parent_type):
                            continue

                        node['children'].append({'file_name': child_file})

            # Repeat for each child
            for child_file in node['children']:
                self.parse_file_tree(child_file['file_name'], parent=node['file_name'])

        else:
            node["file_status"] = "File does not exist"

    def valid_child_types(self, parent_type):

        if self.valid_relationships is None:
            self.get_valid_relationships()

        if parent_type not in self.valid_relationships:
            return []

        else:
            return self.valid_relationships[parent_type]

    def get_xml_root(self, file_path):

        tree = ET.parse(file_path)
        root = tree.getroot()
        return root

    def open_csv(self, file_path=None):

        if file_path is None:
            file_path = self.config.default_out_csv_file

        print("Opening CSV file: " + file_path)

        import subprocess, os, sys
        if sys.platform.startswith('darwin'):
            subprocess.call(('open', file_path))
        elif os.name == 'nt':
            os.startfile(file_path)
        elif os.name == 'posix':
            subprocess.call(('xdg-open', file_path))

    def save_model_properties(self, models, properties):
        self.server.connect()

        if models == ['ALL']:
            models = [model for model in Models.select(Models)]

        for model in models:
            model_id = model.Model_ID if model.__class__.__name__ == "Models" else model

            if model_id.startswith("NMLCL"):
                from cellmodel import CellModel as TargetModel

                if len(properties) == 1 and properties[0] in ('DRUCKMANN_PROPERTIES', 'ramp_ap_onset', 'frequency_filtering'):
                    skip_NEURON = True
                else:
                    skip_NEURON = False

            elif model_id.startswith("NMLCH"):
                from channelmodel import ChannelModel as TargetModel
                skip_NEURON = False

            else:
                from nmldbmodel import NMLDB_Model as TargetModel
                skip_NEURON = True

            with TargetModel(model, server=self.server) as m:
                m.save_properties(properties, skip_conversion_to_NEURON=skip_NEURON)

    def update_model_checksums(self):
        """
        Computes the MD5 hash of each model in the database. Updates DB record if different.
        """
        self.server.connect()

        models = Models.select(Models.Model_ID)
        from nmldbmodel import NMLDB_Model

        for model in models:
            with NMLDB_Model(model.Model_ID, server=self.server) as m:
                m.save_property('checksum')

    def find_waveforms_without_files(self):
        """
        Finds waveforms that are in DB but not in file system
        """
        self.server.connect()
        print('Getting records...')

        from nmldbmodel import NMLDB_Model

        waves = Model_Waveforms\
            .select(Model_Waveforms.ID, Model_Waveforms.Model)


        missing = []
        curr_model_id = None
        m = None
        for wave in waves:
            if wave.Model_id != curr_model_id:
                model_id = wave.Model_id
                curr_model_id = model_id
                print('Checking model: ' + model_id)

                m = NMLDB_Model(model_id, server=self.server, skip_model_record=True)

            file = m.get_waveform_path(wave.ID)

            if not os.path.exists(file):
                missing.append(wave.ID)

        print('The following waveforms are missing in file system:')

        if len(missing) == 0:
            print("NONE -- all DB waves present in file system")

        else:
            print("SELECT * FROM model_waveforms mw WHERE mw.ID IN (")
            print(string.join([str(w) for w in missing], ','))
            print(")")

    def find_multicomp_cells_without_gifs(self):
        """
        Finds multicomp (>1 section) cells that do not have morphology gifs
        """
        self.server.connect()
        print('Getting records...')

        from nmldbmodel import NMLDB_Model

        models = Cells.select(Cells.Model_ID).where(Cells.Sections > 1)

        missing = []
        for model in models:
            print("Checking " + model.Model_ID)
            with NMLDB_Model(model.Model_ID, server=self.server, skip_model_record=True) as m:
                file = m.get_gif_path()

                if not os.path.exists(file):
                    missing.append(model.Model_ID)

        print('The following multi-comp cells do not have morphology cell.gif files:')

        if len(missing) == 0:
            print("NONE -- all multi-comp cells have morphology gifs")

        else:
            print("SELECT * FROM models m WHERE m.Model_ID IN (")
            print(string.join([str(m) for m in missing], ','))
            print(")")


    def get_pre_processed_cell_ephyz_props(self, cells):
        # These will be used to perform clustering
        prop_names = [
            'AP1Amplitude',
            'AP2Amplitude',
            'AP12AmplitudeDrop',
            'AP12AmplitudeChangePercent',
            'AP1SSAmplitudeChange',

            'AP1WidthHalfHeight',
            'AP2WidthHalfHeight',
            'AP12HalfWidthChangePercent',

            'AP1WidthPeakToTrough',
            'AP2WidthPeakToTrough',

            'AP1RateOfChangePeakToTrough',
            'AP2RateOfChangePeakToTrough',
            'AP12RateOfChangePeakToTroughPercentChange',

            'AP1AHPDepth',
            'AP2AHPDepth',
            'AP12AHPDepthPercentChange',

            'AP1DelayMean',
            'AP2DelayMean',

            'AP1DelaySD',
            'AP2DelaySD',

            'AP1DelayMeanStrongStim',
            'AP2DelayMeanStrongStim',

            'AP1DelaySDStrongStim',
            'AP2DelaySDStrongStim',

            'Burst1ISIMean',
            'Burst1ISIMeanStrongStim',

            'Burst1ISISD',
            'Burst1ISISDStrongStim',

            'InitialAccommodationMean',
            'SSAccommodationMean',
            'AccommodationRateToSS',
            'AccommodationAtSSMean',
            'AccommodationRateMeanAtSS',

            'ISIMedian',
            'ISICV',
            'ISIBurstMeanChange',

            'SpikeRateStrongStim',

            'InputResistance',

            'SteadyStateAPs',

            'FrequencyPassAbove',
            'FrequencyPassBelow',

            'RampFirstSpike',
        ]

        props = {}
        for c, cell in enumerate(cells):
            for p, prop in enumerate(prop_names):
                if prop not in props:
                    props[prop] = []

                if prop == 'SteadyStateAPs':
                    props[prop].append(cell.Spikes)

                else:
                    props[prop].append(getattr(cell, prop))

        df = DataFrame(props, columns=prop_names)

        df['AP1Amplitude'].fillna(0, inplace=True)
        df['AP2Amplitude'].fillna(0, inplace=True)

        df['AP1SSAmplitudeChange'].fillna(0, inplace=True)

        df['AP1WidthHalfHeight'].fillna(0, inplace=True)
        df['AP2WidthHalfHeight'].fillna(0, inplace=True)

        df['AP1WidthPeakToTrough'].fillna(0, inplace=True)
        df['AP2WidthPeakToTrough'].fillna(0, inplace=True)

        df['AP1RateOfChangePeakToTrough'].fillna(0, inplace=True)
        df['AP2RateOfChangePeakToTrough'].fillna(0, inplace=True)

        df['AP1AHPDepth'].fillna(0, inplace=True)
        df['AP2AHPDepth'].fillna(0, inplace=True)

        df['AP1DelayMean'].fillna(2000, inplace=True)
        df['AP2DelayMean'].fillna(2000, inplace=True)

        df['AP1DelaySD'].fillna(0, inplace=True)
        df['AP2DelaySD'].fillna(0, inplace=True)

        df['AP1DelayMeanStrongStim'].fillna(2000, inplace=True)
        df['AP2DelayMeanStrongStim'].fillna(2000, inplace=True)

        df['AP1DelaySDStrongStim'].fillna(0, inplace=True)
        df['AP2DelaySDStrongStim'].fillna(0, inplace=True)

        df['Burst1ISIMean'].fillna(2000, inplace=True)
        df['Burst1ISIMeanStrongStim'].fillna(2000, inplace=True)

        df['Burst1ISISD'].fillna(0, inplace=True)
        df['Burst1ISISDStrongStim'].fillna(0, inplace=True)

        df['AccommodationRateMeanAtSS'].fillna(2000, inplace=True)

        df['ISIMedian'].fillna(2000, inplace=True)

        df['ISICV'].fillna(0, inplace=True)

        df['ISIBurstMeanChange'].fillna(0, inplace=True)

        df['SpikeRateStrongStim'].fillna(0, inplace=True)

        df['InputResistance'].fillna(df['InputResistance'].mean(), inplace=True)

        df['FrequencyPassAbove'].fillna(29, inplace=True)
        df['FrequencyPassBelow'].fillna(143, inplace=True)

        df['RampFirstSpike'].fillna(5000, inplace=True)

        for index, row in df.iterrows():

            # No APs
            if (row['AP1Amplitude'] == 0 and row['AP2Amplitude'] == 0):
                df.at[index, 'AP12AmplitudeDrop'] = 0
                df.at[index, 'AP12AmplitudeChangePercent'] = 0
                df.at[index, 'AP1SSAmplitudeChange'] = 0
                df.at[index, 'AP12HalfWidthChangePercent'] = 0
                df.at[index, 'AP12RateOfChangePeakToTroughPercentChange'] = 0
                df.at[index, 'AP12AHPDepthPercentChange'] = 0
                df.at[index, 'InitialAccommodationMean'] = 0
                df.at[index, 'SSAccommodationMean'] = 0
                df.at[index, 'AccommodationRateToSS'] = 0
                df.at[index, 'AccommodationAtSSMean'] = 0

            # Only 1 AP
            if (row['AP1Amplitude'] > 0 and row['AP2Amplitude'] == 0):
                df.at[index, 'AP12AmplitudeDrop'] = row['AP1Amplitude']
                df.at[index, 'AP12AmplitudeChangePercent'] = -100
                df.at[index, 'AP12HalfWidthChangePercent'] = -100
                df.at[index, 'AP12RateOfChangePeakToTroughPercentChange'] = -100
                df.at[index, 'AP12AHPDepthPercentChange'] = -100
                df.at[index, 'AccommodationRateToSS'] = -1
                df.at[index, 'AccommodationAtSSMean'] = -100

            # 1 AP and no SS APs
            if row['AP1SSAmplitudeChange'] == 0 and row['AP1Amplitude'] > 0:
                df.at[index, 'AP1SSAmplitudeChange'] = row['AP1Amplitude']

            if np.isnan(row['AccommodationRateToSS']):
                df.at[index, 'AccommodationRateToSS'] = -1

            if np.isnan(row['AccommodationAtSSMean']):
                df.at[index, 'AccommodationAtSSMean'] = -100

        df['AP12AmplitudeDrop'] = df['AP12AmplitudeDrop'].apply(lambda x: np.log(10 + x))
        df['AP12AmplitudeChangePercent'] = df['AP12AmplitudeChangePercent'].apply(
            lambda x: np.log(-x + 10 + np.abs(np.max(df['AP12AmplitudeChangePercent']))))
        df['AP1SSAmplitudeChange'] = df['AP1SSAmplitudeChange'].apply(
            lambda x: np.log(x + 10 + np.abs(np.min(df['AP1SSAmplitudeChange']))))
        df['AP1WidthHalfHeight'] = df['AP1WidthHalfHeight'].apply(
            lambda x: np.log(x + 0.01 + np.abs(np.min(df['AP1WidthHalfHeight']))))
        df['AP2WidthHalfHeight'] = df['AP2WidthHalfHeight'].apply(
            lambda x: np.log(x + 0.01 + np.abs(np.min(df['AP2WidthHalfHeight']))))
        df['AP1WidthPeakToTrough'] = df['AP1WidthPeakToTrough'].apply(
            lambda x: np.log(x + 1 + np.abs(np.min(df['AP1WidthPeakToTrough']))))
        df['AP2WidthPeakToTrough'] = df['AP2WidthPeakToTrough'].apply(
            lambda x: np.log(x + 1 + np.abs(np.min(df['AP2WidthPeakToTrough']))))
        df['AP1RateOfChangePeakToTrough'] = df['AP1RateOfChangePeakToTrough'].apply(
            lambda x: np.log(-x + 1 + np.abs(np.max(df['AP1RateOfChangePeakToTrough']))))
        df['AP2RateOfChangePeakToTrough'] = df['AP2RateOfChangePeakToTrough'].apply(
            lambda x: np.log(-x + 1 + np.abs(np.max(df['AP2RateOfChangePeakToTrough']))))
        df['AP12RateOfChangePeakToTroughPercentChange'] = df['AP12RateOfChangePeakToTroughPercentChange'].apply(
            lambda x: np.log(x + 10 + np.abs(np.min(df['AP12RateOfChangePeakToTroughPercentChange']))))
        df['AP12AHPDepthPercentChange'] = df['AP12AHPDepthPercentChange'].apply(
            lambda x: np.log(x + 10 + np.abs(np.min(df['AP12AHPDepthPercentChange']))))
        df['AP1DelayMean'] = df['AP1DelayMean'].apply(lambda x: np.log(x + 1 + np.abs(np.min(df['AP1DelayMean']))))
        df['AP2DelayMean'] = df['AP2DelayMean'].apply(lambda x: np.log(x + 1 + np.abs(np.min(df['AP2DelayMean']))))
        df['AP1DelayMeanStrongStim'] = df['AP1DelayMeanStrongStim'].apply(
            lambda x: np.log(x + 1 + np.abs(np.min(df['AP1DelayMeanStrongStim']))))
        df['AP2DelayMeanStrongStim'] = df['AP2DelayMeanStrongStim'].apply(
            lambda x: np.log(x + 1 + np.abs(np.min(df['AP2DelayMeanStrongStim']))))
        df['Burst1ISIMean'] = df['Burst1ISIMean'].apply(
            lambda x: np.log(x + 1 + np.abs(np.min(df['Burst1ISIMean']))))
        df['Burst1ISIMeanStrongStim'] = df['Burst1ISIMeanStrongStim'].apply(
            lambda x: np.log(x + 1 + np.abs(np.min(df['Burst1ISIMeanStrongStim']))))
        df['ISIMedian'] = df['ISIMedian'].apply(lambda x: np.log(x + 1 + np.abs(np.min(df['ISIMedian']))))
        df['ISICV'] = df['ISICV'].apply(lambda x: np.log(x + 1 + np.abs(np.min(df['ISICV']))))
        df['ISIBurstMeanChange'] = df['ISIBurstMeanChange'].apply(
            lambda x: np.log(x + 100 + np.abs(np.min(df['ISIBurstMeanChange']))))
        df['SpikeRateStrongStim'] = df['SpikeRateStrongStim'].apply(
            lambda x: np.log(x + 1 + np.abs(np.min(df['SpikeRateStrongStim']))))
        df['InputResistance'] = df['InputResistance'].apply(
            lambda x: np.log(x + 10 + np.abs(np.min(df['InputResistance']))))
        df['SteadyStateAPs'] = df['SteadyStateAPs'].apply(
            lambda x: np.log(x + 1 + np.abs(np.min(df['SteadyStateAPs']))))
        df['RampFirstSpike'] = df['RampFirstSpike'].apply(
            lambda x: np.log(x + 1 + np.abs(np.min(df['RampFirstSpike']))))
        df['FrequencyPassAbove'] = df['FrequencyPassAbove'].apply(
            lambda x: np.log(x + 1 + np.abs(np.min(df['FrequencyPassAbove']))))
        df['FrequencyPassBelow'] = df['FrequencyPassBelow'].apply(
            lambda x: np.log(x + 1 + np.abs(np.min(df['FrequencyPassBelow']))))

        return df, prop_names

    def get_similar_ephyz_cells(self, cell_id):
        self.server.connect()

        all_cells = list(Cells \
                         .select(Cells, Model_Waveforms.Spikes, Models.Name) \
                         .join(Model_Waveforms, on=(Cells.Model_ID == Model_Waveforms.Model_id)) \
                         .join(Models, on=(Cells.Model_ID == Models.Model_ID)) \
                         .where((Model_Waveforms.Protocol == "STEADY_STATE") & (Model_Waveforms.Variable_Name == "Voltage")) \
                         .order_by(Cells.Model_ID)
                         .objects()
                         )

        df_all, prop_names = self.get_pre_processed_cell_ephyz_props(all_cells)

        from sklearn.externals import joblib

        scaler = joblib.load("ephyz_scaler.pkl")
        x = scaler.transform(df.loc[:, prop_names].values)
        x = DataFrame(x, columns=prop_names)

        pca = joblib.load("ephyz_pca.pkl")
        principalComponents = pca.transform(x)

        principalDf = DataFrame(data=principalComponents)
        X = principalDf

    def cluster_cell_ephyz(self, plot=False):
        """
        Assigns all cells to clusters based on ephyz properties
        """

        # import pydevd
        # pydevd.settrace('192.168.0.34', port=4200, suspend=False)

        pandas.set_option('display.max_columns', None)
        pandas.set_option('display.max_rows', 20)

        self.server.connect()

        all_cells = list(Cells \
                     .select(Cells, Model_Waveforms.Spikes, Models.Name) \
                     .join(Model_Waveforms, on=(Cells.Model_ID == Model_Waveforms.Model_id)) \
                     .join(Models, on=(Cells.Model_ID == Models.Model_ID)) \
                     .where((Model_Waveforms.Protocol == "STEADY_STATE") & (Model_Waveforms.Variable_Name == "Voltage")) \
                     .order_by(Cells.Model_ID)
                     .objects()
                     )

        df_all, prop_names = self.get_pre_processed_cell_ephyz_props(all_cells)

        cluster_names = ["root", "multi_spikers", "multi_spikers_sub_0", "multi_spikers_sub_1"]

        for cluster in cluster_names:
            if cluster == 'root':
                df = df_all # 1st pass
                cluster_level = 'Root_Cluster'

            elif cluster == 'multi_spikers':
                df = df_all[df_all["Root_Cluster"] == 1] # 2nd pass
                cluster_level = "Multi_Spike_Cluster"

            elif cluster == 'multi_spikers_sub_0':
                df = df_all[df_all["Multi_Spike_Cluster"] == 0] # Left of 2nd pass
                cluster_level = "Multi_Spike_0_Cluster"

            else: #multi_spikers_sub_1
                df = df_all[df_all["Multi_Spike_Cluster"] == 1]  # Right of 2nd pass
                cluster_level = "Multi_Spike_1_Cluster"

            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            x = scaler.fit_transform(df.loc[:, prop_names].values)

            x = DataFrame(x, columns=prop_names)
            print('start dims', len(prop_names))
            from sklearn.decomposition import PCA
            pca = PCA(svd_solver='full', n_components=0.95)
            principalComponents = pca.fit_transform(x)

            principalDf = DataFrame(data=principalComponents)
            X = principalDf
            print('post-pca dims', len(principalDf.columns))

            if cluster == "root":
                root_pcs = principalComponents


            if plot:
                from mpl_toolkits import mplot3d
                plt.axes(projection='3d')
                plt.plot(X[0], X[1], X[2], 'bo')

                from scipy.cluster.hierarchy import dendrogram, linkage
                from matplotlib import pyplot as plt

                linked = linkage(X, 'ward', optimal_ordering=False)
                plt.figure(figsize=(15, 7))
                dendrogram(linked,
                           orientation='top',
                           distance_sort='acending',
                           show_leaf_counts=True,
                           truncate_mode='lastp',
                           # p=5,
                           show_contracted=True,
                           )
                plt.show()


            from sklearn.cluster import AgglomerativeClustering

            cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
            cluster.fit_predict(X)

            if plot:
                plt.scatter(X[0], X[1], c=cluster.labels_, cmap='rainbow')
                plt.show()

            # Set subset df cluster
            df[cluster_level] = cluster.labels_

            # Set main df cluster
            df_all[cluster_level] = df[cluster_level]

        def nan_to_none(value):
            import math
            return None if math.isnan(value) else value

        import pydevd
        pydevd.settrace('192.168.0.34', port=4200, suspend=False)

        for index, cell in enumerate(all_cells):
            print("Saving cell",cell.Model_ID,"cluster")
            cell_record = Cells.get_or_none(Cells.Model_ID == cell.Model_ID)

            cell_record.RootCluster = nan_to_none(df_all.at[index, 'Root_Cluster'])
            cell_record.MultiSpikeCluster = nan_to_none(df_all.at[index, 'Multi_Spike_Cluster'])
            cell_record.MultiSpikeClusterSub0 = nan_to_none(df_all.at[index, 'Multi_Spike_0_Cluster'])
            cell_record.MultiSpikeClusterSub1 = nan_to_none(df_all.at[index, 'Multi_Spike_1_Cluster'])

            cell_record.save()


            from scipy.spatial.distance import euclidean
            target = root_pcs[index]

            df = DataFrame(root_pcs)
            df["dist"] = np.apply_along_axis(euclidean, 1, root_pcs, target)
            top_n = df.sort_values(by=['dist']).head(11).index[1:]

            Cells_Similar_Ephyz.delete().where(Cells_Similar_Ephyz.Parent_Cell == cell.Model_ID).execute()

            max_dist = df['dist'].max()

            for similar_cell_loc in top_n:
                similar_cell = all_cells[similar_cell_loc]

                record = Cells_Similar_Ephyz(
                    Parent_Cell=cell.Model_ID,
                    Similar_Cell=similar_cell.Model_ID,
                    Similarity = 1.0 - df.loc[similar_cell_loc]['dist']/max_dist
                )

                record.save()


    def replace_tokens(self, target, reps):
        result = target
        for r in reps.keys():
            result = result.replace(r, str(reps[r]))
        return result
