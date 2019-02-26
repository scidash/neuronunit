import hashlib
import os


def get_file_checksum(path):
    """
    Computes the MD5 hash of file contents
    :param path: path to the file
    :return: Hash value e.g. 391306c88b6f957a9d97a435bfc82e0d. If path doesn't exists, returns None.
    """

    if os.path.exists(path):
        with open(path, 'rb') as file:
            return hashlib.md5(file.read()).hexdigest()

    return None


def is_nmldb_id(value):
    """
    Returns true if the passed-in string is formatted as a NMLDB ID.
    Checks if the string has the correct beginning and length.
    :param value: A string to test e.g. 'NMLCH000001'
    :return: True/False
    """
    return value.startswith("NML") and len(value) == 11


def is_nml2_file(file_name):
    """
    Checks if the passed-in file name has a NeuroML2 extension (.nml)
    :param file_name: Name of a file e.g. 'Gran_NaF_98.channel.nml'
    :return: True/False
    """
    return file_name.endswith(".nml")