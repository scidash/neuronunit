from playhouse.db_url import connect
from sshtunnel import SSHTunnelForwarder

from tables import *
from config import Config

class NMLDB(object):
    def __init__(self):
        self.config = Config()

        self.server = None
        self.db = None

    def connect(self):
        if self.db is not None:
            return self.db

        pwd = self.config.pwd

        if pwd == '':
            raise Exception("The environment variable 'NMLDBPWD' needs to contain the password to the NML database")

        server = SSHTunnelForwarder(
            (self.config.server_IP, 2200),
            ssh_username='neuromine',
            ssh_password=pwd,
            remote_bind_address=('127.0.0.1', 3306),
            set_keepalive=5.0
        )

        connected = False
        while not connected:
            try:
                print("Connecting to server...")
                server.start()
                connected = True

            except:
                print("Could not connect to server. Retrying in 0-60s...")
                import time
                from random import randint
                time.sleep(randint(0, 60))
                print("Retrying...")

        print("Connecting to MySQL database...")
        db = connect('mysql://neuromldb2:' + pwd + '@127.0.0.1:' + str(server.local_bind_port) + '/neuromldb')

        db_proxy.initialize(db)

        self.server = server
        self.db = db

        return db

    def close(self):
        if self.db is not None:
            self.db.close()
            self.db = None

        if self.server is not None:
            self.server.close()
            self.server = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

