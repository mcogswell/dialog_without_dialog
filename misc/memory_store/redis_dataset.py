import numpy as np
import redis
import os.path as pth
import json
import socket
from pprint import pprint


class RedisDataset:
    def __init__(self, key, dataset, sep='|', verbose=False):
        self.dataset = dataset
        self.key = str(key)
        assert sep not in self.key
        self.sep = sep
        self.dtype = self.dataset.dtype
        self.shape = self.dataset.shape[1:]
        self.hits = 0
        self.misses = 0
        self.verbose = verbose

        # figure out how to get data from redis, if possible
        self.conn_failed = False
        # load redis node config
        try:
            with open('.redis_config2.json', 'r') as f:
                config = json.load(f)
            self.config = config
        except BaseException as e:
            print(e)
            print('Could not load configuration. Falling back to file')
            self.conn_failed = True
        # find a replica
        try:
            self.replica = self.connect_to_best_replica()
        except BaseException as e:
            print(e)
            print('Could not connect. Falling back to direct from file')
            self.conn_failed = True
        self._master = None

    def __getitem__(self, idx):
        if self.conn_failed:
            return self.dataset[idx]
        key = self.key + '_' + str(idx)
        try:
            if key in self.replica:
                self.hits += 1
                value = self.replica.get(key)
                result = np.fromstring(value, dtype=self.dtype).reshape(*self.shape)
            else:
                self.misses += 1
                result = self.dataset[idx]
                value = result.ravel().tostring()
                self.master.set(key, value)
        except BaseException as e:
            print(e)
            print('Falling back to loading directly from hdf5 dataset')
            self.conn_failed = True
            return self.dataset[idx]

        if self.hits + self.misses > 20000:
            if self.verbose:
                print(f'hit rate: {self.hits / (self.hits + self.misses):.3f}')
            self.hits = self.misses = 0
        return result

    @property
    def master(self):
        if self._master is not None:
            return self._master

        replication_info = self.replica.info('replication')
        if replication_info['role'] == 'master':
            self._master = self.replica
            return self._master

        master_host = replication_info['master_host']
        master_port = replication_info['master_port']
        password = self.config['password']
        self._master = redis.Redis(host=master_host,
                                   port=master_port,
                                   password=password)
        return self._master

    def connect_to_best_replica(self):
        config = self.config
        password = config['password']
        connections = []
        error = Exception('No viable redis hosts')
        fqdn = socket.getfqdn()

        # connect to all the working replicas
        if len(config['replicas']) == 0:
            raise Exception('There must be at least one replica!')
        for host, port in config['replicas']:
            try:
                conn = redis.Redis(host=host, port=port, password=password)
                if fqdn == host:
                    if self.verbose:
                        print('Using replica: ', conn)
                    return conn
                connections.append(conn)
            except redis.ConnectionError as e:
                error = e
        if len(connections) == 0:
            raise error

        # pick the replica with least clients
        min_clients = np.inf
        replica = None
        for c in connections:
            num_clients = c.info('clients')['connected_clients']
            if num_clients < min_clients:
                replica = c
                min_clients = num_clients

        if self.verbose:
            print('Using replica: ', replica)
        return replica
