import lmdb
from common.registry import Registry
import os
import sys
from numpy import array as array
from numpy import float32 as float32
from numpy import int64 as int64


@Registry.register_dataset('onfly')
class OnFlyDataset(object):
    def __init__(self, path):
        self.path = path
        self.num_samples = 0
        self.ep = -1
        self.act = -1
        self.env = None
        self.txn = None

    def initiate(self, ep, step, interval):
        self.env = lmdb.open(self.path, subdir=True, map_size=1073741824)
        self.ep = ep
        self.act = step//interval

    def flush(self, ldq):
        self.txn = self.env.begin(write=True)
        for dq in ldq:
            for item in dq:
                self.txn.put(item[0].encode(), str(item[1]).encode())
            self.num_samples += 1
        self.txn.commit()
        self.txn = None

    def finalize(self):
        self.env.close()
        self.env = None

    def insert(self, key, value):
        self.txn = self.env.begin(write=True)
        self.txn.put(key.encode(), str(value).encode())
        self.txn.commit()
        self.txn = None
        self.num_samples += 1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        epoch = idx // self.act
        step = (idx % self.act) + 1  # step is the action taken. From 1 to 360 if self.action_interval == 10
        key = f'{epoch}_{step}'
        return self.search(key)

    def delete(self, key):
        self.txn.delete(key.encode())
        self.txn.commmit()
        self.txn = None

    def _search(self):
        self.txn = self.env.begin()

    def search(self, key):
        if self.txn is None:
            self.txn = self.env.begin()
        val = self.txn.get(key.encode())
        result = eval(val.decode())
        return result

