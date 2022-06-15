import hashlib
import struct
import string
from scipy import integrate
import numpy as np
from collections import defaultdict
import random


def sha1_hash32(data):
    return struct.unpack('<I', hashlib.sha1(data).digest()[:4])[0]


_mersenne_prime = (1 << 61) - 1
_max_hash = (1 << 32) - 1
_hash_range = (1 << 32)


class MinHash(object):

    def __init__(self, d=128, seed=1, hashfunc=sha1_hash32, hashvalues=None, permutations=None):
        if hashvalues is not None:
            d = len(hashvalues)
        self.seed = seed
        # Check the hash function.
        if not callable(hashfunc):
            raise ValueError("The hashfunc must be a callable.")
        self.hashfunc = hashfunc

        # Initialize hash values
        if hashvalues is not None:
            self.hashvalues = self._parse_hashvalues(hashvalues)
        else:
            self.hashvalues = self._init_hashvalues(d)
        if permutations is not None:
            self.permutations = permutations
        else:
            generator = np.random.RandomState(self.seed)
            self.permutations = np.array([(generator.randint(1, _mersenne_prime, dtype=np.uint64),
                                           generator.randint(0, _mersenne_prime, dtype=np.uint64))
                                          for _ in range(d)], dtype=np.uint64).T
        if len(self) != len(self.permutations[0]):
            raise ValueError("Numbers of hash values and permutations mismatch")

    def _init_hashvalues(self, d):
        return np.ones(d, dtype=np.uint64) * _max_hash

    def _parse_hashvalues(self, hashvalues):
        return np.array(hashvalues, dtype=np.uint64)

    def add(self, b):

        hv = self.hashfunc(b)
        a, b = self.permutations
        phv = np.bitwise_and((a * hv + b) % _mersenne_prime, np.uint64(_max_hash))
        self.hashvalues = np.minimum(phv, self.hashvalues)

    def jaccard(self, other):

        if other.seed != self.seed:
            raise ValueError("different seeds")
        if len(self) != len(other):
            raise ValueError("different numbers of permutation functions")
        return np.float(np.count_nonzero(self.hashvalues == other.hashvalues)) / np.float(len(self))

    def __len__(self):
        return len(self.hashvalues)

    def __eq__(self, other):
        return type(self) is type(other) and self.seed == other.seed and np.array_equal(self.hashvalues,
                                                                                        other.hashvalues)


class DictListStorage():

    def __getitem__(self, key):
        return self.get(key)

    def __delitem__(self, key):
        return self.remove(key)

    def __len__(self):
        return self.size()

    def __iter__(self):
        for key in self.keys():
            yield key

    def __init__(self, config, name):
        self._dict = defaultdict(list)

    def keys(self):
        return self._dict.keys()

    def get(self, key):
        return self._dict.get(key, [])

    def insert(self, key, *vals, **kwargs):
        self._dict[key].extend(vals)

    def size(self):
        return len(self._dict)

    def itemcounts(self, **kwargs):
        return {k: len(v) for k, v in self._dict.items()}

    def has_key(self, key):
        return key in self._dict


class DictSetStorage():
    def __init__(self, config, name):
        self._dict = defaultdict(set)

    def get(self, key):
        return self._dict.get(key, set())

    def insert(self, key, *vals, **kwargs):
        self._dict[key].update(vals)


def _random_name(length):
    return ''.join(random.choice(string.ascii_lowercase)
                   for _ in range(length)).encode('utf8')


def _false_positive_probability(threshold, b, r):
    _probability = lambda s: 1 - (1 - s ** float(r)) ** float(b)
    a, err = integrate(_probability, 0.0, threshold)
    return a


def _false_negative_probability(threshold, b, r):
    _probability = lambda s: 1 - (1 - (1 - s ** float(r)) ** float(b))
    a, err = integrate(_probability, threshold, 1.0)
    return a


def _optimal_param(threshold, num_perm, false_positive_weight,
                   false_negative_weight):
    min_error = float("inf")
    opt = (0, 0)
    for b in range(1, num_perm + 1):
        max_r = int(num_perm / b)
        for r in range(1, max_r + 1):
            fp = _false_positive_probability(threshold, b, r)
            fn = _false_negative_probability(threshold, b, r)
            error = fp * false_positive_weight + fn * false_negative_weight
            if error < min_error:
                min_error = error
                opt = (b, r, fp, fn)
    return opt


class MinHashLSH(object):

    def __init__(self, threshold=0.9, d=128, weights=(0.5, 0.5),
                 params=None, storage_config=None):
        if storage_config is None:
            storage_config = {'type': 'dict'}

        if sum(weights) != 1.0:
            raise ValueError("Weights must sum to 1.0")
        self.h = d
        if params is not None:
            self.b, self.r = params
            if self.b * self.r > d:
                raise ValueError("The product of b and r in params is "
                                 "{} * {} = {} -- it must be less than d {}. ".format(self.b, self.r, self.b * self.r,
                                                                                      d))
        else:
            false_positive_weight, false_negative_weight = weights
            self.b, self.r, self.fp, self.fn = _optimal_param(threshold, d, false_positive_weight,
                                                              false_negative_weight)
            print('the best parameter b={},r={},fp={},fn={}'.format(self.b, self.r, self.fp, self.fn))

        basename = storage_config.get('basename', _random_name(11))
        self.hashtables = []
        self.hashranges = []
        for i in range(self.b):
            name = b''.join([basename, b'_bucket_', struct.pack('>H', i)])
            item = DictSetStorage(storage_config, name=name)
            self.hashtables.append(item)

            self.hashranges.append((i * self.r, (i + 1) * self.r))

        self.keys = DictListStorage(storage_config, name=b''.join([basename, b'_keys']))

    def insert(self, key, minhash):
        self._insert(key, minhash, buffer=False)

    def _insert(self, key, minhash, buffer=False):
        if key in self.keys:
            raise ValueError("key already exists")
        Hs = []
        for start, end in self.hashranges:
            Hs.append(self._H(minhash.hashvalues[start:end]))

        self.keys.insert(key, *Hs, buffer=buffer)

        for H, hashtable in zip(Hs, self.hashtables):
            hashtable.insert(H, key, buffer=buffer)

    def query(self, minhash):
        candidates = set()
        for (start, end), hashtable in zip(self.hashranges, self.hashtables):
            H = self._H(minhash.hashvalues[start:end])
            for key in hashtable.get(H):
                candidates.add(key)

        return list(candidates)

    def _H(self, hs):
        return bytes(hs.byteswap().data)
