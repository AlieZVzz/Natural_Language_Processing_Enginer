from simhash import fingerprint
from simhash import hamming_distance

hash1 = fingerprint(map(hash, "some text we want to hash"))
hash2 = fingerprint(map(hash, "some more text we want to hash"))
hamming_distance(hash1, hash2)