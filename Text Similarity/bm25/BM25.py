#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

import math
from six import iteritems
from six.moves import xrange
import jieba.posseg as pseg
from gensim import corpora
import os
import re
import jieba

# BM25 parameters.
PARAM_K1 = 1.5
PARAM_B = 0.75
EPSILON = 0.25


class BM25(object):

    def __init__(self, corpus):
        self.corpus_size = len(corpus)
        self.avgdl = sum(map(lambda x: float(len(x)), corpus)) / self.corpus_size
        self.corpus = corpus
        self.f = []
        self.df = {}
        self.idf = {}
        self.initialize()

    def initialize(self):
        for document in self.corpus:
            frequencies = {}
            for word in document:
                if word not in frequencies:
                    frequencies[word] = 0
                frequencies[word] += 1
            self.f.append(frequencies)

            for word, freq in iteritems(frequencies):
                if word not in self.df:
                    self.df[word] = 0
                self.df[word] += 1

        for word, freq in iteritems(self.df):
            self.idf[word] = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)

    def get_score(self, document, index, average_idf):
        score = 0
        for word in document:
            if word not in self.f[index]:
                continue
            idf = self.idf[word] if self.idf[word] >= 0 else EPSILON * average_idf
            score += (idf * self.f[index][word] * (PARAM_K1 + 1)
                      / (self.f[index][word] + PARAM_K1 * (1 - PARAM_B + PARAM_B * self.corpus_size / self.avgdl)))
        return score

    def get_scores(self, document, average_idf):
        scores = []
        for index in xrange(self.corpus_size):
            score = self.get_score(document, index, average_idf)
            scores.append(score)
        return scores


def get_bm25_weights(corpus):
    bm25 = BM25(corpus)
    average_idf = sum(map(lambda k: float(bm25.idf[k]), bm25.idf.keys())) / len(bm25.idf.keys())
    weights = []
    for doc in corpus:
        scores = bm25.get_scores(doc, average_idf)
        weights.append(scores)

    return weights


stop_flag = ['x', 'c', 'u', 'd', 'p', 't', 'uj', 'm', 'f', 'r']
stopwords = []


def tokenization(filename):
    result = []
    with open(filename, 'r') as f:
        text = f.read()
        words = pseg.cut(text)
    for word, flag in words:
        if flag not in stop_flag and word not in stopwords:
            result.append(word)
    return result


corpus = []
dirname = 'document/'
filenames = []
for root, dirs, files in os.walk(dirname):
    for f in files:
        corpus.append(tokenization(dirname + f))
        filenames.append(f)

dictionary = corpora.Dictionary(corpus)

doc_vectors = [dictionary.doc2bow(text) for text in corpus]
vec1 = doc_vectors[0]
vec1_sorted = sorted(vec1, key=lambda x: x[1], reverse=True)
for term, freq in vec1_sorted[:5]:
    print(dictionary[term])

bm25Model = BM25(corpus)
average_idf = sum(map(lambda k: float(bm25Model.idf[k]), bm25Model.idf.keys())) / len(bm25Model.idf.keys())
query_str = '?????????'
query = []
for word in query_str.strip().split():
    query.append(word)
scores = bm25Model.get_scores(query, average_idf)
# scores.sort(reverse=True)
print(scores)
idx = scores.index(max(scores))
fname = filenames[idx]
print(fname)
