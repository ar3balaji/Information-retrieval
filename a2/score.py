""" Assignment 2
"""
import abc
from collections import defaultdict
from math import sqrt,log10

import index
"""
assignment 2 by:
Balaji A R
CWID A20347964
"""

def idf(term, index):
    """ Compute the inverse document frequency of a term according to the
    index. IDF(T) = log10(N / df_t), where N is the total number of documents
    in the index and df_t is the total number of documents that contain term
    t.

    Params:
      terms....A string representing a term.
      index....A Index object.
    Returns:
      The idf value.

    >>> idx = index.Index(['a b c a', 'c d e', 'c e f'])
    >>> idf('a', idx) # doctest:+ELLIPSIS
    0.477...
    >>> idf('d', idx) # doctest:+ELLIPSIS
    0.477...
    >>> idf('e', idx) # doctest:+ELLIPSIS
    0.176...
    """
    return log10(len(index.documents) / index.doc_freqs[term])

class ScoringFunction:
    """ An Abstract Base Class for ranking documents by relevance to a
    query. """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def score(self, query_vector, index):
        """
        Do not modify.

        Params:
          query_vector...dict mapping query term to weight.
          index..........Index object.
        """
        return


class RSV(ScoringFunction):
    """
    See lecture notes for definition of RSV.

    idf(a) = log10(3/1)
    idf(d) = log10(3/1)
    idf(e) = log10(3/2)
    >>> idx = index.Index(['a b c', 'c d e', 'c e f'])
    >>> rsv = RSV()
    >>> rsv.score({'a': 1.}, idx)[1]  # doctest:+ELLIPSIS
    0.4771...
    """

    def score(self, query_vector, index):
        tempDict = defaultdict(int)
        temp = 0
        for doc_id in range(0, len(index.documents)):
            for word in query_vector:
                if word in index.documents[doc_id]:
                    temp += log10(float(len(index.documents))/float(index.doc_freqs[word]))
            tempDict[doc_id + 1] = temp
            temp = 0

        return dict(tempDict)

    def __repr__(self):
        return 'RSV'


class BM25(ScoringFunction):
    """
    See lecture notes for definition of BM25.

    log10(3) * (2*2) / (1(.5 + .5(4/3.333)) + 2) = log10(3) * 4 / 3.1 = .6156...
    >>> idx = index.Index(['a a b c', 'c d e', 'c e f'])
    >>> bm = BM25(k=1, b=.5)
    >>> bm.score({'a': 1.}, idx)[1]  # doctest:+ELLIPSIS
    0.61564032...
    """
    def __init__(self, k=1, b=.5):
        self.k = k
        self.b = b

    def score(self, query_vector, index):
        tempDict = defaultdict(int)
        temp = 0
        for doc_id in range(0, len(index.documents)):
            for word in query_vector:
                if word in index.documents[doc_id]:
                    tf = index.documents[doc_id].count(word)
                    temp += (idf(word,index) * ((self.k + 1) * tf)) / ((self.k * ((1-self.b)+(self.b * (len(index.documents[doc_id])/index.mean_doc_length)))) + tf)
            tempDict[doc_id + 1] = temp
            temp = 0
        return dict(tempDict)

    def __repr__(self):
        return 'BM25 k=%d b=%.2f' % (self.k, self.b)


class Cosine(ScoringFunction):
    """
    See lecture notes for definition of Cosine similarity.  Be sure to use the
    precomputed document norms (in index), rather than recomputing them for
    each query.

    >>> idx = index.Index(['a a b c', 'c d e', 'c e f'])
    >>> cos = Cosine()
    >>> cos.score({'a': 1.}, idx)[1]  # doctest:+ELLIPSIS
    0.792857...
    """
    def score(self, query_vector, index):
        tempDict = defaultdict(int)
        for word in query_vector:
            for tempList in index.index[word]:
                tempDict[tempList[0]] += query_vector[word] * float((1 + log10(tempList[1])) * (idf(word, index)))

        for doc_id in tempDict.keys():
            tempDict[doc_id] /=  index.doc_norms[doc_id]
        return dict(tempDict)

    def __repr__(self):
        return 'Cosine'
