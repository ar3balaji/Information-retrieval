"""
Assignment 5: K-Means. See the instructions to complete the methods below.

Balaji A R
A20347964
"""

from collections import Counter,defaultdict
import gzip
import math

import numpy as np


class KMeans(object):

    def __init__(self, k=2):
        """ Initialize a k-means clusterer. Should not have to change this."""
        self.k = k

    def cluster(self, documents, iters=10):
        """
        Cluster a list of unlabeled documents, using iters iterations of k-means.
        Initialize the k mean vectors to be the first k documents provided.
        After each iteration, print:
        - the number of documents in each cluster
        - the error rate (the total Euclidean distance between each document and its assigned mean vector), rounded to 2 decimal places.
        See Log.txt for expected output.
        The order of operations is:
        1) initialize means
        2) Loop
          2a) compute_clusters
          2b) compute_means
          2c) print sizes and error
        """
        self.mean_documents = documents[0:self.k]
        self.documents = documents

        self.mean_vector = []
        for doc in self.mean_documents:
            self.mean_vector.append(float(np.dot(list(doc.values()), list(doc.values()))))

        for i in range(0, iters):
            self.clusters = defaultdict(list)
            print(self.compute_clusters(documents))
            self.mean_documents = self.compute_means()
            print("%.2f" % self.error(documents))

        self.clusters = defaultdict(list)
        self.compute_clusters(documents)

    def compute_means(self):
        """ Compute the mean vectors for each cluster (results stored in an
        instance variable of your choosing)."""
        self.mean_vector = []

        for doc in self.mean_documents:
            self.mean_vector.append(float(np.dot(list(doc.values()), list(doc.values()))))

        for cluster_id, values in self.clusters.items():
            temp_counter = Counter()
            for document_index, dist in values:
                temp_counter.update(self.documents[document_index])
            for key, value in temp_counter.items():
                temp_counter[key] = float(value) / float(len(values))
            self.mean_documents[cluster_id] = temp_counter

        return self.mean_documents

    def compute_clusters(self, documents):
        """ Assign each document to a cluster. (Results stored in an instance
        variable of your choosing). """
        for document_index in range(0, len(documents)):
            distance = []
            for cluster_id in range(0, self.k):
                distance.append((document_index, self.distance(documents[document_index], self.mean_documents[cluster_id], self.mean_vector[cluster_id])))
            min_distance = min(distance, key = lambda x : x[1])
            self.clusters[distance.index(min_distance)].append(min_distance)

        result = []
        for clusterID in self.clusters.keys():
            result.append(len(self.clusters[clusterID]))

        return result

    def sqnorm(self, d):
        """ Return the vector length of a dictionary d, defined as the sum of
        the squared values in this dict. """
        pass

    def distance(self, doc, mean, mean_norm):
        """ Return the Euclidean distance between a document and a mean vector.
        See here for a more efficient way to compute:
        http://en.wikipedia.org/wiki/Cosine_similarity#Properties"""
        first = mean_norm
        second = float(sum([list(doc.values())[i] ** 2 for i in range(0, len(list(doc.values())))]))
        first_second = 0.0
        for key in doc.keys():
            if key in mean:
                first_second += mean[key] * doc[key]

        return float(math.sqrt(first + second - 2.0 * first_second))

    def error(self, documents):
        """ Return the error of the current clustering, defined as the total
        Euclidean distance between each document and its assigned mean vector."""
        result = 0.0
        self.mean_vector = []

        for doc in self.mean_documents:
            self.mean_vector.append(float(np.dot(list(doc.values()), list(doc.values()))))

        for clusterID, values in self.clusters.items():
            for docID, dist in values:
                result += self.distance(documents[docID], self.mean_documents[clusterID], self.mean_vector[clusterID])
        return round(result,2)

    def print_top_docs(self, n=10):
        """ Print the top n documents from each cluster. These are the
        documents that are the closest to the mean vector of each cluster.
        Since we store each document as a Counter object, just print the keys
        for each Counter (sorted alphabetically).
        Note: To make the output more interesting, only print documents with more than 3 distinct terms.
        See Log.txt for an example."""
        for clusterID, values in self.clusters.items():
            temp_list = sorted(values, key=lambda x: x[1])

            print('CLUSTER ' + str(clusterID))
            iter = 0
            if len(temp_list) > n:
                iter = n
            else:
                iter = len(temp_list)
            i = 0
            while i < iter:
                if len(self.documents[temp_list[i][0]]) > 3:
                    print(' '.join(sorted([k for k in self.documents[temp_list[i][0]]], key=lambda x: x)))
                    #print(' '.join([k for k in self.documents[temp_list[i][0]]]))
                else:
                    iter += 1
                i += 1


def prune_terms(docs, min_df=3):
    """ Remove terms that don't occur in at least min_df different
    documents. Return a list of Counters. Omit documents that are empty after
    pruning words.
    >>> prune_terms([{'a': 1, 'b': 10}, {'a': 1}, {'c': 1}], min_df=2)
    [Counter({'a': 1}), Counter({'a': 1})]
    """
    result = []
    each_term_count = Counter()

    for document in docs:
        for term, value in document.items():
            each_term_count[term] += 1

    for document in docs:
        temp = {}
        for term in document.keys():
            if each_term_count[term] >= min_df:
                temp[term]=document[term]
        if len(temp) > 0:
            result.append(Counter(temp))

    return result

def read_profiles(filename):
    """ Read profiles into a list of Counter objects.
    DO NOT MODIFY"""
    profiles = []
    with gzip.open(filename, mode='rt', encoding='utf8') as infile:
        for line in infile:
            profiles.append(Counter(line.split()))
    return profiles


def main():
    profiles = read_profiles('profiles.txt.gz')
    print('read', len(profiles), 'profiles.')
    profiles = prune_terms(profiles, min_df=2)
    km = KMeans(k=10)
    km.cluster(profiles, iters=20)
    km.print_top_docs()

if __name__ == '__main__':
    main()
