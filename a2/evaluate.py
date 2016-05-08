""" Assignment 2
"""
import abc

import numpy as np

"""
assignment 2 by:
Balaji A R
CWID A20347964
"""
class EvaluatorFunction:
    """
    An Abstract Base Class for evaluating search results.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def evaluate(self, hits, relevant):
        """
        Do not modify.
        Params:
          hits...A list of document ids returned by the search engine, sorted
                 in descending order of relevance.
          relevant...A list of document ids that are known to be
                     relevant. Order is insignificant.
        Returns:
          A float indicating the quality of the search results, higher is better.
        """
        return


class Precision(EvaluatorFunction):

    def evaluate(self, hits, relevant):
        """
        Compute precision.

        >>> Precision().evaluate([1, 2, 3, 4], [2, 4])
        0.5
        """
        tp = 0.0
        fp = 0.0
        precision = 0.0
        for i in range(0,len(hits)):
            if hits[i] in relevant:
                tp +=1
            else:
                fp +=1
        den = tp + fp
        if den != 0:
            precision = tp / den
        return precision


    def __repr__(self):
        return 'Precision'


class Recall(EvaluatorFunction):

    def evaluate(self, hits, relevant):
        """
        Compute recall.

        >>> Recall().evaluate([1, 2, 3, 4], [2, 5])
        0.5
        """
        tp = 0.0
        fp = 0.0
        fn = 0.0
        recall = 0.0
        for i in range(0,len(hits)):
            if hits[i] in relevant:
                tp += 1
            else:
                fp += 1
        fn = len(relevant) - tp

        den = tp + fn
        if den != 0:
            recall = tp / den
        return recall


    def __repr__(self):
        return 'Recall'


class F1(EvaluatorFunction):
    def evaluate(self, hits, relevant):
        """
        Compute F1.

        >>> F1().evaluate([1, 2, 3, 4], [2, 5])  # doctest:+ELLIPSIS
        0.333...
        """
        tp = 0.0
        fp = 0.0
        fn = 0.0
        precision = 0.0
        recall = 0.0
        result = 0.0

        for i in range(0,len(hits)):
            if hits[i] in relevant:
                tp +=1
            else:
                fp +=1

        fn = len(relevant) - tp
        pre_den = tp + fp
        re_de  = tp + fn
        if pre_den != 0:
            precision = tp / pre_den
        if re_de !=0:
            recall = tp / re_de

        if precision + recall !=0:
            result = (2 * precision * recall) / (precision + recall)
        return result

    def __repr__(self):
        return 'F1'


class MAP(EvaluatorFunction):
    def evaluate(self, hits, relevant):
        """
        Compute Mean Average Precision.

        >>> MAP().evaluate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 4, 6, 11, 12, 13, 14, 15, 16, 17])
        0.2
        """
        temp = 0.0
        rel = 0.0
        for i in range(0,len(hits)):
            if hits[i] in relevant:
                rel += 1
                temp += (rel/(i+1))
        return temp / len(relevant)


    def __repr__(self):
        return 'MAP'

