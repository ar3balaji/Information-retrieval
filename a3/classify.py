"""
Assignment 3. Implement a Multinomial Naive Bayes classifier for spam filtering.

You'll only have to implement 3 methods below:

train: compute the word probabilities and class priors given a list of documents labeled as spam or ham.
classify: compute the predicted class label for a list of documents
evaluate: compute the accuracy of the predicted class labels.

Name Balaji A R
CWID A20347964
"""

from collections import defaultdict
from collections import Counter
import glob
import math
import os



class Document(object):
    """ A Document. Do not modify.
    The instance variables are:

    filename....The path of the file for this document.
    label.......The true class label ('spam' or 'ham'), determined by whether the filename contains the string 'spmsg'
    tokens......A list of token strings.
    """

    def __init__(self, filename=None, label=None, tokens=None):
        """ Initialize a document either from a file, in which case the label
        comes from the file name, or from specified label and tokens, but not
        both.
        """
        if label: # specify from label/tokens, for testing.
            self.label = label
            self.tokens = tokens
        else: # specify from file.
            self.filename = filename
            self.label = 'spam' if 'spmsg' in filename else 'ham'
            self.tokenize()

    def tokenize(self):
        self.tokens = ' '.join(open(self.filename).readlines()).split()


class NaiveBayes(object):
    prior = defaultdict(float)
    word_conditional_prob = defaultdict(defaultdict)
    classifier_classes = ['spam', 'ham']

    def get_word_probability(self, label, term):
        """
        Return Pr(term|label). This is only valid after .train has been called.

        Params:
          label: class label.
          term: the term
        Returns:
          A float representing the probability of this term for the specified class.

        >>> docs = [Document(label='spam', tokens=['a', 'b']), Document(label='spam', tokens=['b', 'c']), Document(label='ham', tokens=['c', 'd'])]
        >>> nb = NaiveBayes()
        >>> nb.train(docs)
        >>> nb.get_word_probability('spam', 'a')
        0.25
        >>> nb.get_word_probability('spam', 'b')
        0.375
        >>> nb.get_word_probability('spam', 'e')
        0.0
        >>> nb.get_word_probability('ham', 'e')
        0.0
        >>> nb.get_word_probability('test', 'd')
        0.0
        """
        result = 0.0
        if label in self.word_conditional_prob.keys():
            if term in self.word_conditional_prob[label].keys() :
                result = self.word_conditional_prob[label][term]
        return result

    def get_top_words(self, label, n):
        """ Return the top n words for the specified class, using the odds ratio.
        The score for term t in class c is: p(t|c) / p(t|c'), where c'!=c.

        Params:
          labels...Class label.
          n........Number of values to return.
        Returns:
          A list of (float, string) tuples, where each float is the odds ratio
          defined above, and the string is the corresponding term.  This list
          should be sorted in descending order of odds ratio.

        >>> docs = [Document(label='spam', tokens=['a', 'b']), Document(label='spam', tokens=['b', 'c']), Document(label='ham', tokens=['c', 'd'])]
        >>> nb = NaiveBayes()
        >>> nb.train(docs)
        >>> nb.get_top_words('spam', 2)
        [(2.25, 'b'), (1.5, 'a')]
        """
        temp_list =[]
        other_class = 'spam'
        if label is 'spam':
            other_class ='ham'
        for term in self.word_conditional_prob[label].keys():
            current_class_prob = float(self.word_conditional_prob[label][term])
            other_class_term_prob = float(self.word_conditional_prob[other_class][term])
            res = 0
            if other_class_term_prob != float(0):
                res = (current_class_prob)/(other_class_term_prob)
            temp_list.append((res,term))

        return sorted(temp_list, key=lambda item: item[0],reverse=True)[0:n]



    def train(self, documents):
        """
        Given a list of labeled Document objects, compute the class priors and
        word conditional probabilities, following Figure 13.2 of your
        book. Store these as instance variables, to be used by the classify
        method subsequently.
        Params:
          documents...A list of training Documents.
        Returns:
          Nothing.
        """
        terms = []
        total_class_documents = defaultdict(int)
        each_class_tokens = defaultdict(list)

        for class_type in self.classifier_classes:
            total_class_documents[class_type] = 0

        for document in range(0, len(documents)):
            terms = terms + documents[document].tokens
            total_class_documents[documents[document].label] += 1
            each_class_tokens[documents[document].label] +=  documents[document].tokens

        #   conver list to set, get rid of duplicates
        terms = set(terms)

        for current_class in self.classifier_classes:
            token_count = Counter()
            prob_term = defaultdict(float)
            total = 0

            self.prior[current_class] = float(total_class_documents[current_class]) / float(len(documents))

            # get each tocken count
            for token in each_class_tokens[current_class]:
                token_count[token] += 1

            for term in terms:
                total += (token_count[term] + 1)

            for term in terms:
                prob_term[term] = float(token_count[term] + 1) / float(total)

            self.word_conditional_prob[current_class] = prob_term

    def classify(self, documents):
        """ Return a list of strings, either 'spam' or 'ham', for each document.
        Params:
          documents....A list of Document objects to be classified.
        Returns:
          A list of label strings corresponding to the predictions for each document.
        """
        document_score = defaultdict(defaultdict)
        result = []

        for current_class in self.classifier_classes:

            score = defaultdict(float)

            for document in range(0, len(documents)):
                score[document] = math.log10(self.prior[current_class])
                for token in documents[document].tokens:
                    if self.word_conditional_prob[current_class][token] > 0:
                        score[document] += math.log10(self.word_conditional_prob[current_class][token])

            document_score[current_class] = score

        for document in range(0, len(documents)):
            last_score = 0
            last_class = ''
            for current_class in range(0,len(self.classifier_classes)):
                if current_class == 0:
                    last_score = document_score[self.classifier_classes[current_class]][document]
                    last_class = self.classifier_classes[current_class]
                current_score = document_score[self.classifier_classes[current_class]][document]
                if current_score > last_score:
                    last_class = self.classifier_classes[current_class]
            result.append(last_class)

        return result

def evaluate(predictions, documents):
    """ Evaluate the accuracy of a set of predictions.
    Return a tuple of three values (X, Y, Z) where
    X = percent of documents classified correctly
    Y = number of ham documents incorrectly classified as spam
    X = number of spam documents incorrectly classified as ham

    Params:
      predictions....list of document labels predicted by a classifier.
      documents......list of Document objects, with known labels.
    Returns:
      Tuple of three floats, defined above.
    """
    false_spam = 0
    missed_spam = 0

    for document in range(0, len(documents)):
        if documents[document].label is not 'spam' and predictions[document] is 'spam':
            false_spam += 1
        elif documents[document].label is 'spam' and predictions[document] is not 'spam':
            missed_spam += 1

    accuracy = float((len(documents) - (false_spam + missed_spam))) / float(len(documents))

    return (accuracy, false_spam, missed_spam)

def main():
    """ Do not modify. """
    if not os.path.exists('train'):  # download data
       from urllib.request import urlretrieve
       import tarfile
       urlretrieve('http://cs.iit.edu/~culotta/cs429/lingspam.tgz', 'lingspam.tgz')
       tar = tarfile.open('lingspam.tgz')
       tar.extractall()
       tar.close()
    train_docs = [Document(filename=f) for f in glob.glob("train/*.txt")]
    print('read', len(train_docs), 'training documents.')
    nb = NaiveBayes()
    nb.train(train_docs)
    test_docs = [Document(filename=f) for f in glob.glob("test/*.txt")]
    print('read', len(test_docs), 'testing documents.')
    predictions = nb.classify(test_docs)
    results = evaluate(predictions, test_docs)
    print('accuracy=%.3f, %d false spam, %d missed spam' % (results[0], results[1], results[2]))
    print('top ham terms: %s' % ' '.join('%.2f/%s' % (v,t) for v, t in nb.get_top_words('ham', 10)))
    print('top spam terms: %s' % ' '.join('%.2f/%s' % (v,t) for v, t in nb.get_top_words('spam', 10)))

if __name__ == '__main__':
    main()
