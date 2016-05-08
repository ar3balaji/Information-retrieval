from collections import defaultdict
import tarfile
import tabulate

import index, evaluate, score

"""
assignment 2 by:
Balaji A R
CWID A20347964
"""

def parse_relevance_strings(strings):
    r"""
    Parse lines from TIME.REL.

    Params:
      strings...A list of strings, one per line, from TIME.REL
    Returns:
      A dict from query id to the list of relevant document ids, as ordered in the file.

    >>> strings = '''1  268 288 304
    ...
    ... 2  326 334'''
    >>> result = parse_relevance_strings(strings.split('\n'))
    >>> sorted(result.items())
    [(1, [268, 288, 304]), (2, [326, 334])]
    """
    query_docids={}

    for index in range(0,len(strings)):
        temp_string = strings[index].replace('\n','')
        temp_string_list = (temp_string.strip()).split()
        temp_string_list = [ int(i) for i in temp_string_list]
        if len(temp_string_list) > 0 :
            query_docids[int(temp_string_list[0])] = temp_string_list[1:len(temp_string_list)+1]

    return query_docids

def read_relevances(fname):
    """
    Do not modify.
    Read a map from query ID to a list of relevant doc IDs.
    """
    return parse_relevance_strings(open(fname).readlines())


def parse_query_strings(strings):
    r"""
    Parse lines from TIME.QUE.

    Params:
      strings...A list of strings, one per line, from TIME.QUE
    Returns:
      A dict from query id to query text string.
    >>> string = '''*FIND      1
    ...
    ... KENNEDY ADMINISTRATION PRESSURE ON NGO DINH DIEM TO STOP
    ...
    ... SUPPRESSING THE BUDDHISTS .
    ...
    ... *FIND      2
    ...
    ... EFFORTS OF AMBASSADOR HENRY CABOT LODGE TO GET VIET NAM'S
    ...
    ... PRESIDENT DIEM TO CHANGE HIS POLICIES OF POLITICAL REPRESSION .
    ...
    ... *STOP'''
    >>> res = parse_query_strings(string.split('\n'))
    >>> print('%s' % res[1])
    KENNEDY ADMINISTRATION PRESSURE ON NGO DINH DIEM TO STOP SUPPRESSING THE BUDDHISTS .
    >>> print('%s' % res[2])
    EFFORTS OF AMBASSADOR HENRY CABOT LODGE TO GET VIET NAM'S PRESIDENT DIEM TO CHANGE HIS POLICIES OF POLITICAL REPRESSION .
    """
    queryList = []
    i = 0
    while i < len(strings):
        if "*FIND" in strings[i]:
            j = i + 1
            tempString = ""
            while j < len(strings):
                if "*FIND" in strings[j] or "*STOP" in strings[j]:
                    i = j
                    break
                else:
                    tempString = tempString.strip() + " " + remove_spaces_eof(strings[j])
                    j += 1
            if tempString is not "":
                queryList.append(tempString.strip())
        else:
            i += 1
    return { k+1: v for k, v in enumerate(queryList)}


def read_queries(fname):
    """  Do not modify. Read a map from query id to text."""
    return parse_query_strings(open(fname).readlines())


def parse_document_strings(strings):
    r"""
    Parse lines from TIME.ALL.

    Params:
       strings...A list of strings, one per line, from TIME.ALL
    Returns:
       A list of strings, one per document.
    >>> string = '''*TEXT 017 01/04/63 PAGE 020
    ...
    ... THE ALLIES AFTER NASSAU
    ...
    ... *TEXT 020 01/04/63 PAGE 021
    ...
    ... THE ROAD TO JAIL IS PAVED WITH
    ...
    ... *STOP'''
    >>> parse_document_strings(string.split('\n'))
    ['THE ALLIES AFTER NASSAU', 'THE ROAD TO JAIL IS PAVED WITH']
    """
    documentList = []
    i = 0
    while i < len(strings):
        if "*TEXT" in strings[i]:
            j = i + 1
            tempString = ""
            while j < len(strings):
                if "*TEXT" in strings[j] or "*STOP" in strings[j]:
                    i = j
                    break
                else:
                    tempString = tempString.strip() + " " + remove_spaces_eof(strings[j])
                    j += 1
            if tempString is not "":
                documentList.append(tempString.strip())
        else:
            i += 1
    return documentList

def remove_spaces_eof(str):
    str = str.replace("\n","")
    return str.strip()

def read_documents(fname):
    """ Do not modify. Read a list of documents."""
    return parse_document_strings(open(fname).readlines())


def read_data():
    """ Do not modify. Read data."""
    tarfile.open('time.tar.gz', mode='r').extractall()
    queries = read_queries('TIME.QUE')
    print('read %d queries.' % len(queries))
    relevances = read_relevances('TIME.REL')
    print('read %d relevance judgements.' % len(relevances))
    docs = read_documents('TIME.ALL')
    print('read %d documents.' % len(docs))
    return queries, relevances, docs

def write_results(all_results, fname):
    """ Do not modify. Write results. """
    evals = sorted(list(all_results.values())[0].keys())
    headers = ['System'] + evals
    systems = sorted(all_results.keys())
    vals = []
    for system in systems:
        results = all_results[system]
        vals.append([system] + [results[e] for e in evals])
    f = open(fname, 'w')
    f.write(tabulate.tabulate(vals, headers, floatfmt=".4f"))
    f.write('\n')

def search(query, scorer, index):
    """
    Retrieve documents matching a query using the specified scorer.

    1) Tokenize the query.
    2) Convert the query tokens to a vector, using Index.query_to_vector.
    3) Call the scorer's score function.
    4) Return the list of document ids in descending order of relevance.

    NB: Due to the inconsistency of floating point arithmetic, when sorting,
    round the scores to 6 decimal places (e.g., round(x, 6)). This will ensure
    replicable results.

    Params:
      query....A string representing a search query.
      scorer...A ScoringFunction to retrieve documents.
      index....A Index storing postings lists.
    Returns:
      A list of document ids in descending order of relevance to the query.
    """
    tempResult = scorer.score(index.query_to_vector(index.tokenize(query)),index)
    tempResult = sorted(tempResult.items(), key = lambda key: round(key[1],6), reverse = True)
    result = []
    for i in range(0,len(tempResult)):
        result.append(tempResult[i][0])
    return result




def run_all(queries, relevances, docs, indexer, scorers, evaluators, NHITS):
    """ Do not modify.
    For each query, run all scoring methods and evaluate the results.
    Returns:
      A dict from scoring method to results
    """
    results = defaultdict(lambda: defaultdict(lambda: 0))
    for qid, qtext in queries.items():
        print('\n-------\nQUERY:%d %s' % (qid, qtext))
        relevant = sorted(list(relevances[qid]))
        print('RELEVANT: %s' % relevant)
        for scorer in scorers:
            hits = search(qtext, scorer, indexer)[:NHITS]
            print('\t%s results: %s' % (scorer, hits))
            for evaluator in evaluators:
                evaluation = evaluator.evaluate(hits, relevant)
                results[str(scorer)][str(evaluator)] += evaluation
    for scorer in scorers:
        for evaluator in evaluators:
            results[str(scorer)][str(evaluator)] /= len(queries)
    return results


def main():
    """ Do not modify.
    Run and evaluate all methods.
    """
    queries, relevances, docs = read_data()

    NHITS = 10
    indexer = index.Index(docs)
    scorers = [score.Cosine(),
               score.RSV(),
               score.BM25(k=1, b=.5),
               score.BM25(k=1, b=1),
               score.BM25(k=2, b=.5),
               score.BM25(k=2, b=1)]

    evaluators = [evaluate.Precision(),
                  evaluate.Recall(),
                  evaluate.F1(),
                  evaluate.MAP()]

    all_results = run_all(queries, relevances, docs, indexer, scorers, evaluators, NHITS)
    write_results(all_results, 'Results.md')

if __name__ == '__main__':
    main()
