
# -*- coding: utf-8 -*-
"""
Various utility functions
"""
import numpy as np
from collections import OrderedDict
from pprint import pprint


def match_score(pi, pj, prefix_weight, middle_weight, suffix_weight):
    """ Compute match between phrases using a dot product of vectors
    :param pi Phrase or pattern
    :param pj phrase or pattern
    # add weights to dot products to put more emphasis on matching the middles
    """
    assert(pi.keys() == pj.keys())

    number_of_middles = len([i for i in pi.keys() if i.startswith('middle')])

    prefix_i = pi['prefix']

    middles_i = []
    for m in range(0, number_of_middles):
        middles_i.append(pi['middle_' + str(m+1)])
    # print(middles_i)

    suffix_i = pi['suffix']

    prefix_j = pj['prefix']

    middles_j = []
    for m in range(0, number_of_middles):
        middles_j.append(pj['middle_' + str(m+1)])

    suffix_j = pj['suffix']

    if len(prefix_i) == 0 and len(prefix_j) == 0:
        prefix_dot_prod = 1.0
    else:
        prefix_dot_prod = np.dot(prefix_i, prefix_j)
    # print(prefix_dot_prod)

    middle_dot_prod = []
    for i in range(0, len(middles_i)):
        if len(middles_i[i]) == 0 and len(middles_j[i]) == 0:
            middle_dot_prod.append(1.0)
        elif len(middles_i[i]) != len(middles_j[i]):
            middle_dot_prod.append(0)
        else:
            m_dot = np.dot(middles_i[i], middles_j[i])
            middle_dot_prod.append(m_dot)
    # print(middle_dot_prod)

    if len(suffix_i) == 0 and len(suffix_j) == 0:
        suffix_dot_prod = 1.0
    else:
        suffix_dot_prod = np.dot(suffix_i, suffix_j)
    # print(suffix_dot_prod)

    sim = (prefix_weight * prefix_dot_prod) + ((middle_weight/number_of_middles)*sum(middle_dot_prod)) + (suffix_weight * suffix_dot_prod)
    # print(sim, "\n")

    return sim


def vectorise(phrase, cluster):
    """Vectorise a phrase object against a given cluster
    
    Arguments:
        phrase {[type]} -- [description]
        cluster {[type]} -- [description]
    """
    # print("Vectorising phrase ", phrase)
    # print("Pattern", cluster.pattern)
    phrase_element_vectors = {}
    pattern_element_vectors = {}
    pattern = cluster.pattern
    

    for element in cluster.dictionaries.keys():  # prefix, middles, suffix
        # print("element", element, '\n')
        local_dictionary = OrderedDict()

        '''
        # fill the local dictionary with the cluster tokens
        for token in cluster.dictionaries[element]['token dict']:
            if token in local_dictionary.keys():
                local_dictionary[token] += 1
            else:
                local_dictionary[token] = 1 # cluster.dictionaries[element]['token dict'][token][0]
        
        # Same for the phrase tokens
        for token in phrase.elements[element]['tokens']:
            if token in local_dictionary.keys():
                local_dictionary[token] += 1.0
            else:
                local_dictionary[token] = 1.0
        '''

        # fill the local dictionary with the pattern and phrase tokens
        for i in [pattern, phrase]:
            for token in i.elements[element]['tokens']:
                if token in local_dictionary.keys():
                    local_dictionary[token] += 1.0
                else:
                    local_dictionary[token] = 1.0
        
        # pprint(local_dictionary)
        # print(element, '\n\n')
        
        phrase_element_vector = np.zeros(len(local_dictionary.keys()))
        pattern_element_vector = np.zeros(len(local_dictionary.keys()))
        # Now vectorise the phrase and pattern
        for i, token in enumerate(local_dictionary.keys()):
            if token in phrase.elements[element]['tokens']:
                phrase_element_vector[i] += 1
            if token in pattern.elements[element]['tokens']:
                pattern_element_vector[i] += 1

        # normalise the vectors
        phrase_element_vectors[element] = phrase_element_vector / np.linalg.norm(phrase_element_vector)
        pattern_element_vectors[element] = pattern_element_vector / np.linalg.norm(pattern_element_vector)
    
    # pprint(phrase_element_vectors)
    # print('phrase_element_vectors\n')
    # pprint(pattern_element_vectors)
    # print('pattern_element_vectors\n')

    return phrase_element_vectors, pattern_element_vectors


def match(phrase, cluster, prefix_weight, middles_weight, suffix_weight):
    """Vectorise the phrase against this cluster to determine the match score
    
    Arguments:
        phrase {[type]} -- [description]
        cluster {[type]} -- [description]
    """
    if phrase.order != cluster.order:
        return 0

    phrase_vectors, pattern_vectors = vectorise(phrase, cluster)
    score = match_score(phrase_vectors, pattern_vectors, prefix_weight, middles_weight, suffix_weight)
    # print(phrase, '\n\n', cluster.pattern, '\n')
    # print("similarity = ", score)
    return score


def mode_rows(a):
    """
    Find the modal row of a 2d array
    :param a: The 2d array to process
    :type a: np.array()
    :return: The most frequent row
    """
    a = np.ascontiguousarray(a)
    void_dt = np.dtype((np.void, a.dtype.itemsize * np.prod(a.shape[1:])))
    _, ids, count = np.unique(a.view(void_dt).ravel(),
                              return_index=True,
                              return_counts=True)
    largest_count_id = ids[count.argmax()]
    most_frequent_row = a[largest_count_id]
    return most_frequent_row


def KnuthMorrisPratt(text, pattern):
    """Yields all starting positions of copies of the pattern in the text.
        Calling conventions are similar to string.find, but its arguments can be
        lists or iterators, not just strings, it returns all matches, not just
        the first one, and it does not need the whole text in memory at once.
        Whenever it yields, it will have read the text exactly up to and including
        the match that caused the yield.

    Source: http://code.activestate.com/recipes/117214/"""

    # allow indexing into pattern and protect against change during yield
    pattern = list(pattern)

    # build table of shift amounts
    shifts = [1] * (len(pattern) + 1)
    shift = 1
    for pos in range(len(pattern)):
        while shift <= pos and pattern[pos] != pattern[pos-shift]:
            shift += shifts[pos-shift]
        shifts[pos+1] = shift

    # do the actual search
    startPos = 0
    matchLen = 0
    for c in text:
        while matchLen == len(pattern) or \
              matchLen >= 0 and pattern[matchLen] != c:
            startPos += shifts[matchLen]
            matchLen -= shifts[matchLen]
        matchLen += 1
        if matchLen == len(pattern):
            yield startPos


def subfinder(mylist, pattern):
    for i in range(len(mylist)):
        if mylist[i] == pattern[0] and mylist[i:i+len(pattern)] == pattern:
            return (i, i+len(pattern))
    return None, None


def find_start(tokens, pattern):

    '''
    a replacement for KnuthMorrisPratt, buggy
    '''

    start_pos = []
    pos = -1
    while True:
        try:
            pos = tokens.index(pattern[0], pos+1)
            if len(pattern) == 1:
                start_pos.append(pos)
            else:
                for i in range(1, len(pattern)):
                    try:
                        if tokens[pos+i] == pattern[i]:
                            start_pos.append(pos)
                    except IndexError: # text may not have enough length to compare with all patterns
                        pass
        except ValueError:
            # print(pattern[0], tokens[pos+1:])
            break
    
    return set(start_pos)