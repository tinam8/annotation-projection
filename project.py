import sys
import numpy as np
import time
import pickle
from numba import jit


class TokenConll():
    def __init__(self, fields):
        self.id = fields[0]
        self.form = fields[1]
        self.pos = fields[2]
        self.head = fields[3]


def get_golden_tree(tokens):
    tree = [-1]
    for token in tokens:
        tree.append(int(token.head))

    return tree


class TargetSentence():
    def __init__(self, tokens):
        self.tokens = tokens
        # self.golden_tree = get_golden_tree(tokens)
        self.source = None
        shape = (len(tokens)+1, len(tokens)+1)
        self.T = np.zeros(shape) * np.nan
        self.pos_tags = [[] for i in range(len(tokens))]
        self.accuracy = None


    def get_tree_score(self):
        match_num = -1
        # for node1, node2 in zip(self.T, self.golden_tree):
        for node1, node2 in zip(self.T, get_golden_tree(self.tokens)):
            # nan
            if node1 == node2:
                match_num += 1
        # return match_num / (len(self.golden_tree) - 1)
        return match_num / (len(get_golden_tree(self.tokens)) - 1)


    def get_pos_score(self):
        match_num = 0
        for i in range(len(self.pos_tags)):
            if self.pos_tags[i] == self.tokens[i].pos:
                match_num += 1
        return match_num / (len(self.tokens))


    def get_conll_lines(self, conll_type = '2009'):
        lines = ''
        for i in range(len(self.tokens)):
            if conll_type == '2009':
                lines += '{id}\t{form}\t_\t{pos}\t_\t_\t{head}\t_\t_\t_\n'.format(id=self.tokens[i].id,
                                                                              form=self.tokens[i].form,
                                                                              pos=self.pos_tags[i],
                                                                              head=self.target[i + 1])
        lines += '\n'
        return lines

def get_matrix(sentence):
    """
    Creates matric from list of Tokens
    :param sentence: list of tokens
    :return: matrix that represents sentence
    """
    s = (len(sentence) + 1, len(sentence) + 1)
    matrix = np.zeros(s) * np.nan
    for token in sentence:
        matrix[int(token.id), int(token.head)] = 1
    return matrix


def get_pos_tags(tokens):
    pos_tags = []
    for token in tokens:
        pos_tags.append(token.pos)
    return pos_tags


class SourceSentence():
    def __init__(self, tokens):
        self.matrix = get_matrix(tokens)
        self.pos_tags = get_pos_tags(tokens)


def read_language_from_conll(lines, source = False):
    """
    Reads all sentences form conll file
    :param lines: lines from conll file
    :return: list of instances of TargetSentence
    """
    print("Getting language sentences")
    sentences = []
    sentence = []

    for line in lines:
        if line.startswith('\n'):
            if len(sentence) != 0:
                if not source:
                    sentences.append(TargetSentence(sentence))
                else:
                    sentences.append(SourceSentence(sentence))
                sentence = []
            continue

        parts = line.split('\t')
        fields = [parts[0], parts[1], parts[3], parts[6]]
        sentence.append(TokenConll(fields))

    return sentences


def get_pair_index(word_alignments_, index_range):
    """
    Gets positions of source sentences that are paired with target sentences
    :param f: file path to the source target sentence pairs
    :param index_range: number od target sentences
    :return:
        - indexes - list of aligned sources sentences
            eg. list index represent position of target sentence and value represents position
            of aligned source sentence, if value is np.nan target sentence does not
            have corresponding source sentence
        - aligned_num - number of corresponding sentence pairs
    """
    print('Getting aligned pairs')
    indexes = np.zeros(index_range) * np.nan
    aligned_num = 0

    for line in word_alignments_:
        parts = line.split("\t")
        if len(parts) != 3:
            assert ValueError('invalid line in file: ' + line)
        aligned_num += 1
        indexes[int(parts[1])] = int(parts[0])
    print('number of aligned: ' + str(aligned_num))

    return indexes, aligned_num


def get_sentence_alignment_prob(line, m_plus_one, n_plus_one):
    """
    Creates alignement matrix from string of alignments
    :param line: string that represents alignments
    :param m_plus_one: m is length of source sentence
    :param n_plus_one: n is length of target sentence
    :return: matrix A with dimensions m+1xn+1, first row and column are np.nan except for A[0][0] = 1
    """
    pairs = line.split()
    s = (m_plus_one, n_plus_one)
    A = np.ones(s) * np.nan
    # TODO: = random?
    A[0, 0] = 1

    for i in range(0, len(pairs), 2):
        pair = pairs[i].split('-')
        prob = pairs[i + 1]
        A[int(pair[0]) + 1, int(pair[1]) + 1] = float(prob)

    return A


def get_target_from_alignment(S, A):
    m_plus_one = A.shape[0]
    n_plus_one = A.shape[1]

    assert S.shape[0] == m_plus_one

    T = np.empty((n_plus_one, n_plus_one,)) * np.nan
    T_edge = np.empty_like(T) * np.nan

    for d in range(m_plus_one):
        for h in range(m_plus_one):
            np.dot(A[d].reshape(-1, 1), A[h].reshape(1, -1), out=T_edge)
            T_edge *= S[d, h]
            np.fmax(T, T_edge, out=T)

    return T


def get_coverage(A):
    n_plus_one = A.shape[1]
    aligned_count = 0

    for column in np.transpose(A[:, 1:]):
        alignments = np.nonzero(np.isnan(column) == False)[0]
        if len(alignments) != 0:
            aligned_count += 1

    return aligned_count/(n_plus_one - 1)


def get_pos_from_alignment(source, A):
    n_plus_one = A.shape[1]
    pos_tags = source.pos_tags
    pos_projections = [[] for i in range(n_plus_one - 1)]

    for col in range(1, A.shape[1]):
        projections = np.nonzero(np.isnan(A[:, col]) == False)[0]
        for projection in projections:
            pos_projections[col - 1].append(pos_tags[projection - 1])

    return pos_projections


def project(target, source, word_alignment, pair_index, name):
    count = 0

    for i in range(len(target)):
        if i % 5000 == 0:
            print(i)

        if not np.isnan(pair_index[i]):
            S = source[int(pair_index[i])].matrix
            n_plus_one = len(target[i].tokens) + 1

            # projecting edges
            A = get_sentence_alignment_prob(word_alignment[count], S.shape[0], n_plus_one)
            T = get_target_from_alignment(S, A)
            target[i].T = T
            target[i].coverage = {name: get_coverage(A)}

            # projecting tags
            target[i].pos_tags = get_pos_from_alignment(source[int(pair_index[i])], A)
            count += 1

        else:
            target[i].coverage = {name: 0}


def get_accuracy(target_sentences, pos=False):
    acc = 0

    for target_sentence in target_sentences:
        if not pos:
            acc += target_sentence.get_tree_score()
        else:
            acc += target_sentence.get_pos_score()

    return acc / len(target_sentences)


def count_sources(sc_coverage, coverage):
    count = 0

    for key, value in sc_coverage.items():
        if value > coverage:
            count += 1
    return count


def rank(target, coverage, len_cutoff=0):
    num_of_sources = len(target[0].coverage)
    mapping = {}
    lens = {}

    for i in range(len(target[0].coverage) + 1):
        mapping[i] = []

    for tg_sentence in target:
        if len(tg_sentence.tokens) > len_cutoff:
            mapping[(count_sources(tg_sentence.coverage, coverage))].append(tg_sentence)

    targets = []
    for i in range(num_of_sources, -1, -1):
        targets.extend(mapping[i])
        lens[i] = len(mapping[i])

    return targets, lens



if __name__ == '__main__':
    """
    python project.py
        --source bg.conll
        --target ar.conll
        --word_alignment bg-ar.watchtower.ibm1.reverse.wal
        --sentence_alignment bg-ar.watchtower.sal
        1> bg-ar.proj
    """
    source_ = open(sys.argv[1]).readlines()
    target_ = open(sys.argv[2]).readlines()
    word_alignment_ = open(sys.argv[3]).readlines()
    sentence_alignment_ = open(sys.argv[4]).readlines()

    target = read_language_from_conll(target_, source=False)
    source = read_language_from_conll(source_, source=True)

    pair_index, num_algn = get_pair_index(sentence_alignment_, len(target))
    print(len(pair_index))

    start = time.time()
    project(target, source, word_alignment_, pair_index, sys.argv[1])
    print((time.time()-start)/60)

    dot_position = sys.argv[3].find('.')
    pickle_name = sys.argv[3][dot_position-5:dot_position] + '.proj'
    pickle.dump(target, open('proj/' + pickle_name, 'wb'))
