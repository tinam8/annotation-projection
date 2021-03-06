import sys
import numpy as np
from dependency_decoding import chu_liu_edmonds
import time
from collections import Counter
import pickle
import glob
import sys

def softmax_wall(sources_distribution, temperature=1.0):
    sources = [s for s, _ in sources_distribution.items()]
    weights = np.array([w for _, w in sources_distribution.items()])
    e = np.exp(weights / temperature)
    softmaxed = e / np.sum(e)
    sources_distribution_softmaxed = dict(zip(sources, softmaxed))
    return sources_distribution_softmaxed


def invert(sources_distribution):
    sources = [s for s, _ in sources_distribution.items()]
    weights = [1.0 / w for _, w in sources_distribution.items()]  # FIXME 1-x or 1/x?
    return dict(zip(sources, weights))


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


def read_from_pickle(path):
    print('Reading ' + path)
    with open(path, 'rb') as file:
        try:
            while True:
                # yield pickle.load(file)
                return pickle.load(file)
        except EOFError:
            pass


def add_projection(final_targets, projection_file, source_index, wals=False):
    source_lng = projection_file[5:7]
    projection = read_from_pickle(projection_file)

    for i in range(len(final_targets)):
        target = final_targets[i]

        wals_value = wals_map[source_lng]

        # target.T[:, :, source_index] = softmax(projection[i].T) if not wals else softmax(projection[i].T) * wals_map[list(projection[i].T.coverage.keys())[0]]
        target.T[:, :, source_index] = projection[i].T if wals else wals_value * projection[i].T

        for key, value in projection[i].coverage.items():
            target.coverage[key] = value

        if projection[i].pos_tags is not None:
            for j in range(len(target.pos_tags)):
                target.pos_tags[j].extend(projection[i].pos_tags[j])


def eliminate_all_nan_rows(M_proj):
    by_row = np.max(M_proj, axis=1)
    all_zero_rows = np.where(by_row == 0)
    M_proj[all_zero_rows] = np.min(M_proj)
    return M_proj


def calculate(targets):
    for target in targets:
        target.T = softmax(np.nansum(target.T, axis=2))
        # target.T = eliminate_all_nan_rows(target.T)

        target.T = chu_liu_edmonds(target.T)[0]

        pos_tags = []
        for pos_projection in target.pos_tags:
            # TODO: ako ima istih
            if len(pos_projection) != 0:
                most_common = Counter(pos_projection).most_common(1)[0][0]
                if most_common == '_' and len(Counter(pos_projection)) != 1:
                    most_common = Counter(pos_projection).most_common(1)[1][0]
            else:
                most_common = '_'
            pos_tags.append(most_common)

        target.pos_tags = pos_tags


def softmax(sentence_matrix, temperature=1.0):
    """Softmax normalization.

    :param sentence_matrix: (n+1 x n+1) weight matrix from the parser
    :param temperature: softmax temperature
    :return: softmaxed weight matrix
    """
    m_exp = np.exp(sentence_matrix/temperature)
    return (m_exp.T / np.nansum(m_exp, axis=1)).T


def vote(projections, wals=False):
    num_sources = len(projections)
    print('Reading ' + projections[0])
    target = pickle.load(open(projections[0], 'rb'))
    source_lng = projections[0][5:7]
    wals_value = wals_map[source_lng]

    final_targets = []
    for tar in target:
        n_plus_one = len(tar.tokens) + 1
        shape = (n_plus_one, n_plus_one, num_sources)
        tensor = np.zeros(shape) * np.nan
        # tensor[:, :, 0] = softmax(tar.T) if not wals else softmax(tar.T) * wals_map[list(tar.coverage.keys())[0]]
        tensor[:, :, 0] = tar.T if not wals else wals_value * tar.T

        final = TargetSentence(tar.tokens)
        final.pos_tags = tar.pos_tags
        final.coverage = tar.coverage
        final.T = tensor

        final_targets.append(final)

    print(len(final_targets))

    for i in range(1, len(projections)):
        add_projection(final_targets, projections[i], i, wals)

    calculate(final_targets)
    return final_targets


if __name__ == '__main__':
    """
    python vote.py
        # --target ar.conll
        --projections /home/zagic/nlpfromscratch/data/projections/*-ar.proj
    """
    wals = {'ar': {'fa': 0.66666666666666663, 'ro': 0.5, 'fi': 0.58333333333333337, 'et': 0.58823529411764708,
                   'pl': 0.48484848484848486, 'da': 0.61764705882352944, 'ta': 0.78787878787878785,
                   'hi': 0.55882352941176472, 'hr': 0.5757575757575758, 'it': 0.52941176470588236,
                   'no': 0.63636363636363635, 'he': 0.40000000000000002, 'sl': 0.51724137931034486,
                   'es': 0.52777777777777779, 'de': 0.7142857142857143, 'el': 0.48717948717948717,
                   'nl': 0.74285714285714288, 'pt': 0.4838709677419355, 'bg': 0.5714285714285714,
                   'en': 0.61111111111111116, 'hu': 0.58823529411764708, 'id': 0.54285714285714282, 'cs': 0.6875,
                   'sv': 0.6470588235294118, 'fr': 0.55555555555555558},
            'bg': {'ar': 0.5714285714285714, 'fi': 0.40425531914893614, 'el': 0.25, 'en': 0.36170212765957449,
                   'ta': 0.68421052631578949, 'pt': 0.35714285714285715, 'sv': 0.46575342465753422,
                   'id': 0.5053763440860215, 'fr': 0.45161290322580644, 'sl': 0.28260869565217389,
                   'fa': 0.4942528735632184, 'cs': 0.46551724137931033, 'ro': 0.30379746835443039,
                   'es': 0.36559139784946237, 'it': 0.33333333333333331, 'no': 0.44776119402985076,
                   'de': 0.53333333333333333, 'he': 0.48749999999999999, 'hi': 0.49397590361445781,
                   'pl': 0.29113924050632911, 'hr': 0.33333333333333331, 'nl': 0.57971014492753625,
                   'hu': 0.43956043956043955, 'et': 0.375, 'da': 0.47272727272727272},
            'cs': {'it': 0.55932203389830504, 'da': 0.50877192982456143, 'es': 0.51470588235294112,
                   'el': 0.48529411764705882, 'hi': 0.59999999999999998, 'id': 0.61194029850746268,
                   'hu': 0.56716417910447758, 'no': 0.50877192982456143, 'pt': 0.57692307692307687,
                   'sv': 0.47692307692307695, 'pl': 0.375, 'ar': 0.6875, 'he': 0.54411764705882348,
                   'en': 0.44117647058823528, 'de': 0.578125, 'hr': 0.38983050847457629, 'bg': 0.46551724137931033,
                   'sl': 0.375, 'ta': 0.60416666666666663, 'nl': 0.58461538461538465, 'fi': 0.5,
                   'ro': 0.55932203389830504, 'fr': 0.59090909090909094, 'fa': 0.4576271186440678,
                   'et': 0.49122807017543857},
            'da': {'hr': 0.47272727272727272, 'el': 0.44776119402985076, 'id': 0.59090909090909094,
                   'fr': 0.39393939393939392, 'ar': 0.61764705882352944, 'fa': 0.67241379310344829,
                   'pl': 0.43548387096774194, 'it': 0.44444444444444442, 'ro': 0.44262295081967212,
                   'no': 0.16129032258064516, 'pt': 0.47457627118644069, 'bg': 0.47272727272727272,
                   'es': 0.40909090909090912, 'et': 0.47368421052631576, 'fi': 0.46268656716417911,
                   'nl': 0.36507936507936506, 'ta': 0.69387755102040816, 'sv': 0.15384615384615385,
                   'he': 0.56060606060606055, 'sl': 0.4375, 'en': 0.28358208955223879, 'hi': 0.5892857142857143,
                   'de': 0.36764705882352944, 'hu': 0.55384615384615388, 'cs': 0.50877192982456143},
            'de': {'sl': 0.56603773584905659, 'pt': 0.49275362318840582, 'ar': 0.7142857142857143,
                   'fr': 0.33121019108280253, 'fi': 0.45161290322580644, 'hi': 0.48951048951048953,
                   'fa': 0.54054054054054057, 'cs': 0.578125, 'nl': 0.17391304347826086, 'ta': 0.6785714285714286,
                   'sv': 0.38554216867469882, 'et': 0.63636363636363635, 'it': 0.43333333333333335,
                   'ro': 0.47126436781609193, 'en': 0.32704402515723269, 'no': 0.35064935064935066,
                   'el': 0.40259740259740262, 'pl': 0.45744680851063829, 'es': 0.43312101910828027,
                   'he': 0.52482269503546097, 'id': 0.61038961038961037, 'bg': 0.53333333333333333,
                   'da': 0.36764705882352944, 'hr': 0.58461538461538465, 'hu': 0.49019607843137253},
            'el': {'id': 0.57961783439490444, 'fa': 0.42857142857142855, 'hi': 0.46853146853146854,
                   'hr': 0.37681159420289856, 'pl': 0.30303030303030304, 'ta': 0.6705882352941176,
                   'et': 0.40579710144927539, 'sv': 0.47126436781609193, 'ar': 0.48717948717948717,
                   'fi': 0.44025157232704404, 'ro': 0.31521739130434784, 'he': 0.45205479452054792,
                   'sl': 0.39622641509433965, 'no': 0.45454545454545453, 'nl': 0.50549450549450547,
                   'fr': 0.39743589743589741, 'it': 0.36263736263736263, 'en': 0.41509433962264153,
                   'pt': 0.3188405797101449, 'da': 0.44776119402985076, 'es': 0.29746835443037972,
                   'de': 0.40259740259740262, 'cs': 0.48529411764705882, 'hu': 0.41772151898734178, 'bg': 0.25},
            'en': {'fr': 0.37951807228915663, 'fa': 0.55263157894736847, 'pt': 0.36231884057971014,
                   'pl': 0.37373737373737376, 'sv': 0.25287356321839083, 'hu': 0.4567901234567901,
                   'ta': 0.67816091954022983, 'hi': 0.4391891891891892, 'ro': 0.34782608695652173,
                   'no': 0.31578947368421051, 'nl': 0.34065934065934067, 'cs': 0.44117647058823528,
                   'fi': 0.39393939393939392, 'he': 0.42384105960264901, 'bg': 0.36170212765957449,
                   'hr': 0.44285714285714284, 'es': 0.36969696969696969, 'id': 0.50306748466257667,
                   'it': 0.31868131868131866, 'sl': 0.33962264150943394, 'et': 0.41428571428571431,
                   'ar': 0.61111111111111116, 'el': 0.41509433962264153, 'da': 0.28358208955223879,
                   'de': 0.32704402515723269},
            'es': {'fi': 0.44099378881987578, 'de': 0.43312101910828027, 'el': 0.29746835443037972,
                   'pt': 0.23529411764705882, 'no': 0.46666666666666667, 'en': 0.36969696969696969,
                   'da': 0.40909090909090912, 'sl': 0.39622641509433965, 'ro': 0.31868131868131866,
                   'hu': 0.48734177215189872, 'et': 0.44927536231884058, 'sv': 0.40697674418604651,
                   'hr': 0.47142857142857142, 'nl': 0.44444444444444442, 'he': 0.44217687074829931,
                   'hi': 0.47222222222222221, 'it': 0.20000000000000001, 'fa': 0.51351351351351349,
                   'fr': 0.30246913580246915, 'ar': 0.52777777777777779, 'id': 0.55345911949685533,
                   'pl': 0.39795918367346939, 'bg': 0.36559139784946237, 'cs': 0.51470588235294112,
                   'ta': 0.74117647058823533},
            'et': {'hi': 0.50847457627118642, 'bg': 0.375, 'el': 0.40579710144927539, 'en': 0.41428571428571431,
                   'nl': 0.63636363636363635, 'fa': 0.65573770491803274, 'ro': 0.44444444444444442,
                   'fi': 0.24285714285714285, 'de': 0.63636363636363635, 'no': 0.51724137931034486, 'sv': 0.5,
                   'cs': 0.49122807017543857, 'fr': 0.54285714285714282, 'es': 0.44927536231884058,
                   'pt': 0.45454545454545453, 'it': 0.48333333333333334, 'pl': 0.38461538461538464,
                   'da': 0.47368421052631576, 'id': 0.58571428571428574, 'sl': 0.40425531914893614,
                   'he': 0.46376811594202899, 'hr': 0.42857142857142855, 'ar': 0.58823529411764708,
                   'ta': 0.59999999999999998, 'hu': 0.35294117647058826},
            'fa': {'id': 0.54966887417218546, 'nl': 0.59036144578313254, 'sl': 0.58695652173913049,
                   'no': 0.61194029850746268, 'sv': 0.60256410256410253, 'fr': 0.51006711409395977,
                   'cs': 0.4576271186440678, 'ro': 0.51219512195121952, 'en': 0.55263157894736847,
                   'hi': 0.51006711409395977, 'fi': 0.53289473684210531, 'de': 0.54054054054054057,
                   'et': 0.65573770491803274, 'pt': 0.55932203389830504, 'hr': 0.58064516129032262,
                   'he': 0.48920863309352519, 'pl': 0.550561797752809, 'el': 0.42857142857142855,
                   'it': 0.59999999999999998, 'ar': 0.66666666666666663, 'bg': 0.4942528735632184,
                   'da': 0.67241379310344829, 'ta': 0.62068965517241381, 'es': 0.51351351351351349,
                   'hu': 0.44666666666666666},
            'fi': {'da': 0.46268656716417911, 'it': 0.46153846153846156, 'fa': 0.53289473684210531, 'cs': 0.5,
                   'en': 0.39393939393939392, 'bg': 0.40425531914893614, 'et': 0.24285714285714285,
                   'pl': 0.39393939393939392, 'hr': 0.48571428571428571, 'nl': 0.53846153846153844,
                   'ro': 0.44565217391304346, 'es': 0.44099378881987578, 'pt': 0.47826086956521741,
                   'id': 0.55828220858895705, 'el': 0.44025157232704404, 'no': 0.48684210526315791,
                   'de': 0.45161290322580644, 'fr': 0.46913580246913578, 'sl': 0.45283018867924529,
                   'ta': 0.58620689655172409, 'hi': 0.4391891891891892, 'he': 0.43046357615894038,
                   'sv': 0.45977011494252873, 'ar': 0.58333333333333337, 'hu': 0.35185185185185186},
            'fr': {'el': 0.39743589743589741, 'hr': 0.48529411764705882, 'hu': 0.54088050314465408,
                   'ta': 0.71764705882352942, 'no': 0.38157894736842107, 'nl': 0.4157303370786517,
                   'es': 0.30246913580246915, 'fa': 0.51006711409395977, 'id': 0.59627329192546585,
                   'he': 0.46621621621621623, 'hi': 0.46206896551724136, 'it': 0.26666666666666666,
                   'ar': 0.55555555555555558, 'cs': 0.59090909090909094, 'en': 0.37951807228915663,
                   'sv': 0.41176470588235292, 'bg': 0.45161290322580644, 'de': 0.33121019108280253,
                   'fi': 0.46913580246913578, 'pt': 0.3188405797101449, 'et': 0.54285714285714282,
                   'pl': 0.42857142857142855, 'ro': 0.40659340659340659, 'da': 0.39393939393939392,
                   'sl': 0.47169811320754718},
            'he': {'ro': 0.40000000000000002, 'fa': 0.48920863309352519, 'nl': 0.57777777777777772,
                   'el': 0.45205479452054792, 'es': 0.44217687074829931, 'fi': 0.43046357615894038,
                   'en': 0.42384105960264901, 'fr': 0.46621621621621623, 'bg': 0.48749999999999999,
                   'et': 0.46376811594202899, 'sv': 0.54651162790697672, 'pl': 0.43023255813953487,
                   'hi': 0.48888888888888887, 'da': 0.56060606060606055, 'ta': 0.63953488372093026,
                   'sl': 0.49056603773584906, 'it': 0.49450549450549453, 'ar': 0.40000000000000002,
                   'pt': 0.44927536231884058, 'id': 0.53020134228187921, 'hu': 0.46308724832214765,
                   'cs': 0.54411764705882348, 'no': 0.5714285714285714, 'de': 0.52482269503546097,
                   'hr': 0.5714285714285714},
            'hi': {'pl': 0.40229885057471265, 'el': 0.46853146853146854, 'sv': 0.56164383561643838,
                   'pt': 0.58333333333333337, 'ar': 0.55882352941176472, 'fa': 0.51006711409395977,
                   'fi': 0.4391891891891892, 'et': 0.50847457627118642, 'nl': 0.51219512195121952,
                   'bg': 0.49397590361445781, 'sl': 0.45652173913043476, 'de': 0.48951048951048953,
                   'hr': 0.43333333333333335, 'id': 0.63698630136986301, 'ta': 0.45348837209302323,
                   'it': 0.57333333333333336, 'es': 0.47222222222222221, 'he': 0.48888888888888887,
                   'hu': 0.46575342465753422, 'ro': 0.50617283950617287, 'no': 0.56923076923076921,
                   'en': 0.4391891891891892, 'cs': 0.59999999999999998, 'da': 0.5892857142857143,
                   'fr': 0.46206896551724136},
            'hr': {'da': 0.47272727272727272, 'no': 0.48076923076923078, 'hu': 0.53623188405797106,
                   'pl': 0.34848484848484851, 'fi': 0.48571428571428571, 'ta': 0.64000000000000001,
                   'el': 0.37681159420289856, 'sv': 0.47761194029850745, 'it': 0.50819672131147542,
                   'bg': 0.33333333333333331, 'es': 0.47142857142857142, 'nl': 0.57352941176470584,
                   'en': 0.44285714285714284, 'fr': 0.48529411764705882, 'sl': 0.22448979591836735,
                   'ro': 0.45161290322580644, 'fa': 0.58064516129032262, 'hi': 0.43333333333333335,
                   'id': 0.66666666666666663, 'de': 0.58461538461538465, 'pt': 0.46296296296296297,
                   'et': 0.42857142857142855, 'cs': 0.38983050847457629, 'ar': 0.5757575757575758,
                   'he': 0.5714285714285714},
            'hu': {'de': 0.49019607843137253, 'ro': 0.48351648351648352, 'pl': 0.4845360824742268,
                   'fa': 0.44666666666666666, 'id': 0.53125, 'fi': 0.35185185185185186, 'bg': 0.43956043956043955,
                   'no': 0.54666666666666663, 'es': 0.48734177215189872, 'fr': 0.54088050314465408,
                   'it': 0.55555555555555558, 'et': 0.35294117647058826, 'ar': 0.58823529411764708,
                   'el': 0.41772151898734178, 'cs': 0.56716417910447758, 'sv': 0.57647058823529407,
                   'hr': 0.53623188405797106, 'sl': 0.50943396226415094, 'pt': 0.55882352941176472,
                   'en': 0.4567901234567901, 'ta': 0.55294117647058827, 'nl': 0.58888888888888891,
                   'he': 0.46308724832214765, 'hi': 0.46575342465753422, 'da': 0.55384615384615388},
            'id': {'en': 0.50306748466257667, 'da': 0.59090909090909094, 'hi': 0.63698630136986301,
                   'sl': 0.54716981132075471, 'es': 0.55345911949685533, 'pl': 0.47474747474747475,
                   'it': 0.51111111111111107, 'fr': 0.59627329192546585, 'ar': 0.54285714285714282,
                   'ro': 0.51086956521739135, 'pt': 0.55072463768115942, 'no': 0.53947368421052633,
                   'sv': 0.62790697674418605, 'fi': 0.55828220858895705, 'bg': 0.5053763440860215,
                   'hr': 0.66666666666666663, 'ta': 0.79069767441860461, 'hu': 0.53125, 'fa': 0.54966887417218546,
                   'de': 0.61038961038961037, 'cs': 0.61194029850746268, 'el': 0.57961783439490444,
                   'he': 0.53020134228187921, 'et': 0.58571428571428574, 'nl': 0.6404494382022472},
            'it': {'hi': 0.57333333333333336, 'es': 0.20000000000000001, 'no': 0.44262295081967212,
                   'sl': 0.42857142857142855, 'nl': 0.46052631578947367, 'et': 0.48333333333333334,
                   'sv': 0.42028985507246375, 'el': 0.36263736263736263, 'da': 0.44444444444444442,
                   'pt': 0.18461538461538463, 'pl': 0.40000000000000002, 'fr': 0.26666666666666666,
                   'he': 0.49450549450549453, 'fi': 0.46153846153846156, 'ro': 0.25757575757575757,
                   'ta': 0.77966101694915257, 'cs': 0.55932203389830504, 'en': 0.31868131868131866,
                   'hr': 0.50819672131147542, 'id': 0.51111111111111107, 'bg': 0.33333333333333331,
                   'fa': 0.59999999999999998, 'hu': 0.55555555555555558, 'de': 0.43333333333333335,
                   'ar': 0.52941176470588236},
            'nl': {'he': 0.57777777777777772, 'cs': 0.58461538461538465, 'fr': 0.4157303370786517,
                   'no': 0.40000000000000002, 'hi': 0.51219512195121952, 'et': 0.63636363636363635,
                   'sv': 0.34615384615384615, 'fa': 0.59036144578313254, 'pl': 0.52439024390243905,
                   'ro': 0.54166666666666663, 'sl': 0.59615384615384615, 'da': 0.36507936507936506,
                   'ta': 0.70769230769230773, 'hr': 0.57352941176470584, 'fi': 0.53846153846153844,
                   'bg': 0.57971014492753625, 'el': 0.50549450549450547, 'id': 0.6404494382022472,
                   'hu': 0.58888888888888891, 'de': 0.17391304347826086, 'pt': 0.54838709677419351,
                   'en': 0.34065934065934067, 'es': 0.44444444444444442, 'ar': 0.74285714285714288,
                   'it': 0.46052631578947367},
            'no': {'sl': 0.43478260869565216, 'sv': 0.16393442622950818, 'ar': 0.63636363636363635,
                   'he': 0.5714285714285714, 'de': 0.35064935064935066, 'et': 0.51724137931034486,
                   'hi': 0.56923076923076921, 'hu': 0.54666666666666663, 'pl': 0.47297297297297297,
                   'en': 0.31578947368421051, 'fi': 0.48684210526315791, 'el': 0.45454545454545453,
                   'id': 0.53947368421052633, 'fa': 0.61194029850746268, 'cs': 0.50877192982456143,
                   'bg': 0.44776119402985076, 'fr': 0.38157894736842107, 'ro': 0.42857142857142855,
                   'da': 0.16129032258064516, 'hr': 0.48076923076923078, 'it': 0.44262295081967212,
                   'ta': 0.68888888888888888, 'es': 0.46666666666666667, 'pt': 0.49122807017543857,
                   'nl': 0.40000000000000002},
            'pl': {'hu': 0.4845360824742268, 'da': 0.43548387096774194, 'no': 0.47297297297297297,
                   'id': 0.47474747474747475, 'pt': 0.3968253968253968, 'de': 0.45744680851063829,
                   'sv': 0.47945205479452052, 'fr': 0.42857142857142855, 'ar': 0.48484848484848486,
                   'bg': 0.29113924050632911, 'cs': 0.375, 'it': 0.40000000000000002, 'he': 0.43023255813953487,
                   'fi': 0.39393939393939392, 'hr': 0.34848484848484851, 'nl': 0.52439024390243905,
                   'ro': 0.37349397590361444, 'et': 0.38461538461538464, 'hi': 0.40229885057471265,
                   'sl': 0.30769230769230771, 'en': 0.37373737373737376, 'fa': 0.550561797752809,
                   'es': 0.39795918367346939, 'ta': 0.66666666666666663, 'el': 0.30303030303030304},
            'pt': {'hu': 0.55882352941176472, 'pl': 0.3968253968253968, 'hr': 0.46296296296296297,
                   'cs': 0.57692307692307687, 'it': 0.18461538461538463, 'el': 0.3188405797101449,
                   'da': 0.47457627118644069, 'no': 0.49122807017543857, 'ro': 0.26153846153846155,
                   'fr': 0.3188405797101449, 'en': 0.36231884057971014, 'fa': 0.55932203389830504,
                   'es': 0.23529411764705882, 'fi': 0.47826086956521741, 'sv': 0.453125, 'hi': 0.58333333333333337,
                   'ta': 0.81132075471698117, 'ar': 0.4838709677419355, 'id': 0.55072463768115942,
                   'bg': 0.35714285714285715, 'sl': 0.36170212765957449, 'et': 0.45454545454545453,
                   'nl': 0.54838709677419351, 'de': 0.49275362318840582, 'he': 0.44927536231884058},
            'ro': {'pl': 0.37349397590361444, 'sv': 0.44, 'pt': 0.26153846153846155, 'hu': 0.48351648351648352,
                   'it': 0.25757575757575757, 'bg': 0.30379746835443039, 'id': 0.51086956521739135,
                   'en': 0.34782608695652173, 'el': 0.31521739130434784, 'hr': 0.45161290322580644,
                   'de': 0.47126436781609193, 'no': 0.42857142857142855, 'fi': 0.44565217391304346, 'ar': 0.5,
                   'ta': 0.69841269841269837, 'fa': 0.51219512195121952, 'nl': 0.54166666666666663,
                   'et': 0.44444444444444442, 'hi': 0.50617283950617287, 'es': 0.31868131868131866,
                   'fr': 0.40659340659340659, 'da': 0.44262295081967212, 'cs': 0.55932203389830504,
                   'sl': 0.41304347826086957, 'he': 0.40000000000000002},
            'sl': {'es': 0.39622641509433965, 'id': 0.54716981132075471, 'hi': 0.45652173913043476,
                   'hr': 0.22448979591836735, 'hu': 0.50943396226415094, 'nl': 0.59615384615384615,
                   'pt': 0.36170212765957449, 'it': 0.42857142857142855, 'fi': 0.45283018867924529,
                   'pl': 0.30769230769230771, 'fr': 0.47169811320754718, 'sv': 0.42307692307692307,
                   'ro': 0.41304347826086957, 'et': 0.40425531914893614, 'da': 0.4375, 'en': 0.33962264150943394,
                   'ar': 0.51724137931034486, 'el': 0.39622641509433965, 'no': 0.43478260869565216,
                   'bg': 0.28260869565217389, 'fa': 0.58695652173913049, 'he': 0.49056603773584906,
                   'de': 0.56603773584905659, 'ta': 0.72222222222222221, 'cs': 0.375},
            'sv': {'hr': 0.47761194029850745, 'en': 0.25287356321839083, 'fr': 0.41176470588235292,
                   'hi': 0.56164383561643838, 'no': 0.16393442622950818, 'ro': 0.44, 'pt': 0.453125, 'et': 0.5,
                   'nl': 0.34615384615384615, 'id': 0.62790697674418605, 'ar': 0.6470588235294118,
                   'de': 0.38554216867469882, 'cs': 0.47692307692307695, 'pl': 0.47945205479452052,
                   'fa': 0.60256410256410253, 'fi': 0.45977011494252873, 'he': 0.54651162790697672,
                   'el': 0.47126436781609193, 'es': 0.40697674418604651, 'bg': 0.46575342465753422,
                   'it': 0.42028985507246375, 'sl': 0.42307692307692307, 'da': 0.15384615384615385, 'ta': 0.6875,
                   'hu': 0.57647058823529407},
            'ta': {'pl': 0.66666666666666663, 'fi': 0.58620689655172409, 'it': 0.77966101694915257,
                   'en': 0.67816091954022983, 'et': 0.59999999999999998, 'hu': 0.55294117647058827,
                   'he': 0.63953488372093026, 'fr': 0.71764705882352942, 'es': 0.74117647058823533, 'sv': 0.6875,
                   'id': 0.79069767441860461, 'cs': 0.60416666666666663, 'pt': 0.81132075471698117,
                   'da': 0.69387755102040816, 'nl': 0.70769230769230773, 'hi': 0.45348837209302323,
                   'sl': 0.72222222222222221, 'bg': 0.68421052631578949, 'hr': 0.64000000000000001,
                   'fa': 0.62068965517241381, 'no': 0.68888888888888888, 'el': 0.6705882352941176,
                   'de': 0.6785714285714286, 'ar': 0.78787878787878785, 'ro': 0.69841269841269837}}

    dot_position = sys.argv[1].find('.')
    lng = sys.argv[1][dot_position - 2:dot_position]
    wals_map = dict(sorted(softmax_wall(invert(wals[lng]), temperature=1).items()))

    print('Wals = ' + sys.argv[2])
    if sys.argv[2] == 'True':
        wals = True
        pickle_name = '.wals.vote'
        temperature = sys.argv[3]
    else:
        pickle_name = '.vote'
        wals = False

    projections = glob.glob('proj/' + sys.argv[1])
    print(sys.argv[1])
    print()

    start = time.time()
    target = vote(projections, wals)
    print((time.time() - start) / 60)

    pickle.dump(target, open('vote/' + lng + pickle_name, 'wb'))


