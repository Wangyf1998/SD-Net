import json
from collections import Counter, defaultdict
import nltk


concept_pool = json.load(open("/home/wyf/open_source_dataset/artemis_dataset/4.14/concept_pool.json", 'r'))
concept_dict = json.load(open("/home/wyf/open_source_dataset/artemis_dataset/4.14/concept_dict.json", 'r'))

# for style in concept_dict:
#     print(f"len of {style} positive:{len(concept_dict[style]['0'])}")
#     print(f"len of {style} negative:{len(concept_dict[style]['1'])}")
#     print(f"len of {style} something else:{len(concept_dict[style]['2'])}")
most_common = defaultdict()
most_dif = defaultdict(dict)

for k, v in concept_dict.items():
    positive_set = set(concept_dict[k]['0'])
    negative_set = set(concept_dict[k]['1'])
    se = set(concept_dict[k]['2'])
    common_set = list(positive_set.intersection(negative_set))
    print(f"len of {k} positive:{len(concept_dict[k]['0'])}")
    print(f"len of {k} negative:{len(concept_dict[k]['1'])}")
    print(f"len of {k} something else:{len(concept_dict[k]['2'])}")
    print(f"length of common words for {k}: {len(common_set)}")
    dif_set_0 = list(positive_set.difference(negative_set))
    print(f"length of dif words for positive {k}: {len(dif_set_0)}")
    dif_set_1 = list(negative_set.difference(positive_set))
    print(f"length of dif words for negative {k}: {len(dif_set_1)}")
    most_common[k] = common_set
    most_dif[k]['positive'] = dif_set_0
    most_dif[k]['negative'] = dif_set_1

# json.dump(most_common, open("/home/wyf/open_source_dataset/artemis_dataset/4.15/most_common.json", "w"))
# json.dump(most_dif, open("/home/wyf/open_source_dataset/artemis_dataset/4.15/most_diff.json", "w"))
#
# former = {'NN': [], 'VB': [], 'else': []}
# clean_dict_common = defaultdict(lambda: former.copy())
# clean_dict_dif_pos = defaultdict(lambda: former.copy())
# clean_dict_dif_neg = defaultdict(lambda: former.copy())
# for k in most_common:
#     nn = []
#     vb = []
#     ad = []
#     for word in most_common[k]:
#         word = nltk.word_tokenize(word)
#         tags = nltk.pos_tag(word)
#         if tags[0][1] == 'NN':
#             nn.append(word)
#         elif tags[0][1] == 'VB':
#             vb.append(word)
#         elif tags[0][1] == 'JJ' or 'RB':
#             ad.append(word)
#         clean_dict_common[k]['NN'] = nn
#         clean_dict_common[k]['VB'] = vb
#         clean_dict_common[k]['else'] = ad
#
# for k in most_dif:
#     nnp = []
#     vbp = []
#     adp = []
#     nnn = []
#     vbn = []
#     adn = []
#     for word in most_dif[k]['positive']:
#         word = nltk.word_tokenize(word)
#         tags = nltk.pos_tag(word)
#         if tags[0][1] == 'NN':
#             nnp.append(word)
#         elif tags[0][1] == 'VB':
#             vbp.append(word)
#         elif tags[0][1] == 'JJ' or 'RB':
#             adp.append(word)
#         clean_dict_dif_pos[k]['NN'] = nnp
#         clean_dict_dif_pos[k]['VB'] = vbp
#         clean_dict_dif_pos[k]['else'] = adp
#     for word in most_dif[k]['negative']:
#         word = nltk.word_tokenize(word)
#         tags = nltk.pos_tag(word)
#         if tags[0][1] == 'NN':
#             nnn.append(word)
#         elif tags[0][1] == 'VB':
#             vbn.append(word)
#         elif tags[0][1] == 'JJ' or 'RB':
#             adn.append(word)
#         clean_dict_dif_neg[k]['NN'] = nnn
#         clean_dict_dif_neg[k]['VB'] = vbn
#         clean_dict_dif_neg[k]['else'] = adn
#
#
#
# json.dump(clean_dict_common, open("/home/wyf/open_source_dataset/artemis_dataset/4.15/clean_dict_common.json", "w"))
# json.dump(clean_dict_dif_pos, open("/home/wyf/open_source_dataset/artemis_dataset/4.15/clean_dict_dif_pos.json", "w"))
# json.dump(clean_dict_dif_neg, open("/home/wyf/open_source_dataset/artemis_dataset/4.15/clean_dict_dif_neg.json", "w"))

