# 从get_concept_pool.py获取concept pool，去除名词以外的词得到most_common.json（命名写错）。然后挑选出现次数大于10的词作为concept_pool


import json
from collections import Counter, defaultdict
import nltk

# concept_pool = json.load(open("/home/wyf/open_source_dataset/artemis_dataset/4.15/concept_pool.json", 'r'))
# clean_dict = defaultdict(dict)
# for style in concept_pool:
#     for emotion in concept_pool[style]:
#         nn = []
#         for word in concept_pool[style][emotion]:
#             # 这里的word是counter dict得到的，word[0]是单词，word[1]是出现次数
#             select_word = nltk.word_tokenize(word[0])
#             tags = nltk.pos_tag(select_word)
#             if tags[0][1] == 'NN':
#                 nn.append(word)
#         clean_dict[style][emotion] = nn
# json.dump(clean_dict, open("/home/wyf/open_source_dataset/artemis_dataset/4.15/most_common.json", "w"))

concept_pool = json.load(open("/home/wyf/open_source_dataset/artemis_dataset/4.19/concept_pool.json", 'r'))
clean_dict = defaultdict(dict)
clean_dict_i = defaultdict(dict)
for style in concept_pool:
    for emotion in concept_pool[style]:
        nn = []
        inn = []
        for word in concept_pool[style][emotion]:
            if word[1] >= 20:
                select_word = nltk.word_tokenize(word[0])
                tags = nltk.pos_tag(select_word)
                if tags[0][1] == 'NN':
                    nn.append(word)
                    inn.append(word[0])
            else:
                break
        clean_dict[style][emotion] = nn
        clean_dict_i[style][emotion] = inn
json.dump(clean_dict, open("/home/wyf/open_source_dataset/artemis_dataset/4.19/clean_dict.json", "w"))
json.dump(clean_dict_i, open("/home/wyf/open_source_dataset/artemis_dataset/4.19/clean_dict_i.json", "w"))

for style in clean_dict_i:
    print(f"len of {style} is {len(clean_dict_i[style]['0'])}")

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
