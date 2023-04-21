import json
from collections import Counter, defaultdict
import nltk
import numpy as np
import pickle as pkl


#############################################################################################################
# 该脚本用于生成clean_dict和clean_dict_i
# concept_pool = json.load(open("/home/wyf/open_source_dataset/artemis_dataset/4.18/concept_pool.json", 'r'))
# concept_dict = json.load(open("/home/wyf/open_source_dataset/artemis_dataset/4.18/concept_dict.json", 'r'))
#
# clean_dict = {'0': [], '1': [], '2': []}
# clean_dict_i = {'0': [], '1': [], '2': []}
# for k in concept_pool:
#     for words in concept_pool[k]:
#         word = nltk.word_tokenize(words[0])
#         tags = nltk.pos_tag(word)
#         if words[1] >= 20:
#             if tags[0][1] == 'NN':
#                 clean_dict[k].append(words)
#                 clean_dict_i[k].append(word)
#         else:
#             break
#
# json.dump(clean_dict, open("/home/wyf/open_source_dataset/artemis_dataset/4.18/clean_dict.json", "w"))
# json.dump(clean_dict_i, open("/home/wyf/open_source_dataset/artemis_dataset/4.18/clean_dict_i.json", "w"))
##############################################################################################################

# 该脚本用于利用wtoi生成一个编码为ids的concept pool
# wtoi = json.load(open("/home/wyf/open_source_dataset/artemis_dataset/4.17/wtoi.json", 'r'))
# clean_dict = json.load(open("/home/wyf/open_source_dataset/artemis_dataset/4.18/clean_dict_i.json", 'r'))
# concept_list_dict = {'0': [], '1': [], '2': []}
# for k in clean_dict:
#     for word in clean_dict[k]:
#         concept_list_dict[k].extend(word)
#
# positive_list = concept_list_dict['0']
# negative_list = concept_list_dict['1']
# something_else = concept_list_dict['2']
#
# pos_len = len(positive_list)
# neg_len = len(negative_list)
# se_len = len(something_else)
#
# pos = np.zeros((1, pos_len), dtype='uint32')
# neg = np.zeros((1, neg_len), dtype='uint32')
# se = np.zeros((1, se_len), dtype='uint32')
# for num, word in enumerate(positive_list):
#     pos[0, num] = wtoi.get(word, -1)
# for num, word in enumerate(negative_list):
#     neg[0, num] = wtoi.get(word, -1)
# for num, word in enumerate(something_else):
#     se[0, num] = wtoi.get(word, -1)
#
# pos = np.squeeze(pos)
# bool_arr = pos != -1
# new_pos = np.array(pos[bool_arr])
#
# neg = np.squeeze(neg)
# bool_arr = neg != -1
# new_neg = np.array(neg[bool_arr])
#
# se = np.squeeze(se)
# bool_arr = se != -1
# new_se = np.array(se[bool_arr])
#
# print(len(new_neg), len(new_pos), len(new_se))
# with open("/home/wyf/open_source_dataset/artemis_dataset/4.18/positive_concepts.pkl", "wb") as f:
#     pkl.dump(new_pos, f)
# with open("/home/wyf/open_source_dataset/artemis_dataset/4.18/negative_concepts.pkl", "wb") as f:
#     pkl.dump(new_neg, f)
# with open("/home/wyf/open_source_dataset/artemis_dataset/4.18/se_concept.pkl", "wb") as f:
#     pkl.dump(new_se, f)
#####################################################################################################
# 该脚本用于将concept_pool依照art style分成多类,最终结果分成了四个concept_pool
# concept_pool = json.load(open("/home/wyf/open_source_dataset/artemis_dataset/4.19/clean_dict_i.json", 'r'))
# concept_pool_1 = {}
# concept_pool_2 = {}
# concept_pool_3 = {}
# concept_pool_4 = {}
# concept_pool_5 = {}
# concept_pool_6 = {}
#
# concept_pool_1['0'] = list(set(concept_pool['Realism']['0']) | set(concept_pool['Impressionism']['0']))
# concept_pool_1['1'] = list(set(concept_pool['Realism']['1']) | set(concept_pool['Impressionism']['1']))
# concept_pool_2['0'] = list(set(concept_pool['Romanticism']['0']) | set(concept_pool['Baroque']['0'])
#                            | set(concept_pool['Color_Field_Painting']['0']) | set(concept_pool['Art_Nouveau_Modern']['0']))
# concept_pool_2['1'] = list(set(concept_pool['Realism']['1']) | set(concept_pool['Impressionism']['1'])
#                            | set(concept_pool['Color_Field_Painting']['1']) | set(concept_pool['Art_Nouveau_Modern']['1']))
# concept_pool_3['0'] = list(set(concept_pool['Post_Impressionism']['0']) | set(concept_pool['Northern_Renaissance']['0'])
#                            | set(concept_pool['High_Renaissance']['0']) | set(concept_pool['Fauvism'])
#                            | set(concept_pool['Expressionism']['0']))
# concept_pool_3['1'] = list(set(concept_pool['Post_Impressionism']['1']) | set(concept_pool['Northern_Renaissance']['1'])
#                            | set(concept_pool['High_Renaissance']['1']) | set(concept_pool['Fauvism'])
#                            | set(concept_pool['Expressionism']['1']))
# concept_pool_4['0'] = list(set(concept_pool['Symbolism']['0']) | set(concept_pool['Rococo']['0'])
#                            | set(concept_pool['Naive_Art_Primitivism']['0']) | set(concept_pool['Ukiyo_e']['0'])
#                            | set(concept_pool['Minimalism']['0']))
# concept_pool_4['1'] = list(set(concept_pool['Symbolism']['1']) | set(concept_pool['Rococo']['1'])
#                            | set(concept_pool['Naive_Art_Primitivism']['1']) | set(concept_pool['Ukiyo_e']['1'])
#                            | set(concept_pool['Minimalism']['1']))
# concept_pool_5['0'] = list(set(concept_pool['Cubism']['0']) | set(concept_pool['Pop_Art']['0'])
#                            | set(concept_pool['Pointillism']['0']) | set(concept_pool['Synthetic_Cubism']['0'])
#                            | set(concept_pool['Contemporary_Realism']['0']) | set(concept_pool['Analytical_Cubism']['0'])
#                            | set(concept_pool['Action_painting']['0']) | set(concept_pool['Abstract_Expressionism']['0'])
#                            | set(concept_pool['Mannerism_Late_Renaissance']['0']) | set(concept_pool['New_Realism']['0']))
# concept_pool_5['1'] = list(set(concept_pool['Cubism']['1']) | set(concept_pool['Pop_Art']['1'])
#                            | set(concept_pool['Pointillism']['1']) | set(concept_pool['Synthetic_Cubism']['1'])
#                            | set(concept_pool['Contemporary_Realism']['1']) | set(concept_pool['Analytical_Cubism']['1'])
#                            | set(concept_pool['Action_painting']['1']) | set(concept_pool['Abstract_Expressionism']['1'])
#                            | set(concept_pool['Mannerism_Late_Renaissance']['1']) | set(concept_pool['New_Realism']['1']))
# concept_pool_6['0'] = list(set(concept_pool_4['0']) | set(concept_pool_5['0']))
# concept_pool_6['1'] = list(set(concept_pool_4['1']) | set(concept_pool_5['1']))
# print(len(concept_pool_1['0']))
# print(len(concept_pool_1['1']))
# print(len(concept_pool_2['0']))
# print(len(concept_pool_2['1']))
# print(len(concept_pool_3['0']))
# print(len(concept_pool_3['1']))
# print(len(concept_pool_4['0']))
# print(len(concept_pool_4['1']))
# print(len(concept_pool_5['0']))
# print(len(concept_pool_5['1']))
# print(len(concept_pool_6['0']))
# print(len(concept_pool_6['1']))
#
# json.dump(concept_pool_1, open("/home/wyf/open_source_dataset/artemis_dataset/4.19/pool_1.json", 'w'))
# json.dump(concept_pool_2, open("/home/wyf/open_source_dataset/artemis_dataset/4.19/pool_2.json", 'w'))
# json.dump(concept_pool_3, open("/home/wyf/open_source_dataset/artemis_dataset/4.19/pool_3.json", 'w'))
# json.dump(concept_pool_6, open("/home/wyf/open_source_dataset/artemis_dataset/4.19/pool_4.json", 'w'))
# json.dump(concept_pool_5, open("/home/wyf/open_source_dataset/artemis_dataset/4.19/pool_5.json", 'w'))

#######################################################################################################
# 该脚本将新得到的四个concept_pool转换成ids文件
# wtoi = json.load(open("/home/wyf/open_source_dataset/artemis_dataset/4.17/wtoi.json", 'r'))
# dict_1 = json.load(open("/home/wyf/open_source_dataset/artemis_dataset/4.19/pool_1.json", 'r'))
# dict_2 = json.load(open("/home/wyf/open_source_dataset/artemis_dataset/4.19/pool_2.json", 'r'))
# dict_3 = json.load(open("/home/wyf/open_source_dataset/artemis_dataset/4.19/pool_3.json", 'r'))
# dict_4 = json.load(open("/home/wyf/open_source_dataset/artemis_dataset/4.19/pool_4.json", 'r'))
#
# dict_1['0'].remove('wildflower')
# dict_1['1'].remove('doldrum')
# dict_1['1'].remove('crosse')
# dict_2['1'].remove('doldrum')
# dict_2['1'].remove('crosse')
# dict_3['1'].remove('crosse')
#
# json.dump(dict_1, open("/home/wyf/open_source_dataset/artemis_dataset/4.19/pool_1.json", 'w'))
# json.dump(dict_2, open("/home/wyf/open_source_dataset/artemis_dataset/4.19/pool_2.json", 'w'))
# json.dump(dict_3, open("/home/wyf/open_source_dataset/artemis_dataset/4.19/pool_3.json", 'w'))
#
# def encode_dict(dict, wtoi):
#     encode_dict = {'0': [], '1': []}
#     for k in dict:
#         length = len(dict[k])
#         ls = np.zeros((1, length), dtype='int32')
#         for num, word in enumerate(dict[k]):
#             try:
#                 ls[0, num] = wtoi[word]
#             except KeyError:
#                 print(f"dict{k} of {word} is not exist")
#         result = np.squeeze(ls)
#         encode_dict[k] = result
#     return encode_dict
#
#
# dict_ids_1 = encode_dict(dict_1, wtoi)
# dict_ids_2 = encode_dict(dict_2, wtoi)
# dict_ids_3 = encode_dict(dict_3, wtoi)
# dict_ids_4 = encode_dict(dict_4, wtoi)
# with open("/home/wyf/open_source_dataset/artemis_dataset/4.19/dict_1.pkl", "wb") as f:
#     pkl.dump(dict_ids_1, f)
# with open("/home/wyf/open_source_dataset/artemis_dataset/4.19/dict_2.pkl", "wb") as f:
#     pkl.dump(dict_ids_2, f)
# with open("/home/wyf/open_source_dataset/artemis_dataset/4.19/dict_3.pkl", "wb") as f:
#     pkl.dump(dict_ids_3, f)
# with open("/home/wyf/open_source_dataset/artemis_dataset/4.19/dict_4.pkl", "wb") as f:
#     pkl.dump(dict_ids_4, f)
############################################################################################
# 把训练集里所有情感倾向为8的清除
train = pkl.load(open("/home/wyf/open_source_dataset/for_debug/4.20/artemis_caption_anno_train.pkl", 'rb'),
                    encoding='bytes')
test = pkl.load(open("/home/wyf/open_source_dataset/for_debug/4.20/artemis_caption_anno_test.pkl", 'rb'),
                    encoding='bytes')
val = pkl.load(open("/home/wyf/open_source_dataset/for_debug/4.20/artemis_caption_anno_val.pkl", 'rb'),
                    encoding='bytes')
