# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Jianjie Luo
@contact: jianjieluo.sysu@gmail.com
"""

def flat_list_of_lists(l):
    """flatten a list of lists [[1,2], [3,4]] to [1,2,3,4]"""
    return [item for sublist in l for item in sublist]

# def load_data():
#     concept_pool = json.load(open("/home/wyf/open_source_dataset/artemis_dataset/4.18/clean_dict_i.json", 'r'))
#     positive_pool = concept_pool['0']
#     negative_pool = concept_pool['1']
#     positive_ids = pickle.load(open("/home/wyf/open_source_dataset/artemis_dataset/4.18/positive_concepts.pkl", 'rb'),
#                                encoding='bytes')
#     negative_ids = pickle.load(open("/home/wyf/open_source_dataset/artemis_dataset/4.18/positive_concepts.pkl", 'rb'),
#                                encoding='bytes')
#     return concept_pool, positive_pool, negative_pool, positive_ids, negative_ids
