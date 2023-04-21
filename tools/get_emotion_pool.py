import os
import json
import argparse
from random import shuffle, seed
import string
import itertools

import numpy as np
import torch

import pickle as pkl
import nltk
import spacy
from collections import Counter, defaultdict
import sng_parser
from nltk.corpus import stopwords




def get_concept(list):
    nlp_model = spacy.load('en_core_web_sm')
    emotion_pool = defaultdict(Counter)             # counter方法的特殊性，update会更新计数而不是替代内容
    emotion_dict = {}
    count_thr = 5
    for imgs in list:
        for img in imgs['utterance_result']:
            emotion_label = img['emotion_label']
            spell = img['utterance_spelled']
            sent = nlp_model(spell)
            adj = [token.text for token in sent if token.pos_ == 'ADJ']
            for word in adj:
                emotion_pool[emotion_label].update([word])
    emotion_pool = {k: v.most_common() for k, v in emotion_pool.items()}
    emotion_dict = {key: [words[0] for words in list] for key, list in emotion_pool.items()}
        # 把counter类型的词典转化为普通词典
    return emotion_pool, emotion_dict



def main(params):
    if not os.path.exists(params["output_dir"]):
        os.makedirs(params["output_dir"])

    # new_words = ['painting', 'painter', 'picture', 'look', 'day', 'colours', 'sense', 'color', 'colour', 'colours',
    #              'colour', 'background', 'I', 'colouring', 'artwork', 'beginning', 'end', 'ending', 'begin',
    #              'atmosphere', 'image', 'activity', 'something', 'nothing', 'anything', 'emotion', 'feeling',
    #              'size', 'scene', 'use', 'work', 'lack', 'detail', 'place', 'landscape', 'environment', 'posture',
    #              'distance', 'one', 'situation', 'right', 'what', 'whatever', 'wherever', 'however', 'therefore']
    # stopword = stopwords.words('english')
    # stopword.extend(new_words)

    imgs = json.load(open(params['input_json'], 'r'))
    emo_pool, emo_dict = get_concept(imgs)
    # concept_pool：counter类型，concept_dict，dict类型，依照出现次数排序
    json.dump(emo_pool, open(os.path.join(params['output_dir'], "emo_pool.json"), "w"))
    json.dump(emo_dict, open(os.path.join(params['output_dir'], "emo_dict.json"), "w"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--input_json', default="/home/wyf/artemis_fullcombined.json", help='input json file to process into hdf5')
    parser.add_argument('--output_dir', default="/home/wyf/open_source_dataset/artemis_dataset/4.18/", help='output directory')

    # options

    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    print('parsed input parameters:')
    print(json.dumps(params, indent=2))
    main(params)