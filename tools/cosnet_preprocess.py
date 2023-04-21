from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import argparse
from random import shuffle, seed
import string
import itertools
# non-standard dependencies:
# import h5py
import numpy as np
import torch
import torchvision.models as models
# import skimage.io
import pickle as pkl
import nltk
from nltk.corpus import stopwords
from collections import Counter, defaultdict
import sng_parser
import spacy


new_words = ['painting', 'painter', 'picture', 'look', 'day', 'colours', 'sense', 'color', 'colour', 'colours',
             'colour', 'background', 'I', 'colouring', 'artwork', 'beginning', 'end', 'ending', 'begin',
             'atmosphere', 'image', 'activity', 'something', 'nothing', 'anything', 'emotion', 'feeling',
             'size', 'scene', 'use', 'work', 'lack', 'detail', 'place', 'landscape', 'environment', 'posture',
             'distance', 'one', 'situation', 'right']
stopwords = stopwords.words('english')
stopwords.extend(new_words)


def get_concept(imgs, counts, stopwords):
    """
    0:positive,1:negative,2:something else
    该函数从每一个image下对应的多个caption中提取head中的名词，并选取positive以及negative倾向中出现频次最高的前五个作为该image的物体label
    """
    concept = defaultdict(Counter)
    count_thr = 5
    for img in imgs['utterance_result']:
        emotion_label = img['emotion_label']
        spell = img['utterance_spelled']
        graph = sng_parser.parse(spell)
        object = defaultdict(list)
        positive_set = {0, 1, 2, 3}
        negative_set = {4, 5, 6, 7}
        for entity in graph['entities']:             # 得到场景图中的head，并将其写入对应的positive/negative字典中
            word = entity['lemma_head'].split(' ')   # lemma head有可能是多个词组成的词组，将其split并分别送入list中
            for w in word:
                token = nltk.word_tokenize(w)
                tags = nltk.pos_tag(token)
                if (w not in stopwords) and (tags[0][1] == 'NN'):   # 保证单词不在stopwords中且单词的词性为名词
                    if emotion_label in positive_set:
                        object[0].append(w)
                    elif emotion_label in negative_set:
                        object[1].append(w)
                    else:
                        object[2].append(w)
        for k, v in object.items():
            concept[k].update(v)
    concept = {k: v.most_common() for k, v in concept.items()}
    concept_dict = {key: [words[0] for words in list] for key, list in concept.items()}
    # 把counter类型的词典转化为普通词典
    for k, v in concept_dict.items():
        concept_dict[k] = [w if counts.get(w, 0) > count_thr else 'UNK' for w in concept_dict[k]]
    # 把未达到thr的词用UNK代替
    final_dict = {key: [word for word in word_list if word != 'UNK'] for key, word_list in concept_dict.items()}
    # 移除concept_dict里的UNK
    for k, v in final_dict.items():
        final_dict[k] = v[:5]
    return final_dict


def get_emotion(imgs, counts):
    """
    该函数提取每个image下caption中的形容词及副词，并选取每个情感倾向下出现频次最高的前5个词作为该image的某个情感倾向的label
    """
    model = spacy.load('en_core_web_sm')
    emo = defaultdict(Counter)
    count_thr = 5
    for img in imgs['utterance_result']:
        emotion_label = img['emotion_label']
        spell = img['utterance_spelled']
        sent = model(spell)
        adj = [token.lemma_ for token in sent if token.pos_ == 'ADJ']
        for word in adj:
            emo[emotion_label].update([word])
    emo = {k: v.most_common() for k, v in emo.items()}
    emo_dict = {key: [words[0] for words in list] for key, list in emo.items()}
    for k, v in emo_dict.items():
        emo_dict[k] = [w if counts.get(w, 0) > count_thr else 'UNK' for w in emo_dict[k]]
    final_dict = {key: [word for word in word_list if word != 'UNK'] for key, word_list in emo_dict.items()}
    for k, v in final_dict.items():
        final_dict[k] = v[:5]
    return final_dict


def build_vocab(imgs, params, stopwords):
    """
    该函数对传入的json做最初的处理，包括去除出现次数少于阈值的单词，将需要的信息写入image中
    """
    count_thr = params['word_count_threshold']

    # count up the number of words
    counts = {}
    for img in imgs:
        for sent in img['utterance_result']:
            df = []
            df.extend(sent['tokens'].strip('[]').replace("'", "").replace('"', '').split(', '))
            for w in df:
                counts[w] = counts.get(w, 0) + 1
    cw = sorted([(count, w) for w, count in counts.items()], reverse=True)
    print('top words and their counts:')
    print('\n'.join(map(str, cw[:20])))

    # print some stats
    total_words = sum(counts.values())
    print('total words:', total_words)
    bad_words = [w for w, n in counts.items() if n <= count_thr]

    # vocab = ['.'] if params['include_bos'] else []
    vocab = [w for w, n in counts.items() if n > count_thr]

    bad_count = sum(counts[w] for w in bad_words)
    print('number of bad words: %d/%d = %.2f%%' % (len(bad_words), len(counts), len(bad_words) * 100.0 / len(counts)))
    print('number of words in vocab would be %d' % (len(vocab),))
    print('number of UNKs: %d/%d = %.2f%%' % (bad_count, total_words, bad_count * 100.0 / total_words))

    # lets look at the distribution of lengths as well
    sent_lengths = {}
    for img in imgs:
        concept = get_concept(img, counts, stopwords)
        img['concept'] = concept
        emo = get_emotion(img, counts)
        img['emotion'] = emo
        for sent in img['utterance_result']:
            txt = sent['tokens'].replace(" ","").strip('[]').split(',')
            nw = len(txt)
            sent_lengths[nw] = sent_lengths.get(nw, 0) + 1
    max_len = max(sent_lengths.keys())
    print('max length sentence in raw data: ', max_len)
    print('sentence length distribution (count, number of words):')
    sum_len = sum(sent_lengths.values())
    for i in range(max_len + 1):
        print('%2d: %10d   %f%%' % (i, sent_lengths.get(i, 0), sent_lengths.get(i, 0) * 100.0 / sum_len))

    # lets now produce the final annotations
    if bad_count > 0:
        # additional special UNK token we will use below to map infrequent words to
        print('inserting the special UNK token')
        vocab.append('UNK')

    for img in imgs:
        img['final_captions'] = {}
        img['emo_name'] = []
        img['emo_label'] = []
        for num, sent in enumerate(img['utterance_result']):
            emo_name = sent['emotion']
            label = sent['emotion_label']
            txt = sent['tokens'].strip('[]').replace("'", "").replace('"', '').split(', ')
            caption = [w if counts.get(w, 0) > count_thr else 'UNK' for w in txt]
            img['emo_name'].append(emo_name)
            img['final_captions'][num] = caption
            img['emo_label'].append(label)
            # emo_embedidng.append(emo_glove)
    return vocab

def encode_captions(imgs, params, wtoi):
    """
    encode captions of the same video into one 2-D array,
    also produces a dict to point out the array based on image id
    """
    max_length = params['max_length']

    datalist = {"train": [], "val": [], "test": []}
    min_cap_count = 10000
    def get_token_ids(img):
        input_List = []
        output_List = []
        for key in img['final_captions']:
            input_Li = np.zeros((1, max_length + 1), dtype='uint32')
            output_Li = np.zeros((1, max_length + 1), dtype='int32') - 1
            caption = img['final_captions'][key]
            for k, w in enumerate(caption):
                if k < max_length:
                    input_Li[0, k + 1] = wtoi[w]  # one shift for <BOS>
                    output_Li[0, k] = wtoi[w]
            seq_len = len(caption)
            if seq_len <= max_length:
                output_Li[0, seq_len] = 0
            else:
                output_Li[0, max_length] = wtoi[caption[max_length]]
            input_List.append(input_Li)
            output_List.append(output_Li)
        return input_List, output_List

    def get_concept_ids(concept):
        """
        将concepts编码成词表形式，每一个的长度不固定
        """
        output_list = {}
        for k, v in concept.items():
            output_Li = np.zeros((1, len(v)), dtype='int32')
            for num, word in enumerate(v):
                output_Li[0, num] = wtoi[word]
            output_list[k] = output_Li
        return output_list

    def get_emowords_ids(emowords):
        """
        将emo words编码成词表格式，每一个的长度不固定
        """
        output_list = {}
        for k, v in emowords.items():
            output_Li = np.zeros((1, len(v)), dtype='int32')
            for num, word in enumerate(v):
                output_Li[0, num] = wtoi[word]
            output_list[k] = output_Li
        return output_list

    def get_id_label(img_id):
        """
        将image_id分割后，经由编码，作为一个额外的label。每一个的长度不固定
        """
        text = img_id.split('_')[1]
        model = spacy.load('en_core_web_sm')
        sent = model(text)
        adj = [token.lemma_ for token in sent if token.pos_ == 'ADJ']
        noun = [token.lemma_ for token in sent if token.pos_ == 'NOUN']
        # adj_list = np.zeros((1, len(adj)), dtype='int32')
        # for num, word in enumerate(adj):
        #     adj_list[0, num] = wtoi[word]
        # noun_list = np.zeros((1, len(noun)), dtype='int32')
        # for num, word in enumerate(adj):
        #     noun_list[0, num] = wtoi[word]
        return adj, noun





    for img in imgs:
        split = img["split"]
        img_id = img["painting"]
        art_style = img['art_style']
        n = len(img['final_captions'])
        min_cap_count = min(min_cap_count, n)
        emo_name = img['emo_name']
        emo_label = img['emo_label']
        concepts = img['concept']
        emo_word = img['emotion']
        concept_ids = get_concept_ids(concepts)
        emoword_ids = get_emowords_ids(emo_word)
        noun_from_id_ids, adj_from_id_ids = get_id_label(img_id)
        input_Li, output_Li = get_token_ids(img)
        # for index, s in enumerate(img['final_captions']):
        #     input_Li = np.zeros((1, max_length + 1), dtype='uint32')
        #     output_Li = np.zeros((1, max_length + 1), dtype='int32') - 1
        #     emotion_embedding = emo_embedding[index]
        #     for k, w in enumerate(s):
        #         if k < max_length:
        #             input_Li[0, k + 1] = wtoi[w]  # one shift for <BOS>
        #             output_Li[0, k] = wtoi[w]
        #
        #     seq_len = len(s)
        #     if seq_len <= max_length:
        #         output_Li[0, seq_len] = 0
        #     else:
        #         output_Li[0, max_length] = wtoi[s[max_length]]

        new_data = {
                "image_id": img_id,
                "tokens_ids": input_Li,
                "target_ids": output_Li,
                "emo_name": emo_name,
                "emo_label": emo_label,
                "concepts": concepts,
                "concept_ids": concept_ids,
                "art_style": art_style,
                "emo_word": emo_word,
                "emoword_ids": emoword_ids,
                "noun_from_id_ids": noun_from_id_ids,
                "adj_from_id_ids": adj_from_id_ids
            }
        datalist[split].append(new_data)
    return datalist

def save_pkl_file(datalist, output_dir):
    for split in datalist:
        pkl.dump(datalist[split], open(os.path.join(output_dir, "artemis_caption_anno_{}.pkl".format(split)), "wb"))


def save_id_file(imgs, output_dir):
    ids = {"train": [], "val": [], "test": [], }

    for img in imgs:
        split = img["split"]
        img_id = img["painting"]
        if split == "train":
            for j, _ in enumerate(img["utterance_result"]):
                ids[split].append("{}_{}".format(img_id, j))
        else:
            ids[split].append(img_id)

    for split, _ids in ids.items():
        with open(os.path.join(output_dir, "{}_ids.txt".format(split)), "w") as fout:
            for imgid in _ids:
                fout.write("{}\n".format(imgid))


def save_split_json_file(imgs, output_dir):
    split_data = {"train": {"images": [], "annotations": []},
                  "val": {"images": [], "annotations": []},
                  "test": {"images": [], "annotations": []},
                  }

    for img in imgs:
        split = img["split"]

        new_image = {
            "id": img["painting"],
            "file_name": img["painting"]
        }

        split_data[split]["images"].append(new_image)

        for sent in img["utterance_result"]:
            new_caption = {
                "image_id": img["painting"],
                "id": sent["ID"],
                "caption": sent["utterance_spelled"]
            }
            split_data[split]["annotations"].append(new_caption)

    for split, data in split_data.items():
        if split == "train":
            continue
        json.dump(data, open(os.path.join(output_dir, "captions_{}_artemis.json".format(split)), "w"))


def main(params):
    if not os.path.exists(params["output_dir"]):
        os.makedirs(params["output_dir"])

    imgs = json.load(open(params['input_json'], 'r'))


    seed(123)  # make reproducible

    # create the vocab
    vocab = build_vocab(imgs, params, stopwords)
    itow = {i + 1: w for i, w in enumerate(vocab)}  # a 1-indexed vocab translation table
    wtoi = {w: i + 1 for i, w in enumerate(vocab)}  # inverse table
    json.dump(wtoi, open(os.path.join(params['output_dir'], "wtoi.json"), "w"))
    print(len(vocab))
    with open(os.path.join(params["output_dir"], "artemis_vocabulary.txt"), "w") as fout:
        for w in vocab:
            fout.write("{}\n".format(w))

    # encode captions in large arrays, ready to ship to hdf5 file
    datalist = encode_captions(imgs, params, wtoi)
    # create output file
    save_pkl_file(datalist, params['output_dir'])
    save_split_json_file(imgs, params['output_dir'])
    #
    # json.dump(concept_pool, open(os.path.join(params['output_dir'], "concept_pool.json"), "w"))
    # pkl.dump(concept_pool_ids, open(os.path.join(params['output_dir'], "concept_pool_ids.pkl"), "wb"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--input_json', default="/home/wyf/artemis_100.json", help='input json file to process into hdf5')
    parser.add_argument('--output_dir', default="/home/wyf/open_source_dataset/for_debug/4.20/", help='output directory')
    parser.add_argument('--input_emotion', default = "/home/wyf/emotion_embedding/", help='get emotion embedding file')

    # options
    parser.add_argument('--max_length', default=17, type=int,
                        help='max length of a caption, in number of words. captions longer than this get clipped.')
    parser.add_argument('--word_count_threshold', default=5, type=int,
                        help='only words that occur more than this number of times will be put in vocab')

    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    print('parsed input parameters:')
    print(json.dumps(params, indent=2))
    main(params)