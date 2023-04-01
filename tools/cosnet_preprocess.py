from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import argparse
from random import shuffle, seed
import string
# non-standard dependencies:
# import h5py
import numpy as np
import torch
import torchvision.models as models
# import skimage.io
# from PIL import Image
import pickle as pkl
import nltk
from collections import Counter, defaultdict
import sng_parser
from nltk.corpus import stopwords

new_words = ['painting', 'painter', 'picture', 'look', 'day', 'colours', 'sense', 'color', 'colour', 'colours',
             'colour', 'background', 'I', 'colouring', 'artwork', 'beginning', 'end', 'ending', 'begin',
             'atmosphere', 'image', 'activity', 'something', 'nothing', 'anything',
             'size', 'scene', 'use', 'work']
stopwords = stopwords.words('english')
stopwords.extend(new_words)


def get_concept(imgs, counts, stopwords):
    # 0:positive,1:negative,2:something else
    concept = defaultdict(Counter)
    count_thr = 3
    for img in imgs['utterance_result']:
        emotion_label = img['emotion_label']
        spell = img['utterance_spelled']
        graph = sng_parser.parse(spell)
        object = defaultdict(list)
        positive_set = {1, 2, 3, 4}
        negative_set = {5, 6, 7, 8}
        for entity in graph['entities']:
            word = entity['lemma_head']
            if word not in stopwords:
                if emotion_label in positive_set:
                    object[0].append(word)
                elif emotion_label in negative_set:
                    object[1].append(word)
                else:
                    object[2].append(word)
        for k, v in object.items():
            concept[k].update(v)
    for k, v in concept.items():
        v = v.most_common()
    concept_dict = {key: [word for word, count in counter.items()] for key, counter in concept.items()}
    # 把counter类型的词典转化为普通词典
    for k, v in concept_dict.items():
        concept_dict[k] = [w if counts.get(w, 0) > count_thr else 'UNK' for w in concept_dict[k]]
    # 把未达到thr的词用UNK代替
    final_dict = {key: [word for word in word_list if word != 'UNK'] for key, word_list in concept_dict.items()}
    # 移除concept_dict里的UNK
    for k, v in final_dict.items():
        final_dict[k] = v[:5]
        # if len(concept) > 5:
        #     concept = concept[:5]
        # else:
        #     while len(concept) < 5:
        #         for relation in relations:
        #             word = relation['lemma_realtion']
        #             if word not in stop_words:
        #                 concept.append(word)
    return final_dict


def build_vocab(imgs, params, stopwords):
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
        img['emo_embedding'] = []
        img['emo_name'] = []
        img['emo_label'] = []
        for num, sent in enumerate(img['utterance_result']):
            emo_name = sent['emotion']
            label = sent['emotion_label']
            emo_glove = np.load("{0}.npy".format(os.path.join(params['input_emotion'], emo_name)))
            txt = sent['tokens'].strip('[]').replace("'", "").replace('"', '').split(', ')
            caption = [w if counts.get(w, 0) > count_thr else 'UNK' for w in txt]
            img['emo_embedding'].append(emo_glove)
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
        emotion_embedding = []
        for key in img['final_captions']:
            input_Li = np.zeros((1, max_length + 1), dtype='uint32')
            output_Li = np.zeros((1, max_length + 1), dtype='int32') - 1
            emotion = emo_embedding[key]
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
            emotion_embedding.append(emotion)
        return input_List, output_List, emotion_embedding

    def get_attr_ids(concept):
        # 将concept_label编码成词表形式，每一个情感倾向对应的concept列表长度都为5
        output_list = {}
        for k, v in concept.items():
            # 把补上的用-1来替代
            output_Li = np.zeros((1, 5) - 1, dtype='int32')
            for num, word in enumerate(v):
                output_Li[0, num] = wtoi[word]
            output_list[k] = (output_Li)
        return output_list

    for img in imgs:
        split = img["split"]
        img_id = img["painting"]
        n = len(img['final_captions'])
        min_cap_count = min(min_cap_count, n)
        emo_embedding = img['emo_embedding']
        emo_name = img['emo_name']
        emo_label = img['emo_label']
        concepts = img['concept']
        attr_gt = get_attr_ids(concepts)
        input_Li, output_Li, emotion_embedding = get_token_ids(img)
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
                "emotion_embedding": emotion_embedding,
                "emo_name": emo_name,
                "emo_label": emo_label,
                "concepts": concepts,
                "attr_gt": attr_gt
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

    print(len(vocab))
    with open(os.path.join(params["output_dir"], "artemis_vocabulary.txt"), "w") as fout:
        for w in vocab:
            fout.write("{}\n".format(w))

    # encode captions in large arrays, ready to ship to hdf5 file
    datalist = encode_captions(imgs, params, wtoi)

    # create output file
    save_pkl_file(datalist, params['output_dir'])
    save_split_json_file(imgs, params['output_dir'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--input_json', default="/home/wyf/open_source_dataset/for_debug/data_10.json", help='input json file to process into hdf5')
    parser.add_argument('--output_dir', default='.', help='output directory')
    parser.add_argument('--input_emotion', default = "/home/wyf/emotion_embedding/", help='get emotion embedding file')

    # options
    parser.add_argument('--max_length', default=30, type=int,
                        help='max length of a caption, in number of words. captions longer than this get clipped.')
    parser.add_argument('--word_count_threshold', default=3, type=int,
                        help='only words that occur more than this number of times will be put in vocab')

    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    print('parsed input parameters:')
    print(json.dumps(params, indent=2))
    main(params)