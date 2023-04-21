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
import spacy
# import skimage.io
# from PIL import Image
import pickle as pkl


def build_vocab(imgs, params):
    count_thr = params['word_count_threshold']

    # count up the number of words
    counts = {}
    for img in imgs:
        for sent in img['utterance_result']:
            df = sent['tokens'].replace(" ","").strip('[]').split(',')
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
        img['emotion_label'] = []

        for num, sent in enumerate(img['utterance_result']):
            emo_name = sent['emotion']
            emo_label = sent['emotion_label']
            emo_glove = np.load("{0}.npy".format(os.path.join(params['input_emotion'], emo_name)))
            txt = sent['tokens'].replace(" ","").strip('[]').split(',')
            caption = [w if counts.get(w, 0) > count_thr else 'UNK' for w in txt]
            img['emo_embedding'].append(emo_glove)
            img['emotion_label'].append(emo_label)
            img['final_captions'][num] = caption
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

    for img in imgs:
        split = img["split"]
        img_id = img["painting"]
        n = len(img['final_captions'])
        min_cap_count = min(min_cap_count, n)
        emo_embedding = img['emo_embedding']
        emo_label = img['emotion_label']
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
                "emotion_label": emo_label

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
    vocab = build_vocab(imgs, params)
    itow = {i + 1: w for i, w in enumerate(vocab)}  # a 1-indexed vocab translation table
    wtoi = {w: i + 1 for i, w in enumerate(vocab)}  # inverse table
    json.dump(wtoi, open(os.path.join(output_dir, "wtoi.json"), "w"))

    # print(len(vocab))
    # with open(os.path.join(params["output_dir"], "artemis_vocabulary.txt"), "w") as fout:
    #     for w in vocab:
    #         fout.write("{}\n".format(w))

    # encode captions in large arrays, ready to ship to hdf5 file
    # datalist = encode_captions(imgs, params, wtoi)
    #
    # # create output file
    # save_pkl_file(datalist, params['output_dir'])
    # save_split_json_file(imgs, params['output_dir'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--input_json', default="/home/wyf/artemis_fullcombined.json", help='input json file to process into hdf5')
    parser.add_argument('--output_dir', default="/home/wyf/open_source_dataset/artemis_dataset/4.15/", help='output directory')
    parser.add_argument('--input_emotion', help='get emotion embedding file')

    # options
    parser.add_argument('--max_length', default=16, type=int,
                        help='max length of a caption, in number of words. captions longer than this get clipped.')
    parser.add_argument('--word_count_threshold', default=5, type=int,
                        help='only words that occur more than this number of times will be put in vocab')

    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    print('parsed input parameters:')
    print(json.dumps(params, indent=2))
    main(params)