import json
import tqdm
import nltk
from collections import Counter, defaultdict
import numpy as np
import os
import pickle
from multiprocessing import Pool
corpus_type = 'part'
senti_corpus_g = json.load(open("/home/wyf/artemis_fullcombined.json"))
corpus_dir = "/home/wyf/open_source_dataset/artemis_dataset/new/"
# with open ("/home/wyf/open_source_dataset/artemis_dataset/artemis_vocabulary.txt") as f:
#     vocab = f.read()
# vocab = vocab.split()
# wtoi = {w: i+1 for i, w in enumerate(vocab)}
chunk_size = len(senti_corpus_g) // 20
chunks = [senti_corpus_g[i:i+chunk_size] for i in range(0, len(senti_corpus_g), chunk_size)]
def create_defaultdict():
    return defaultdict(Counter)
def process_chunk(chunk):
    senti_corpus = chunk
    tmp_senti_corpus = defaultdict(list)
    tmp_senti_corpus_pos = defaultdict(list)
    all_sentis = Counter()
    sentis = defaultdict(Counter)
    sentiment_detector = defaultdict(create_defaultdict)
    # sent_num = len(senti_corpus)
    negative_set = {'sadness', 'disgust', 'fear', 'anger'}
    positive_set = {'amusement', 'awe', 'contentment', 'excitement'}
    objects = []
    for i in tqdm.tqdm(range(0, len(senti_corpus), 100)):
        cur_lines = senti_corpus[i:i + 100]
        tmp_sents = []
        for line in cur_lines:
            for sent in line['utterance_result']:
                senti_label = sent['emotion']
                tmp_sents.append(nltk.word_tokenize(sent['utterance_spelled']))
                tagged_sents = nltk.pos_tag_sents(tmp_sents, tagset='universal')
                for tagged_tokens in tagged_sents:
                    words = []
                    poses = []
                    nouns = []
                    adjs = []
                    for w, p in tagged_tokens:
                        words.append(w)
                        poses.append(p)
                        if p == 'ADJ':
                            adjs.append(w)
                        elif p == 'NOUN':
                            nouns.append(w)
                    for item in nouns:
                        objects.append(item)
                    tmp_senti_corpus[senti_label].append(words)
                    tmp_senti_corpus_pos[senti_label].append(poses)
                    if adjs:
                        all_sentis.update(adjs)
                        sentis[senti_label].update(adjs)
                        for noun in nouns:
                            sentiment_detector[senti_label][noun].update(adjs)
    ret = {}
    ret.update({'sentiment_detector': sentiment_detector,
               'all_sentis': all_sentis,
                'objects': objects,
                'sentis': sentis

    })
    return ret
with Pool(20) as p:
    results = p.map(process_chunk, chunks)
# json.dump(tmp_senti_corpus, open(os.path.join(corpus_dir, 'for_debug_tmp_senti_corpus.json'), 'w'))
# json.dump(tmp_senti_corpus_pos, open(os.path.join(corpus_dir, 'for_debug_tmp_senti_corpus.json'), 'w'))
length_sadness = 0
length_contentment = 0
length_anger = 0
length_awe = 0
length_fear = 0
length_disgust = 0
length_s = 0
length_amusement = 0
length_e = 0
all_sentis = {}
sentis = {}
sentiment_detector = {}
for i, result in enumerate(results):
    objects = np.unique(result['objects'])
    # with open('objects.pkl', 'wb') as f:
    #     pickle.dump(objects, f)
    all_senti = result['all_sentis'].most_common()
    all_sentis[i] = [w for w in all_senti]
    sentis[i] = {k: v.most_common() for k, v in result['sentis'].items()}
    sentiment_detector[i] = {k: dict(v) for k, v in result['sentiment_detector'].items()}
# json.dump(sentiment_detector, open("/home/wyf/open_source_dataset/artemis_dataset/senti_detector.json", 'w'))
# json.dump(sentis, open("/home/wyf/open_source_dataset/artemis_dataset/sentis.json", 'w'))

    for objs, sentis in sentiment_detector[i]['sadness'].items():
        length_sadness += len(sentiment_detector[i]['sadness'][objs])
    print("sadness:", length_sadness)

    for objs, sentis in sentiment_detector[i]['contentment'].items():
        length_contentment += len(sentiment_detector[i]['contentment'][objs])
    print("contentment:", length_contentment)

    for objs, sentis in sentiment_detector[i]['anger'].items():
        length_anger += len(sentiment_detector[i]['anger'][objs])
    print("anger:", length_anger)

    for objs, sentis in sentiment_detector[i]['awe'].items():
        length_awe += len(sentiment_detector[i]['awe'][objs])
    print("awe:", length_awe)

    for objs, sentis in sentiment_detector[i]['fear'].items():
        length_fear += len(sentiment_detector[i]['fear'][objs])
    print("fear:", length_fear)

    for objs, sentis in sentiment_detector[i]['disgust'].items():
        length_disgust += len(sentiment_detector[i]['disgust'][objs])
    print("disgust:", length_disgust)

    for objs, sentis in sentiment_detector[i]['something else'].items():
        length_s += len(sentiment_detector[i]['something else'][objs])
    print("something else:", length_s)

    for objs, sentis in sentiment_detector[i]['excitement'].items():
        length_e += len(sentiment_detector[i]['excitement'][objs])
    print("excitement:", length_e)

    for objs, sentis in sentiment_detector[i]['amusement'].items():
        length_amusement += len(sentiment_detector[i]['amusement'][objs])
    print("amusement:", length_amusement)

with open("/home/wyf/open_source_dataset/for_debug/len.txt", 'w') as f:
        f.write(length_sadness, length_contentment, length_anger, length_awe,  length_fear,  length_disgust,  length_s,  length_amusement,
             length_e)

