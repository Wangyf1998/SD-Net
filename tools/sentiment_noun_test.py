import json
import tqdm
import nltk
from collections import Counter, defaultdict
from nltk.stem import PorterStemmer
import numpy as np
from multiprocessing import Pool

# corpus_type = 'part'
senti_corpus_g = json.load(open("/home/wyf/open_source_dataset/for_debug/data_10.json"))
with open("/home/wyf/open_source_dataset/for_debug/new_3.18/artemis_vocabulary.txt") as f:
    vocab = f.read()
# chunk_len = 5
tmp_senti_corpus = defaultdict(list)
tmp_senti_corpus_pos = defaultdict(list)
# chunk_size = len(senti_corpus_g) // chunk_len
# chunks = [senti_corpus_g[i:i + chunk_size] for i in range(0, len(senti_corpus_g), chunk_size)]


def create_defaultdict():
    return defaultdict(Counter)


# def process_chunk(chunk):
# senti_corpus = chunk
all_sentis = Counter()
sentis = defaultdict(Counter)
sentiment_detector = defaultdict(create_defaultdict)
noun_detector = defaultdict(create_defaultdict)
sent_num = len(senti_corpus)
objects = []
for i in tqdm.tqdm(range(0, len(senti_corpus), 20)):
    cur_lines = senti_corpus[i:i + 20]
    tmp_sents = []
    for lines in cur_lines:
        for k, sent in enumerate(lines['utterance_result']):
            senti_label = sent['emotion']
            #             tmp_sents = []
            #             for i in range(0, sent_num):
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
                noun_detector[k][senti_label].update(nouns)
                tmp_senti_corpus_pos[senti_label].append(poses)
                if adjs:
                    all_sentis.update(adjs)
                    sentis[senti_label].update(adjs)
                    for noun in nouns:
                        sentiment_detector[senti_label][noun].update(adjs)
    # ret = {}
    # ret.update({'sentiment_detector': sentiment_detector,
    #             'sentis': sentis,
    #             'objects': objects,
    #             'all_sentis': all_sentis,
    #             'noun_detector': noun_detector
    #
    #             })
    # return ret


# print("tmp_senti_corpus:", tmp_senti_corpus)
# print("tmp_senti_corpus_pos:", tmp_senti_corpus_pos)
# objects = {v for k,v in sentiment_detector.items()}
# objects = np.unique()
with Pool(chunk_len) as p:
    results = p.map(process_chunk, chunks)
print(len(results))
all_sentis = {}
sentis = {}
sentiment_detector = {}
for i, result in enumerate(results):
    objects = np.unique(result['objects'])
    #     with open('objects.pkl', 'wb') as f:
    #         pickle.dump(objects, f)
    all_senti = result['all_sentis'].most_common()
    all_sentis[i] = [w for w in all_senti]
    sentis[i] = {k: v.most_common() for k, v in result['sentis'].items()}
    sentiment_detector[i] = {k: dict(v) for k, v in result['sentiment_detector'].items()}