import json
import tqdm
import nltk
import os
from collections import Counter, defaultdict
import pickle

corpus_type = 'part'
senti_corpus = json.load(open("/home/wyf/open_source_dataset/for_debug/artemis_new_20.json"))
corpus_dir = "/home/wyf/open_source_dataset/for_debug/"
tmp_senti_corpus = defaultdict(list)
tmp_senti_corpus_pos = defaultdict(list)
all_sentis = Counter()
sentis = defaultdict(Counter)
sentiment_detector = defaultdict(Counter)
sent_num = len(senti_corpus)
for lines in senti_corpus:
    for sent in lines['utterance_result']:
        senti_label = sent['emotion']
        tmp_sents = []
        for i in range(0, sent_num):
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
            tmp_senti_corpus[senti_label].append(words)
            tmp_senti_corpus_pos[senti_label].append(poses)
            if adjs:
                all_sentis.update(adjs)
                sentis[senti_label].update(adjs)
                for noun in nouns:
                    sentiment_detector[noun].update(adjs)

json.dump(tmp_senti_corpus, open(os.path.join(corpus_dir, 'for_debug_tmp_senti_corpus.json'), 'w'))
json.dump(tmp_senti_corpus_pos, open(os.path.join(corpus_dir, 'for_debug_tmp_senti_corpus_pos.json'), 'w'))
objects = {k for k, v in sentiment_detector.items()}
with open('objects.pkl', 'wb') as f:
    pickle.dump(objects, f)
all_sentis = all_sentis.most_common()
all_sentis = [w for w in all_sentis]
sentis = {k: v.most_common() for k, v in sentis.items()}
sentiment_detector = {k: v.most_common() for k, v in sentiment_detector.items()}
all_sentis = {k: v for k, v in all_sentis}
len_sentis = defaultdict(int)
for k, v in sentis.items():
    for _, n in v:
        len_sentis[k] += n
tf_sentis = defaultdict(dict)
tmp_sentis = defaultdict(dict)
for k, v in sentis.items():
    for w, n in v:
        tf_sentis[k][w] = n / len_sentis[k]
        tmp_sentis[k][w] = n
sentis = tmp_sentis
sentis_result = defaultdict(dict)
for k, v in tf_sentis.items():
    for w, tf in v.items():
        if w in all_sentis:
            sentis_result[k][w] = tf * (sentis[k][w] / all_sentis[w])
sentiment_words = {}
sentis_result_trible = defaultdict(dict)
negative_set = {'sadness', 'disgust', 'fear', 'anger'}
positive_set = {'amusement', 'awe', 'contentment', 'excitement'}
for k, v in sentis_result.items():
    if k in negative_set:
        sentis_result_trible['negative'].update(v)
    elif k in positive_set:
        sentis_result_trible['positive'].update(v)
    elif k == 'something else':
        sentis_result_trible['common'].update(v)
sentis_result_trible['positive'] = {'hopeful': 0.2, 'happy': 0.1, 'great': 0.5}
for k in sentis_result_trible:
    sentiment_words[k] = list(sentis_result_trible[k].items())
    sentiment_words[k].sort(key=lambda p: p[1], reverse=True)
    sentiment_words[k] = [w[0] for w in sentiment_words[k]]
print(sentiment_words)
common_rm = []
pos_rm = []
neg_rm = []
for i, w in enumerate(sentiment_words['positive']):
    if w in sentiment_words['negative']:
        n_idx = sentiment_words['negative'].index(w)
        if abs(i-n_idx) < 5:
            common_rm.append(w)
        elif i > n_idx:
            pos_rm.append(w)
for w in common_rm:
    sentiment_words['positive'].remove(w)
    sentiment_words['negative'].remove(w)
for w in pos_rm:
    sentiment_words['positive'].remove(w)
for w in neg_rm:
    sentiment_words['negative'].remove(w)
tmp_sentiment_words = {}
for senti in sentiment_words:
    tmp_sentiment_words[senti] = {}
    for w in sentiment_words[senti]:
        tmp_sentiment_words[senti][w] = sentis_result_trible[senti][w]
sentiment_words = tmp_sentiment_words

json.dump(sentiment_words, open(os.path.join(corpus_dir, 'for_debug_sentiment_words.json'), 'w'))
tmp_sentiment_words = {}
tmp_sentiment_words.update(sentiment_words['positive'])
tmp_sentiment_words.update(sentiment_words['negative'])
sentiment_words = tmp_sentiment_words
tmp_sentiment_detector = defaultdict(list)
for noun, senti_words in sentiment_detector.items():
    number = sum([w[1] for w in senti_words])
    for senti_word in senti_words:
        if senti_word[0] in sentiment_words:
            tmp_sentiment_detector[noun].append(
                (senti_word[0], senti_word[1] / number * sentiment_words[senti_word[0]]))
sentiment_detector = tmp_sentiment_detector
tmp_sentiment_detector = {}
for noun, senti_words in sentiment_detector.items():
    if len(senti_words) <= 50:
        tmp_sentiment_detector[noun] = senti_words
json.dump(tmp_sentiment_detector, open(os.path.join(corpus_dir, 'sentiment_detector.json'), 'w'))


