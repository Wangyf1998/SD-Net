import json
import tqdm
from collections import Counter, defaultdict
import sng_parser
import numpy as np
from multiprocessing import Pool
from nltk.corpus import stopwords
senti_corpus = json.load(open("/home/wyf/open_source_dataset/for_debug/data_10.json"))
new_words = ['painting', 'painter', 'picture', 'look', 'day', 'colours', 'sense', 'color', 'colour', 'colours',
             'colour', 'background', 'I', 'colouring', 'artwork', 'beginning', 'end', 'ending', 'begin', 'atmosphere',
             'size', 'scene', 'use', 'work']
stopwords = stopwords.words('english')
stopwords.extend(new_words)



def get_full_concept(senti_corpus, stop_words):
    concept = defaultdict(Counter)
    for i in tqdm.tqdm(range(0, len(senti_corpus), 100)):
        current_data = senti_corpus[i:i+100]
        for data in current_data:
            for records in data:
                id = records['painting']
                split = records['split']
                art_style = records['art_style']
                for record in records['utterance_result']:
                    caption = record['utterance_spelled']
                    emotion = record['emotion']
                    emotion_label = record['emotion_label']
                    graph = sng_parser.parse(caption)
                    object = defaultdict(list)
                    for entity in graph['entities']:
                        word = entity['lemma_head']
                        if word not in stop_words:
                            object[emotion_label].append(word)
                    for k, v in object.items():
                        concept[str(k)].update(v)
    return concept


concept = get_full_concept(senti_corpus, new_words)
for k, v in concept.items():
    v = [item[0] for item in v.most_common()]
    concept[k] = v[:250]

for k, v in concept.items():
    concept[k].most_common()
concept = get_full_concept(senti_corpus, stopwords)


