import scipy
import time
import json
import pickle
import argparse
import numpy as np
from Levenshtein import ratio
from nltk import ngrams

ap = argparse.ArgumentParser()
ap.add_argument("-w", "--text", required = True, help="Input text")
ap.add_argument("-bp", "--binderPath", default='model/binder.json', help="path to store bidner json")
ap.add_argument("-vp", "--vectorizerPath", default='model/vectorizer.pickle', help="path to store vectorizer pickle")
args = vars(ap.parse_args())

def ngram_generator(text):
    ngram = []
    for gram in ngrams(text,2):
        ngram.append(''.join(gram))
    for gram in ngrams(text,3):
        ngram.append(''.join(gram))
        
    return ngram

def spellchecker_text(string):
    start_time = time.time()
    with open(args['vectorizerPath'],"rb") as fp:
        vectorizer = pickle.load(fp)
    with open(args['binderPath'],'r') as fp:
        binder = json.load(fp)
    
    corrected_string = []
    for text in string.split(" "):
        text = text.lower().strip()
        test_gram_text = []
        ngram = []
        if text in binder['stop']:
            best_term = text
        else:
            ngram = ngram_generator(text)
            test_gram_text.append(' '.join(ngram))

            test_vector = vectorizer.transform(test_gram_text)
            test_vector = (np.asarray(test_vector.todense()))[0]

            searching_terms = []
            for gram in test_gram_text[0].split(' '):
                if gram in binder['ngram_lookup']:
                    for term in binder['ngram_lookup'][gram]:
                        if binder['spell_corpus'][term][0] == text[0] and binder['spell_corpus'][term] not in searching_terms:
                            searching_terms.append(binder['spell_corpus'][term])

            sim_dict = {}
            for ngram_term in searching_terms:
                sim = 1-scipy.spatial.distance.cosine(np.asarray(vectorizer.transform([' '.join(ngram_generator(text))]).todense())[0].reshape(-1,1), test_vector.reshape(-1,1))
                lev_ratio = ratio(ngram_term,text)
                sim_dict[ngram_term] = (sim + lev_ratio)/2
            try:
                final_dict = {k: v for k, v in sorted(sim_dict.items(), key=lambda item: item[1],reverse = True)}
                best_term = max(final_dict, key=final_dict.get)
            except:
                best_term = text
        corrected_string.append(best_term)
        
    print("INPUT TEXT     :: {}\nCORRECTED TEXT :: {}\nPROCESS TIME   :: {}".format(string,' '.join(corrected_string),time.time() - start_time))
    return ' '.join(corrected_string)

spellchecker_text(args['text'])