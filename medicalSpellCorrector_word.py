import scipy
import numpy as np
import time
from nltk import ngrams
import json
import pickle
import argparse
from Levenshtein import ratio

ap = argparse.ArgumentParser()
ap.add_argument("-w", "--word", required = True, help="Input word")
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

def spellchecker_word(word):
    start_time = time.time()
    with open(args['vectorizerPath'],"rb") as fp:
        vectorizer = pickle.load(fp)
    with open(args['binderPath'],'r') as fp:
        binder = json.load(fp)
    
    word = word.lower().strip()
    word_gram_text = []
    ngram = []
    if word in binder['stop']:
        best_term = word
    else:
        ngram = ngram_generator(word)
        word_gram_text.append(' '.join(ngram))

        word_vector = vectorizer.transform(word_gram_text)
        word_vector = (np.asarray(word_vector.todense()))[0]

        searching_terms = []
        for gram in word_gram_text[0].split(' '):
            if gram in binder['ngram_lookup']:
                for term in binder['ngram_lookup'][gram]:
                    if binder['spell_corpus'][term][0] == word[0] and binder['spell_corpus'][term] not in searching_terms:
                        searching_terms.append(binder['spell_corpus'][term])

        sim_dict = {}
        for ngram_term in searching_terms:
            sim = 1-scipy.spatial.distance.cosine(np.asarray(vectorizer.transform([' '.join(ngram_generator(word))]).todense())[0].reshape(-1,1), word_vector.reshape(-1,1))
            lev_ratio = ratio(ngram_term,word)
            sim_dict[ngram_term] = (sim + lev_ratio)/2

        final_dict = {k: v for k, v in sorted(sim_dict.items(), key=lambda item: item[1],reverse = True)}
        best_term = max(final_dict, key=final_dict.get)
        print("INPUT WORD     :: {}\nCORRECTED WORD :: {}\nPROCESS TIME   :: {}".format(word,best_term,time.time() - start_time))
    return best_term

spellchecker_word(args['word'])