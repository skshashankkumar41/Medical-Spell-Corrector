import numpy as np
import json
import os 
import argparse
import errno
import pickle 
from nltk import ngrams
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

ap = argparse.ArgumentParser()
ap.add_argument("-sp", "--specialistJson", default = "data/specialists.json",help="path of specialist data")
ap.add_argument("-b", "--bodyPartsJson", default='data/body_parts.json',help="path of body parts data")
ap.add_argument("-ns", "--nhsSymptomJson", default='data/nhs_symptoms.json',help="path of nhs syptoms data")
ap.add_argument("-ws", "--wikiSymptomJson", default='data/wiki_symptoms.json',help="path of wiki data")
ap.add_argument("-mc", "--mostCommomJson", default='data/most_common.json',help="path of most common data")
ap.add_argument("-bp", "--binderPath", default='model/binder.json', help="path to store bidner json")
ap.add_argument("-vp", "--vectorizerPath", default='model/vectorizer.pickle', help="path to store vectorizer pickle")
args = vars(ap.parse_args())

def extract_json():
    print("[INFO] extrating JSON files...")
    with open(args['specialistJson'], 'r') as fp:
        specialists = json.load(fp)
    specialists = specialists[0]['specialists']
    
    with open(args['bodyPartsJson'], 'r') as fp:
        body_parts = json.load(fp)
    body_parts = body_parts[0]['body_parts']   

    with open(args['nhsSymptomJson'], 'r') as fp:
        nhs_symptoms = json.load(fp)
    nhs_symptoms = nhs_symptoms[0]['nhs_symptoms']  

    with open(args['wikiSymptomJson'], 'r') as fp:
        wiki_symptoms = json.load(fp)
    wiki_symptoms = wiki_symptoms[0]['wiki_symptoms']   

    with open(args['mostCommomJson'], 'r') as fp:
        most_common = json.load(fp)
    most_common = most_common[0]['most_common']    

    return [specialists, body_parts, nhs_symptoms, wiki_symptoms, most_common]   

def ngram_generator(text):
    ngram = []
    for gram in ngrams(text,2):
        ngram.append(''.join(gram))
    for gram in ngrams(text,3):
        ngram.append(''.join(gram))
        
    return ngram

def train():
    print("[INFO] training spellcorrector started...")
    spell_corpus = []
    train_data = extract_json()
    
    for category in train_data:
        for term in category:
            for single_term in term.split(" "):
                if single_term not in spell_corpus:
                    spell_corpus.append(single_term)

    ngram_text = []
    ngram_word_mapping = {}
    all_ngram = []

    print("[INFO] creating ngrams...")
    for text in spell_corpus:
        ngram = ngram_generator(text)
        for gram in ngram:
            if gram not in all_ngram: all_ngram.append(gram)
        
        ngram_text.append(' '.join(ngram))
        ngram_word_mapping[text] = ngram

    ngram_lookup = {}
    
    print("[INFO] creating lookup dictionaries...")
    for all_gram in all_ngram:
        for term in ngram_word_mapping:
            if all_gram in ngram_word_mapping[term]:
                if all_gram in ngram_lookup:
                    ngram_lookup[all_gram].append(spell_corpus.index(term))
                else:
                    ngram_lookup[all_gram]= [spell_corpus.index(term)]

    print("[INFO] creating vectorizer...")
    vectorizer = CountVectorizer()
    vectorizer.fit(ngram_text)

    words = stopwords.words('english')
    binder = {}
    binder['ngram_lookup'] = ngram_lookup
    binder['spell_corpus'] = spell_corpus
    binder['stop'] = words

    if not os.path.exists(os.path.dirname(args['binderPath'])):
        try:
            os.makedirs(os.path.dirname(args['binderPath']))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    print("[INFO] storing vectorizer and binder...")
    with open(args['binderPath'],'w') as fp:
        json.dump(binder,fp)

    with open(args['vectorizerPath'],"wb") as fp:
        pickle.dump(vectorizer,fp)

    print("[INFO] DONE...")
    return True

train()
