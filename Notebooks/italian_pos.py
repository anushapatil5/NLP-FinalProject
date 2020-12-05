# Rules for Italian:
# past participle verbs vs. adjectives share the same suffix (-to):
# verbs ending in -to e.g. andato (gone), considerato (considered)
# adjectives ending in -to e.g. delicato (delicate), rigato (striped)

# TODO
# adjectives vs. adverbs share the same root:
# adverb is usually denoted by the suffix -mente e.g. adverb: velocemente (quickly) vs. adjective: veloce (quick)

import spacy_udpipe
import time
from data_generation import generate_clean_text
import json
import argparse

def get_doc(language="it", size=1):
    PATH = "/Users/francescaguiso/Desktop/" + language # use it.zip
    clean_texts = generate_clean_text(PATH)
    spacy_udpipe.download(language)
    nlp = spacy_udpipe.load(language)
    text_length = len(clean_texts)
    hundredth = text_length // 100
    start_time = time.time()
    for i in range(size):
        print(i * hundredth, (1+i) * hundredth)
    doc = nlp(clean_texts[:100000])
    print("--- %s seconds ---" % (time.time() - start_time))
    return doc

# Keep verbs ending in -to, -ta, -te, -ti
def filter_it_verbs(text):
    if text[-2:] == "to" or text[-2:] == "ti" or text[-2:] == "ta" or text[-2:] == "te":
        return True
    return False

# Keep adjectives ending in -to, -ta, -te, -ti
def filter_it_adj(text):
    if text[-2:] == "to" or text[-2:] == "ti" or text[-2:] == "ta" or text[-2:] == "te":
        return True
    return False

def get_italian_features(doc):
    feats_count_dict = {
        'VERB': 0,
        'ADJ': 0
    }

    feats_dict = {
        'VERB': [],
        'ADJ': []
    }

    for token in doc:
        if token.pos_ == 'VERB':
            feats_count_dict['VERB'] += 1
            if filter_it_verbs(token.text):
                if token.text not in feats_dict['VERB']:
                    feats_dict['VERB'].append(token.text)
            print(token.text, token.lemma_, token.pos_, token.dep_)
        if token.pos_ == 'ADJ':
            feats_count_dict['ADJ'] += 1
            if filter_it_adj(token.text):
                if token.text not in feats_dict['ADJ']:
                    feats_dict['ADJ'].append(token.text)
            print(token.text, token.lemma_, token.pos_, token.dep_)
    return feats_dict, feats_count_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("language", help="Enter the language en, ar, it, or hi")
    parser.add_argument("size", help="Enter the size of loop", type=int)
    args = parser.parse_args()
    doc = get_doc(args.language, args.size)
    feats_dict, feats_count_dict = get_italian_features(doc)
    print("DONE")
    file1 = open("output"+str(args.size)+".txt", "w")
    L = [""]
    file1.writelines(L)
    file1.close()

    for feature in feats_dict.keys():
        with open("output"+str(args.size)+".txt", "a") as text_file:
            for instance in set(feats_dict[feature]):
                print("{} {} {}".format(instance, feature, 10), file=text_file)

if __name__ == '__main__':
    main()
