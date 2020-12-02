import spacy_udpipe
import time
from data_generation import generate_clean_text
import json
import argparse


def get_doc(language="ar", size=1):
    PATH = "/Users/abdulrahimqaddoumi/Desktop/" + language
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


def filter_en_plural(text):
    return True


def filter_en_singular(text):
    return True


def filter_en_verbs(text):
    return True


def filter_en_s_end(text):
    pass


def filter_en_ing_end(text):
    pass


def get_english_features(doc):
    verb_count, noun_count, plural_count, singular_count, ing_count, s_count = 0, 0, 0, 0, 0, 0
    nouns, verbs, plural, singular, s_end, ing_end = [], [], [], [], [], []
    for token in doc:
        if token.text[-1] == "s":
            s_end.append(token.text)
            s_count += 1
        elif token.text[-3:] == "ing":
            ing_end.append(token.text)
            ing_count += 1
        if token.pos_ == 'VERB':
            verb_count += 1
            if filter_en_verbs(token.text):
                verbs.append(token.text)
        elif token.pos_ == 'NOUN':
            noun_count += 1
            if token.tag_ == 'NNS':
                plural_count += 1
                if filter_en_plural(token.text):
                    plural.append(token.text)
            if token.tag_ == 'NN':
                singular_count += 1
                if filter_en_singular(token.text):
                    singular.append(token.text)
            nouns.append(token.text)
    return nouns, verbs, plural, singular

#with open('data.txt') as json_file:
#    data = json.load(json_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("language", help="Enter the language en, ar, it, or hi")
    parser.add_argument("size", help="Enter the size of loop", type=int)
    args = parser.parse_args()
    doc = get_doc(args.language, args.size)
    get_english_features(doc)
    print("DONE")


if __name__ == '__main__':
    main()
