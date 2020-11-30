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


def filter_ar_plural(text):
    if text[-2:] == "ات" or text[-2:] == "ون" or text[-2:] == "ين":
        return False
    return True


def filter_ar_singular(text):
    # text[-2:] == "ات" or text[-2:] == "ون" or text[:2] == "ال"
    if text[-1] == "ة":
        return False
    return True


def filter_ar_verbs(text):
    if text[-2:] == "وا" or text[-2:] == "ون" or text[-1] == "ت" or len(text) < 2:
        return False
    return True


def get_arabic_features(doc):
    verb_count, noun_count, plural_count, singular_count = 0, 0, 0, 0
    nouns, verbs, plural, singular = [], [], [], []
    for token in doc:
        if token.pos_ == 'VERB':
            verb_count += 1
            if filter_ar_verbs(token.text):
                verbs.append(token.text)
            print(token.text, token.lemma_, token.pos_, token.dep_)
        if token.pos_ == 'NOUN':
            noun_count += 1
            if token.tag_[:8] == 'N------P':
                plural_count += 1
                print(token.text)
                if filter_ar_plural(token.text):
                    plural.append(token.text)
            if token.tag_[:8] == 'N------S':
                singular_count += 1
                print(token.text)
                if filter_ar_singular(token.text):
                    singular.append(token.text)
            nouns.append(token.text)
            print(token.text, token.lemma_, token.pos_, token.dep_)
    return nouns, verbs, plural, singular

#with open('data.txt') as json_file:
#    data = json.load(json_file)


# start_time = time.time()
# nlp(clean_texts[:1000000])
# print("--- %s seconds ---" % (time.time() - start_time))
# 184 second
# ValueError: [E088] Text of length 8252584 exceeds maximum of 1000000. The v2.x parser and NER models require roughly 1GB of temporary memory per 100,000 characters in the input. This means long texts may cause memory allocation errors. If you're not using the parser or NER, it's probably safe to increase the `nlp.max_length` limit. The limit is in number of characters, so you can check whether your inputs are too long by checking `len(text)`.


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("language", help="Enter the language en, ar, it, or hi")
    parser.add_argument("size", help="Enter the size of loop", type=int)
    args = parser.parse_args()
    doc = get_doc(args.language, args.size)
    get_arabic_features(doc)
    print("DONE")


if __name__ == '__main__':
    main()
