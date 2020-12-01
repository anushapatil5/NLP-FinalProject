"""
1- Download the WikiPedia Dump for the language:
English:
https://dumps.wikimedia.org/enwiki/20201020/enwiki-20201020-pages-articles-multistream.xml.bz2

Hindi:
https://dumps.wikimedia.org/hiwiki/20201020/hiwiki-20201020-pages-articles-multistream.xml.bz2

Arabic:
https://dumps.wikimedia.org/arwiki/20201020/arwiki-20201020-pages-articles-multistream.xml.bz2

Italian:
https://dumps.wikimedia.org/itwiki/20201020/itwiki-20201020-pages-articles-multistream.xml.bz2

2- Install WikiExtractor:
pip install wikiextractor

3- Go to the directory of WikiExtrator in your system:
cd /Users/abdulrahimqaddoumi/opt/anaconda3/lib/python3.7/site-packages/wikiextractor/

4- Run this command: data is the target folder and /Users/abdulrahimqaddoumi/... is the dump location.
python WikiExtractor.py -o data /Users/abdulrahimqaddoumi/Downloads/hiwiki-20201020-pages-articles-multistream.xml.bz2

5- The results should be Folders starting from AA, AB, AC, ... etc. Each folder will have text files named wiki_00, wiki_01, ... etc.

6- Keep only the folders from AA-AE to keep things consistent because Hindi only have ~500MBs worth of data.

Usage: python data_genration.py LANG
e.g. python data_genration.py en
"""

import sys
import os
import json
from tqdm import tqdm

def generate_clean_text(path):
    files_path = []
    for subdir, dir, files in os.walk(path):
        for file in files:
            if file[0] == "w":
                files_path.append(subdir + "/" + file)

    texts = ""
    for file in tqdm(files_path):
        f = open(file, "r")
        texts += f.read()

    # TODO Check spliting by (".") for Hindi
    texts = texts.split("\n")
    clean_texts = ""
    for text in tqdm(texts):
        if len(text) > 0 and text[:4] != "<doc" and text[:5] != "</doc":
            clean_texts += text
    return clean_texts


def main(LANGUAGE, PATH):
    # This first line takes the text from Wikipedia articles and split it by "\n". Then remove some tags <doc> and </doc>
    clean_texts = generate_clean_text(PATH)
    # This is just splitting the training, test and validation datasets and store them in a dictionary.
    training_text = clean_texts[:int(len(clean_texts) * 0.8)]
    remaining_text = clean_texts[int(len(clean_texts) * 0.8):]
    test_text = remaining_text[:len(remaining_text) // 2]
    valid_text = remaining_text[len(remaining_text) // 2:]
    all_text = {"train": training_text, "test": test_text, "valid": valid_text}

    # TODO: Change LANGUAGES to change name
    # This loops through the dictionary and split each paragraphs into sentences "." and then split the sentence into tokens " " and then dumps a json file
    for key, text in all_text.items():
        data = []
        for sentence in tqdm(text.split(".")):
            data.append({"tokens": sentence.split(" ")})
        name = LANGUAGE + "_" + key + '.jsonl'
        print(len(data))
        with open(name, 'w') as outfile:
            json.dump(data, outfile)


if __name__ == '__main__':
    LANGUAGE = sys.argv[1]
    PATH = "~/Desktop/"+str(LANGUAGE)
    main(LANGUAGE, PATH)
