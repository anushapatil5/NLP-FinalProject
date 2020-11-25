
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

6- The results should be Folders starting from AA, AB, AC, ... etc. Each folder will have text files named wiki_00, wiki_01, ... etc.
7- Keep only the folders from AA-AE to keep things consistent because Hindi only have ~500MBs worth of data.
"""

import os
import json
PATH = "/Users/abdulrahimqaddoumi/opt/anaconda3/lib/python3.7/site-packages/wikiextractor/data"
files_path = []
for subdir, dir, files in os.walk(PATH):
    for file in files:
        if file[0] == "w":
            files_path.append(subdir + "/" + file)


texts = ""
for file in files_path[:3]:
    f = open(file, "r")
    texts += f.read()


# TODO Check spliting by (".") for Hindi
texts = texts.split("\n")
clean_texts = ""
for text in texts:
    if len(text) > 0 and text[:4] != "<doc" and text[:5] != "</doc":
        clean_texts += text


training_text = clean_texts[:int(len(clean_texts) * 0.8)]
remaining_text = clean_texts[int(len(clean_texts) * 0.8):]
test_text = remaining_text[:len(clean_texts ) // 2]
valid_text = remaining_text[len(clean_texts ) // 2:]

data = []
for sentence in training_text.split("."):
    data.append({"tokens": sentence.split(" ")})

with open('train.jsonl', 'w') as outfile:
    json.dump(data, outfile)
print("DONE")