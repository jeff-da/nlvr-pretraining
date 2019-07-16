# Generates prediction examples to "example_examples.jsonl" from the dev set.
import os
import random
import json

PATH = "/home/jzda/storage/ice/pretrain/features/pretrain/dev"
OUT = "example_examples.jsonl"
NUMBER_TO_GENERATE = 20

outfile = open(OUT, "w")
count = 0

data = []
all_phrases = []

for file in os.listdir(PATH):
    if random.random() < 0.1 and count < NUMBER_TO_GENERATE:
        count = count + 1
        file_data = json.load(open(os.path.join(PATH, file)))
        data.append(file_data)
        all_phrases.append(file_data['phrase'])

for item in data:
    for phrase in all_phrases:
        item['phrase'] = phrase
        outfile.write(json.dumps(item) + "\n")
