import os
import ast

phrases = set()

for line in open('visual_genome_parse_v3.tsv'):
    try:
        region_id, sentence, phrase_list, image_id, x, y, width, height = line.rstrip().split('\t')
        for i, phrase in enumerate(ast.literal_eval(phrase_list)):
            phrases.add(phrase)
    except:
        pass

outfile = open('phrase_vocab.txt', 'w')
for phrase in phrases:
    outfile.write(phrase + "\n")