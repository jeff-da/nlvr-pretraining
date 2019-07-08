#!/usr/bin/env python

import base64
import numpy as np
import csv
import sys
import zlib
import time
import mmap
import os
import json
from tqdm import tqdm
csv.field_size_limit(sys.maxsize)
   
FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']
# infile = '/home/jzda/storage/test2014_36/test2014_resnet101_faster_rcnn_genome_36.tsv'
# infile = '/home/jzda/storage/trainval_36/trainval_resnet101_faster_rcnn_genome_36.tsv'
# infile = '/home/jzda/storage/test2015_36/test2015_resnet101_faster_rcnn_genome_36.tsv'
infile = '/media/drive2/jzda/ice/pretrain/features/max/jeff.regions.36'

outdir = './features/train/'

if __name__ == '__main__':

    # Verify we can read a tsv
    in_data = {}
    with open(infile, "r") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = FIELDNAMES)
        for item in tqdm(reader):
            item['image_id'] = int(item['image_id'])
            if os.path.exists(os.path.join(outdir, str(item['image_id']) + ".json")):
                continue
            item['image_h'] = int(item['image_h'])
            item['image_w'] = int(item['image_w'])   
            item['num_boxes'] = int(item['num_boxes'])
            for field in ['boxes', 'features']:
                item[field] = str.encode(item[field])
                item[field] = np.frombuffer(base64.decodestring(item[field]), 
                      dtype=np.float32).reshape((item['num_boxes'],-1))
                item[field] = str(item[field].tolist())
            in_data[item['image_id']] = item
            outfile = open(os.path.join(outdir, str(item['image_id']) + ".json"), 'w')
            outfile.write(json.dumps(item))