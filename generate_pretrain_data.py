import os
import ast
import json
from tqdm import tqdm
from collections import defaultdict
import random

def get_iou(vg_tuple, tuple2, both_absolute=False):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    tuple: x y width height
    """
    if not both_absolute:
        x, y, width, height = vg_tuple
        bb1 = {}
        bb1['x1'] = int(x)
        bb1['y1'] = int(y)
        bb1['x2'] = int(x) + int(width)
        bb1['y2'] = int(y) + int(height)
    else:
        x1, y1, x2, y2 = vg_tuple
        bb1 = {}
        bb1['x1'] = int(x1)
        bb1['y1'] = int(y1)
        bb1['x2'] = int(x2)
        bb1['y2'] = int(y2)
    x1, y1, x2, y2 = tuple2
    bb2 = {}
    bb2['x1'] = int(x1)
    bb2['y1'] = int(y1)
    bb2['x2'] = int(x2)
    bb2['y2'] = int(y2)
    if not bb1['x1'] < bb1['x2'] or not bb1['y1'] < bb1['y2'] or not bb2['x1'] < bb2['x2'] or not bb2['y1'] < bb2['y2']:
        return 0.0

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

data_dir = "./features/train/"
outdir = './features/pretrain'

stored_data = defaultdict(list)

vg_image_ids = []
for e, line in tqdm(enumerate(open("visual_genome_parse_v3.tsv", "r")), total=433793):
    try:
        region_id, sentence, phrase_list, image_id, x, y, width, height = line.rstrip().split('\t')
        if os.path.exists(os.path.join(data_dir, image_id + ".json")):
            data = json.load(open(os.path.join(data_dir, image_id + ".json")))
            # find lowest iou
            truth = (x, y, width, height)
            max_iou = -1
            max_box = None
            max_box_i = None
            for i, box in enumerate(ast.literal_eval(data['boxes'])):
                current_iou = get_iou(truth, box)
                if current_iou > max_iou:
                    max_iou = current_iou
                    max_box = box
                    max_box_i = i
            if max_iou > 0.4:
                feature_list = ast.literal_eval(data['features'])
                phrase_dicts = []
                for i, phrase in enumerate(ast.literal_eval(phrase_list)):
                    datum = {}
                    datum['region_id'] = region_id
                    datum['image_id'] = image_id
                    datum['box'] = max_box
                    datum['iou'] = max_iou
                    datum['phrase'] = phrase
                    datum['phrase_num'] = i
                    datum['features'] = feature_list[max_box_i]
                    phrase_dicts.append(datum)
                
                for datum in tqdm(phrase_dicts):
                    data_list = datum
                    # print positive
                    filename = str(datum['region_id']).zfill(8) + str(datum['phrase_num']).zfill(2) + '.json'
                    outfile = open(os.path.join(outdir, filename), 'w')
                    json.dump(datum, outfile)
                    outfile.write('\n')
                    outfile.close()
        vg_image_ids.append(int(image_id))
    except:
        pass


