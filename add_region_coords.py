import json
from tqdm import tqdm

vg_file = '/home/jzda/storage/visual_genome/region_old.json'

region_data = {}

for i, image in enumerate(json.load(open(vg_file))):
    for region in image['regions']:
        region_id = region['id']
        region_image_id = region['image']
        height = region['height']
        width = region['width']
        x = region['x']
        y = region['y']
        region_data[region_id] = (region_image_id, x, y, width, height)

vg_file = '/home/jzda/storage/visual_genome/region_descriptions.json'
for i, image in enumerate(json.load(open(vg_file))):
    for region in image['regions']:
        region_id = region['region_id']
        region_image_id = region['image_id']
        height = region['height']
        width = region['width']
        x = region['x']
        y = region['y']
        region_data[region_id] = (region_image_id, x, y, width, height)

tsv_file = '/home/jzda/storage/visual_genome/visual_genome_parse_v2.tsv'
outfile = open('./visual_genome_parse_v3.tsv', 'w')
for line in tqdm(open(tsv_file)):
    data = line.split('\t')
    try:
        region_id = int(data[0])
        region_image_id, x, y, width, height = region_data[region_id]
        result = '\t'.join([data[0], data[1], data[2], str(region_image_id), str(x), str(y), str(width), str(height)])
        outfile.write(result + "\n")
    except:
        opps = 1
