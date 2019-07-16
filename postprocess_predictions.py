# After running `allennlp predict`, use this to create a file to use for bounding box investigation.
import json
from collections import defaultdict

PREDICTION_OUTPUT = 'pred.jsonl'
PREDICTION_INPUT = 'example_examples.jsonl'
outfile = open("predicted_data.jsonl", 'w')
batches = defaultdict(list)

for output, input in zip(open(PREDICTION_OUTPUT), open(PREDICTION_INPUT)):
    input = json.loads(input.rstrip())
    output = json.loads(output.rstrip())
    key = input['region_id']
    batches[key].append((input, output))

for key in batches:
    data_list = batches[key]
    result = data_list[0][0]
    phrases = []
    """
    max_logits = -10
    argmax_logits = 0
    for i, datum in enumerate(data_list):
        if datum[1]['logits'][0] > max_logits:
            argmax_logits = i
            max_logits = datum[1]['logits'][0]
    """

    if datum[1]['logits'][0] > datum[1]['logits'][1]:
        phrases.append(datum[0]['phrase'])
    del result['features']
    result['phrases'] = phrases
    outfile.write(json.dumps(hehe) + "\n")

