from typing import Dict
import json
import logging
import os
import random

from overrides import overrides

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

NLVR2_IMAGE_FEATURE_JSONS_DIR = "/home/jzda/storage/ice/pretrain/features/nlvr2/"

@DatasetReader.register("vg_reader")
class SemanticScholarDatasetReader(DatasetReader):
    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file):
        for line in open(file):
            data = json.loads(line.rstrip())
            # convert NLVR2 to Max id
            id = data['identifier']
            x = 1 if 'train' in id else 2
            yy = str(data['directory']) if 'directory' in data else '00'
            zzzzz = id.split('-')[1]
            w = id.split('-')[2]
            i = None

            id = str(x) + yy.zfill(2) + zzzzz.zfill(5) + w
            phrases = data['phrases']
            for left_or_right, image_file in enumerate([id + "0" + ".json", id + "1" + ".json"]): # 2 (left and right)
                image_regions_data = json.load(open(os.path.join(NLVR2_IMAGE_FEATURE_JSONS_DIR, image_file)))
                for phrase in phrases: # P
                    for i, box in enumerate(image_regions_data['boxes']): # R (36)
                        metadata = {
                            'box': box,
                            'features': image_regions_data['features'][i],
                            'id': data['identifier'],
                            'region_id': i,
                            'phrase': phrase,
                            'image': 'Left' if left_or_right is 0 else 'Right'
                        }
                        tokens = phrase
                        yield self.text_to_instance(tokens, metadata)

    @overrides
    def text_to_instance(self, tokens: str, metadata: Dict[str, list], label: str = None) -> Instance:  # type: ignore
        tokenized_tokens = self._tokenizer.tokenize(tokens)
        tokens_field = TextField(tokenized_tokens, self._token_indexers)

        fields = {'tokens': tokens_field, 'metadata': MetadataField(metadata)}
        if label is not None:
            fields['label'] = LabelField(label)
        return Instance(fields)
