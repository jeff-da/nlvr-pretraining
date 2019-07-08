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
    def _read(self, file_path):
        phrases = set() # save the phrases for negative sampling

        for file in os.listdir(file_path):
            with open(os.path.join(file_path, file), "r") as data_file:
                vg_json = json.load(data_file)
                tokens = vg_json['phrase']
                label = 'True' # positive examples only
                metadata = {
                    'box': vg_json['box'],
                    'features': vg_json['features']
                }
                phrases.add(vg_json['phrase'])
                yield self.text_to_instance(tokens, metadata, label)
        
        # negative sampling
        for file in os.listdir(file_path):
            with open(os.path.join(file_path, file), "r") as data_file:
                vg_json = json.load(data_file)
                tokens = vg_json['phrase']
                while tokens is vg_json['phrase']:
                    tokens = random.sample(phrases, 1)[0]
                label = 'False'
                metadata = {
                    'box': vg_json['box'],
                    'features': vg_json['features']
                }
                yield self.text_to_instance(tokens, metadata, label)

    @overrides
    def text_to_instance(self, tokens: str, metadata: Dict[str, list], label: str = None) -> Instance:  # type: ignore
        tokenized_tokens = self._tokenizer.tokenize(tokens)
        tokens_field = TextField(tokenized_tokens, self._token_indexers)

        fields = {'tokens': tokens_field, 'metadata': MetadataField(metadata)}
        if label is not None:
            fields['label'] = LabelField(label)
        return Instance(fields)
