{

  "train_data_path": "/home/jzda/storage/ice/pretrain/features/pretrain/train",
  "validation_data_path": "/home/jzda/storage/ice/pretrain/features/pretrain/dev",

  "dataset_reader": {
    "type": "vg_reader",
    "token_indexers": {
       "tokens": {
         "type": "single_id",
         "lowercase_tokens": true
       }
    }
  },

  "model": {
    "type": "vg_classifier",
    "text_field_embedder": {
      "token_embedders": {
         "tokens": {
            "type": "embedding",
            "pretrained_file": "https://allennlp.s3.amazonaws.com/datasets/glove/glove.6B.300d.txt.gz",
            "embedding_dim": 300,
            "trainable": false
         }
      }
    },
    "abstract_encoder": {
      "type": "lstm",
      "bidirectional": true,
      "input_size": 300,
      "hidden_size": 300,
      "num_layers": 2,
      "dropout": 0.2
    },
    "classifier_feedforward": {
      "input_dim": 2648,
      "num_layers": 2,
      "hidden_dims": [512, 2],
      "activations": ["relu", "linear"],
      "dropout": [0.2, 0.0]
    }
  },

  "iterator": {
    "type": "basic",
    "batch_size": 32
  },

  "trainer": {
    "num_epochs": 10,
    "cuda_device": 0,
    "grad_clipping": 5.0,
    "validation_metric": "+accuracy",
    "optimizer": {
      "type": "adagrad"
    }
  }
}
