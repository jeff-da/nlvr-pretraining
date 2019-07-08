{

  "train_data_path": "/home/jzda/storage/ice/pretrain/features/pretrain_tiny",
  "validation_data_path": "/home/jzda/storage/ice/pretrain/features/pretrain_tiny",

  "dataset_reader": {
    "type": "vg_reader",
    "token_indexers": {
        "bert": {
            "type": "bert-pretrained",
            "pretrained_model": "bert-large-uncased",
            "do_lowercase": false,
            "use_starting_offsets": true
        },
        "token_characters": {
          "type": "characters",
          "min_padding_length": 3
        }
    }
  },

  "model": {
    "type": "vg_classifier",
    "text_field_embedder": {
        "allow_unmatched_keys": true,
        "embedder_to_indexer_map": {
            "bert": ["bert", "bert-offsets"],
            "token_characters": ["token_characters"],
        },
        "token_embedders": {
            "bert": {
                "type": "bert-pretrained",
                "pretrained_model": "bert-base-uncased"
            },
            "token_characters": {
                "type": "character_encoding",
                "embedding": {
                    "embedding_dim": 16
                },
                "encoder": {
                    "type": "cnn",
                    "embedding_dim": 16,
                    "num_filters": 128,
                    "ngram_filter_sizes": [3],
                    "conv_layer_activation": "relu"
                }
            }
        }
    },
    "abstract_encoder": {
      "type": "lstm",
      "bidirectional": true,
      "input_size": 896,
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
    "batch_size": 1
  },

  "trainer": {
    "num_epochs": 15,
    "cuda_device": 0,
    "grad_clipping": 5.0,
    "validation_metric": "+accuracy",
    "optimizer": {
      "type": "adagrad"
    }
  }
}
