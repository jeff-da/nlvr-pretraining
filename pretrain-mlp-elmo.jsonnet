{

  "train_data_path": "/home/jzda/storage/ice/pretrain/features/pretrain_tiny",
  "validation_data_path": "/home/jzda/storage/ice/pretrain/features/pretrain_tiny",

  "dataset_reader": {
    "type": "vg_reader",
    "token_indexers": {
       "tokens": {
         "type": "single_id",
         "lowercase_tokens": true
       },
        "elmo": {
          "type": "elmo_characters"
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
         },
          "elmo": {
            "type": "elmo_token_embedder",
            "options_file": "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json",
            "weight_file": "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5",
            "do_layer_norm": false,
            "dropout": 0.5
          },
      }
    },
    "abstract_encoder": {
      "type": "lstm",
      "bidirectional": true,
      "input_size": 1324,
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
