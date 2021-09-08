# Efficient Contrastive Learning via Novel Data Augmentation and Curriculum Learning (EMNLP 2021)

The official code for our paper: [Efficient Contrastive Learning via Novel Data Augmentation and Curriculum Learning] (Accepted at EMNLP 2021 short paper)
The implementation is based on the paper [DeCLUTR: Deep Contrastive Learning for Unsupervised Textual Representations](https://arxiv.org/abs/2006.03659) and code implementation at (https://github.com/JohnGiorgi/DeCLUTR).

## Table of contents

- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluating with SentEval](#evaluating-with-senteval)
- [Citing](#citing)

## Installation

This repository requires Python 3.6.1 or later.

### Setting up a virtual environment

Before installing, you should create and activate a Python virtual environment. See [here](https://github.com/allenai/allennlp#installing-via-pip) for detailed instructions.

### Installing the library and dependencies

```bash
git clone https://github.com/vano1205/EfficientCL
cd EfficientCL
pip install -r requirements.txt
```

#### Gotchas

- If you plan on training your own model, you should also install [PyTorch](https://pytorch.org/) with [CUDA](https://developer.nvidia.com/cuda-zone) support by following the instructions for your system [here](https://pytorch.org/get-started/locally/).

## Usage

### Preparing a dataset

A dataset is simply a file containing one item of text (a document, a scientific paper, etc.) per line. For demonstration purposes, we have provided a script that will download the [WikiText-103](https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/) dataset and match our minimal preprocessing

```bash
python scripts/preprocess_wikitext_103.py path/to/output/wikitext-103/train.txt --min-length 2048
```

> See [scripts/preprocess_openwebtext.py](scripts/preprocess_openwebtext.py) for a script that can be used to recreate the (much larger) dataset used in our paper.

You can specify the train set path in the [configs](training_config) under `"train_data_path"`.

### Training

To train the model, use the [`allennlp train`](https://docs.allennlp.org/master/api/commands/train/) command with our [`efficientcl.jsonnet`](training_config/efficientcl.jsonnet) config. Run the following

```bash
allennlp train "training_config/efficientcl.jsonnet" \
    --serialization-dir "output" \
    --overrides "{'train_data_path': 'path/to/your/dataset/train.txt'}" \
    --include-package "efficientcl"
```

The `--overrides` flag allows you to override any field in the config with a JSON-formatted string, but you can equivalently update the config itself if you prefer. During training, models, vocabulary, configuration, and log files will be saved to the directory provided by `--serialization-dir`. This can be changed to any directory you like. 

#### Multi-GPU training

To train on more than one GPU, provide a list of CUDA devices in your call to `allennlp train`. For example, to train with four CUDA devices with IDs `0, 1, 2, 3`

```bash
--overrides "{'distributed.cuda_devices': [0, 1, 2, 3]}"
```

#### Training with mixed-precision

If your GPU supports it, [mixed-precision](https://devblogs.nvidia.com/mixed-precision-training-deep-neural-networks/) will be used automatically during training and inference.


### Evaluating with SentEval

[SentEval](https://github.com/facebookresearch/SentEval) is a library for evaluating the quality of sentence embeddings. We provide a script to evaluate our model against SentEval.

First, clone the SentEval repository and download the transfer task datasets (you only need to do this once)

```bash
git clone https://github.com/facebookresearch/SentEval.git
cd SentEval/data/downstream/
./get_transfer_data.bash
cd ../../../
```

> See the [SentEval](https://github.com/facebookresearch/SentEval) repository for full details.

Then you can run our [script](scripts/run_senteval.py) to evaluate a trained model against SentEval

```bash
python scripts/run_senteval.py allennlp "SentEval" "output"
 --output-filepath "output/senteval_results.json" \
 --cuda-device 0  \
 --include-package "declutr"
```

The results will be saved to `"output/senteval_results.json"`. This can be changed to any path you like.

> Pass the flag `--prototyping-config` to get a proxy of the results while dramatically reducing computation time.

For a list of commands, run

```bash
python scripts/run_senteval.py --help
```

For help with a specific command, e.g. `allennlp`, run

```
python scripts/run_senteval.py allennlp --help
```

#### Gotchas

- Evaluating the `"SNLI"` task of SentEval will fail without [this fix](https://github.com/facebookresearch/SentEval/pull/52).

<!-- ## Citing

If you use DeCLUTR in your work, please consider citing our preprint

```
@article{Giorgi2020DeCLUTRDC,
  title={DeCLUTR: Deep Contrastive Learning for Unsupervised Textual Representations},
  author={John M Giorgi and Osvald Nitski and Gary D. Bader and Bo Wang},
  journal={ArXiv},
  year={2020},
  volume={abs/2006.03659}
}
``` -->
