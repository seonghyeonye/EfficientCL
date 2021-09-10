# Efficient Contrastive Learning via Novel Data Augmentation and Curriculum Learning (EMNLP 2021)

The official code for our paper: [Efficient Contrastive Learning via Novel Data Augmentation and Curriculum Learning] (Accepted at EMNLP 2021 short paper)

The implementation is based on the paper [DeCLUTR: Deep Contrastive Learning for Unsupervised Textual Representations](https://arxiv.org/abs/2006.03659) and code implementation at (https://github.com/JohnGiorgi/DeCLUTR).

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

### Exporting a trained model to HuggingFace Transformers

We have provided a simple script to export a trained model so that it can be loaded with [Hugging Face Transformers](https://github.com/huggingface/transformers)

```bash
python scripts/save_pretrained_hf.py "output" "pretrained"
```

### Evaluating with GLUE

[Jiant](https://github.com/nyu-mll/jiant) package is used for evaluation of GLUE benchmark. First, download the specific dataset (CoLA, MNLI, MRPC, QNLI, QQP, RTE, SST, STSB) by the following command:

```bash
cd jiant
export PYTHONPATH=/path/to/jiant:$PYTHONPATH
python jiant/scripts/download_data/runscript.py \
    download \
    --tasks mrpc \
    --output_path data
```

Evaluate each dataset by loading the exported model from above.

```bash
python jiant/proj/simple/runscript.py run --run_name simple --data_dir data --hf_pretrained_model_name_or_path ../pretrained --tasks mrpc --train_batch_size 16 --num_train_epochs 10  --exp_dir roberta 
```

