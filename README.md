# Mini-GPT

This project implements a Mini-GPT model with approximately 10 million parameters, trained on the entirety of Shakespeare's works. The model is designed to generate Shakespearean-style text by predicting the next token in a sequence based on context. The code is written in PyTorch and utilizes self-attention, transformer blocks, and multi-head attention mechanisms.

## Table of Contents
- [Overview](#overview)
- [Model Architecture](#model-architecture)
- [Training Details](#training-details)
- [Installation](#installation)
- [How to Use](#how-to-use)
  - [Training](#training)
  - [Generating Text](#generating-text)
- [Results](#results)
- [Credits](#credits)

## Overview

This repository contains an implementation of a character-level Mini-GPT language model trained on the complete works of Shakespeare. The model is built using PyTorch and employs the Transformer architecture, including self-attention, multi-head attention, and LayerNorm for language modeling tasks. The model is capable of generating Shakespeare-like text based on an initial context.

Key components include:
- **Self-attention**: Computes dependencies between all tokens.
- **Multi-head attention**: Allows the model to focus on different parts of the input.
- **Transformer blocks**: Sequential layers to process input sequences.

The model is trained to minimize cross-entropy loss between predicted and actual characters.

## Model Architecture

The model consists of:
- **Token and positional embeddings**: Tokens are mapped to vectors of size `n_embd`, and positional embeddings are added to capture the order of tokens.
- **Transformer blocks**: Each block contains multi-head self-attention followed by a feedforward layer. Layer normalization is applied at each stage.
- **Final Layer Norm**: The output is normalized before making predictions using a linear layer.

### Hyperparameters
- `batch_size`: 64
- `block_size`: 256 (maximum context length)
- `n_embd`: 384 (embedding dimension)
- `n_head`: 6 (number of attention heads)
- `n_layer`: 6 (number of transformer layers)
- `dropout`: 0.2 (dropout rate)
- `learning_rate`: 3e-4
- `max_iters`: 5000 (maximum iterations for training)

## Training Details

The model is trained on a corpus of Shakespeare's works, which is loaded from `input.txt`. The dataset is tokenized at the character level and split into training and validation sets. The training process is designed to minimize cross-entropy loss using the AdamW optimizer.

### Loss Evaluation
The model is evaluated every `eval_interval` iterations to compute training and validation losses. The loss is averaged over a set of evaluation iterations.

### Data Splits
- 90% of the dataset is used for training, and the remaining 10% is used for validation.

### Logging
The script logs the start and end times of training, allowing for easy tracking of how long it took to complete.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Sudhan-Dahake/Mini-GPT.git
   ```

2. **Install Dependencies**: Ensure you have Python 3.7 or higher installed. Then, install the required Python packages using pip.
   ```bash
   pip install torch
   ```

## How to Use

### Training

To train the model from scratch using the provided dataset:

1. Ensure that you have the Shakespeare dataset saved as `input.txt` in the root directory.
2. Run the training script:
   ```bash
   python train.py
   ```
3. The script will print the starting time, log training and validation loss at regular intervals, and generate sample Shakespearean text during training.
4. For reference, the training took 2.5 hrs on my Nvidia GTX 1650. This time could be more or less depending on your GPU.

### Generating Text

Once training is complete, you can generate text using the trained model by running the script. By default, it generates 10,000 new tokens and writes the output to a file `more.txt`.

## Results

After training, the model will generate Shakespearean-style text. Please refer to `more.txt` for the full generated text.

## Credits

Inspired from Andrej Karpathy
