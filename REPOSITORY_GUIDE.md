# Repository Guide

### Welcome to **Transformers From First Principles**.

This guide explains the purpose of every folder and file in the repository so that you can easily navigate through both the documentation and the NumPy implementation.

---

# 📂 Repository Structure

```text
transformers-from-first-principles/

├── docs/
├── pics/
├── src/
│
├── README.md
├── REPOSITORY_GUIDE.md
├── ROADMAP.md
├── CONTRIBUTION.md
├── requirements.txt
└── .gitignore
```

---

# Documentation ( `docs/` )

The documentation is organized in a learning sequence.

It starts with basic concepts and gradually builds towards the complete Transformer architecture.

---

## 00_Introduction.md

Introduces the repository, learning objectives, prerequisites, and the overall roadmap.

---

## 01_Why_Transformers.md

Explains why Transformers replaced RNNs and LSTMs and introduces the motivation behind the architecture.

---

## 02_Matrix_mul.md

Covers the matrix operations used throughout the repository, including matrix multiplication and shape intuition.

---

## 03_Embeddings.md

Introduces word embeddings, embedding matrices, vocabulary lookup, and why embeddings are necessary.

---

## 04_positional_encoding.md

Explains positional encoding, sinusoidal encoding, and how positional information is injected into embeddings.

---

## 05_Attention.md

Introduces the intuition behind Attention and explains the concepts of Query, Key, and Value.

---

## 06_Scaled_Dot_Product_Attention.md

Explains the mathematical formulation of Scaled Dot-Product Attention with numerical examples.

---

## 07_Multi_Head_Attention.md

Shows how multiple attention heads work in parallel and why they improve representation learning.

---

## 08_Residual_Connections_&_LayerNorm.md

Explains Residual Connections and Layer Normalization, along with their importance for stable training.

---

## 09_Feed_Forward_Network.md

Describes the position and role of the Feed Forward Network inside every Transformer block.

---

## 10_Encoder.md

Builds the complete Encoder by assembling Multi-Head Attention, LayerNorm, Residual Connections, and FFN.

---

## 11_decoder.md

Explains the Decoder architecture, Masked Attention, Cross Attention, and Decoder Stack.

---

## 12_Full_Transformer.md

Connects the Encoder and Decoder into the complete Transformer architecture.

---

## 13_Training.md

Provides a high-level overview of forward propagation, loss computation, backpropagation, and parameter updates.

---

## 14_GPT_vs_BERT_vs_Transformer.md

Compares Encoder-only, Decoder-only, and Encoder–Decoder architectures and discusses their common applications.

---

# Images ( `pics/` )

Contains all diagrams, illustrations, and visual assets used throughout the documentation.

These images are referenced by the markdown files inside the `docs/` directory.

---

# Source Code (`src/`)

The source code implements every major component of the Transformer using only NumPy.

Each module is designed to closely follow the mathematical equations discussed in the documentation.

---

# Layers

Basic neural network building blocks used throughout the Transformer.

### linear.py

Implements a fully connected (dense) layer.

Used for:

- Query Projection
- Key Projection
- Value Projection
- Feed Forward Networks
- Output Projection

---

### ReLU.py

Implements the Rectified Linear Unit activation function.

Used inside the Feed Forward Network.

---

### softmax.py

Implements the Softmax function used to convert scores into probability distributions.

---

### Embedding.py

Implements token embeddings using an embedding matrix.

---

### positional_encoding.py

Generates sinusoidal positional encodings.

---

### LayerNorm.py

Implements Layer Normalization for stable training.

---

# Attention

Contains all attention mechanisms used by the Transformer.

---

### Scaled_Dot_Product.py

Implements the mathematical core of Attention.

---

### Self_attention.py

Implements Self-Attention where Query, Key, and Value originate from the same input.

---

### multi_head_attention.py

Runs multiple Self-Attention heads in parallel and concatenates their outputs.

---

### masked_attention.py

Implements causal masking to prevent future token leakage during decoding.

---

### cross_attention.py

Implements Encoder–Decoder Attention where the Decoder attends to the Encoder output.

---

# Encoder

Contains the complete Encoder implementation.

---

### encoder_block.py

Implements a single Encoder Block consisting of:

- Multi-Head Attention
- Residual Connection
- Layer Normalization
- Feed Forward Network

---

### encoder.py

Stacks multiple Encoder Blocks to form the complete Encoder.

---

# Decoder

Contains the complete Decoder implementation.

---

### decoder_block.py

Implements a single Decoder Block containing:

- Masked Multi-Head Attention
- Cross Attention
- Feed Forward Network
- Residual Connections
- Layer Normalization

---

### decoder.py

Stacks multiple Decoder Blocks to form the complete Decoder.

---

# transformer.py

Combines the Encoder and Decoder into the complete Transformer architecture.

---

# Suggested Learning Order

If you're new to Transformers, follow this order:

```
Documentation

↓

Layer Implementations

↓

Attention Modules

↓

Encoder

↓

Decoder

↓

Complete Transformer
```

Following this sequence will help you understand both the theory and the implementation together.

---

# Contributing

If you'd like to improve the repository, fix bugs, or enhance the documentation, please read [**CONTRIBUTION.md**](CONTRIBUTION.md) before submitting a Pull Request.

---

**Happy Learning!** 