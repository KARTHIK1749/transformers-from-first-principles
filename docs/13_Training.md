# Training a Transformer

**"A Transformer learns by comparing its predictions with the correct answers and gradually updating its parameters to minimize the error."**

---

# Learning Objectives

By the end of this chapter, you will be able to:

- Understand how a Transformer learns.
- Learn the role of the Loss Function.
- Understand Backpropagation.
- Learn how Optimizers update the model.
- Understand the complete training pipeline.

---

# How Does a Transformer Learn?

A Transformer initially makes random predictions.

For example,

```
Input

I love AI
```

Expected Output

```
J'aime l'IA
```

Initially, the model may predict

```
Bonjour AI
```

which is incorrect.

The difference between the prediction and the correct answer is measured using a **Loss Function**.

The model then updates its parameters to reduce this error.

---

# Step 1 — Forward Pass

The input sentence passes through

- Embedding Layer
- Positional Encoding
- Encoder
- Decoder
- Linear Layer
- Softmax

to generate probabilities for the next token.

Mathematically,

$$
P = Transformer(X)
$$

where

- $X$ is the input sequence.
- $P$ is the predicted probability distribution.

---

# Step 2 — Loss Function

The Loss Function measures how far the prediction is from the correct answer.

For language models,

the most common choice is **Cross-Entropy Loss**.

$$
L=-\sum_i y_i\log(\hat y_i)
$$

where

- $y$ is the true label.
- $\hat y$ is the predicted probability.

A lower loss indicates better predictions.

---

# Step 3 — Backpropagation

After computing the loss,

Backpropagation calculates how much each parameter contributed to the error.

Using the chain rule,

the gradients are computed for every learnable parameter.

$$
\frac{\partial L}{\partial W}
$$

These gradients indicate

- how much each weight should change
- in which direction it should move

---

# Step 4 — Optimizer

The optimizer updates the model parameters using the computed gradients.

A simple update rule is

$$
W=W-\eta\frac{\partial L}{\partial W}
$$

where

- $W$ is a learnable parameter.
- $\eta$ is the learning rate.

Although the original Transformer used the **Adam** optimizer, the update principle remains the same: adjust parameters to reduce the loss.

---

# Step 5 — Repeat

Training consists of repeating the same process over many batches and epochs.

```
Input

↓

Forward Pass

↓

Prediction

↓

Loss

↓

Backpropagation

↓

Optimizer

↓

Updated Parameters

↓

Repeat
```

Gradually,

the model becomes better at predicting the next token.

---

# Example

Suppose

```
Target

cat
```

Predicted probabilities

```
cat : 0.25

dog : 0.60

bird : 0.15
```

The prediction is incorrect,

resulting in a high loss.

After several training iterations,

the probabilities may become

```
cat : 0.95

dog : 0.03

bird : 0.02
```

The loss decreases,

indicating that the model has learned.

---

# Training Pipeline

The complete training process can be summarized as

```
Dataset

↓

Forward Pass

↓

Loss Function

↓

Backpropagation

↓

Optimizer

↓

Parameter Update

↓

Repeat for Many Epochs
```

This iterative process enables the Transformer to learn meaningful language representations.

---

# Key Takeaways

- Training begins with random predictions.
- The Loss Function measures prediction error.
- Backpropagation computes gradients.
- The Optimizer updates model parameters.
- Repeating this process gradually improves the model.

---

# Common Mistakes to Know

- Thinking Backpropagation updates the weights.

> No. It only computes gradients.

- Thinking the Loss Function updates the model.

> No. It only measures error.

- Assuming one epoch is enough.

> No . Transformers typically require many epochs and large datasets.

- Confusing learning rate with loss.

---

# Summary

Training a Transformer consists of five fundamental steps:

1. Forward Pass
2. Compute Loss
3. Backpropagation
4. Parameter Update
5. Repeat

By continuously minimizing the loss, the Transformer gradually learns meaningful representations and improves its predictions.

---

# What's Next?

We have now completed the complete Transformer architecture and its training process.

The final chapter compares three of the most influential Transformer-based architectures:

- Transformer (Encoder–Decoder)
- BERT (Encoder-Only)
- GPT (Decoder-Only)

Understanding their differences will help you recognize why different Transformer variants are suited for different tasks.

➡ **Next Chapter:** `14_GPT_vs_BERT_vs_Transformer.md`