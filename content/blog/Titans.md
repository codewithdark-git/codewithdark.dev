---
title: "Titans Transformer Model "
date: "2025-01-17"
readTime: "7 min"
categories: ["Machine Learning", "Python", "Scratch", "Transformer"]
---


This implementation is based on the **Titans** architecture introduced in the paper ***Titans: Learning to Memorize at Test Time***  -> [Paper](https://arxiv.org/abs/2501.00663). The architecture integrates a memory module with the traditional Transformer model to enable the model to remember and leverage long-term context across sequences. This capability is crucial for tasks that require understanding dependencies beyond a fixed-length context window, which is often a limitation of standard Transformers.

The key innovation is the **Titans Memory Module**, which is a dynamic memory component that learns to update and retrieve relevant memory entries based on the input sequence, providing the model with access to long-term memory during inference.

![](https://raw.github.com/codewithdark-git/codewithdark.dev/4d9982308f2a31c9da22c70552a4bb81ae7792d9/public/Untitled%20diagram-2025-01-16-135202.png)

---

### **Classes and Functions**

#### **1. `TitansMemoryModule`**
The **TitansMemoryModule** maintains and updates a memory matrix, which is used to store long-term context information. The module utilizes attention to interact with this memory, and the memory is updated based on the surprise (difference between input values and retrieved memory).

**Initialization:**
```python
def __init__(self, d_model, memory_size=512):
```
- `d_model`: The dimensionality of the model.
- `memory_size`: The number of memory slots.

**Mathematical Formulas:**
1. **Attention Scores:** The attention scores between the input sequence   $(X)$ and the memory $(M)$ are computed as:
$$
   \text{Attention_Scores} = X \cdot M^T
$$   
   Where $(X)$ represents the input, and $(M)$ represents the memory matrix.

2. **Attention Weights:** The attention weights are calculated using the softmax function:
$$
   \text{Attention_Weights} = \text{softmax}(\text{Attention_Scores})
$$
   This normalizes the scores and assigns a weight to each memory entry.

3. **Retrieved Memory:** The retrieved memory is a weighted sum of memory entries:
$$
   \text{Retrieved_Memory} = \text{Attention_Weights} \cdot M
$$

4. **Memory Update:** The memory is updated based on the values and the retrieved memory, modulated by a forgetting gate $ f(x)$:

$$
   \text{Memory_Update} = f(x) \cdot M + (1 - f(x)) \cdot \text{Retrieved_Memory}
$$

   Where the forgetting gate $ f(x)$ is a sigmoid function applied to the input values:
$$
   f(x) = \sigma(W_f \cdot V)
$$
   Where $(V)$ represents the values of the input sequence.

---

#### **2. `TitansTransformerEncoderLayer`**
The **TitansTransformerEncoderLayer** implements a single layer of the Transformer encoder. It incorporates the **TitansMemoryModule** to allow the model to leverage both attention and memory-based long-term context.

**Initialization:**
```python
def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, memory_size=512):
```
- `d_model`: The dimensionality of the model.
- `nhead`: The number of attention heads.
- `dim_feedforward`: The dimensionality of the feed-forward network.
- `dropout`: The dropout rate.

**Mathematical Formulas:**
1. **Self-Attention:** The self-attention mechanism computes the attention between input tokens as:

$$
   \text{Attention}(Q, K, V) = \text{softmax}\left( \frac{Q K^T}{\sqrt{d_k}} \right) V
$$

   Where $(Q) (query)$, $(K) (key)$, and $(V) (value)$ are derived from the input sequence.

2. **Feed-Forward Network:** The encoder applies a two-layer feed-forward network with ReLU activation:
$$
   \text{FFN}(x) = \text{ReLU}(x W_1 + b_1) W_2 + b_2
$$

---

#### **3. `TitansTransformer`**
The **TitansTransformer** is the main model that combines the embedding layer, positional encoding, multiple encoder layers with Titans memory, and the final output layer. This model processes token sequences, updating its memory based on the input and utilizing it during inference.

**Initialization:**
```python
def __init__(self, num_tokens, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1, memory_size=512):
```
- `num_tokens`: The size of the vocabulary (number of tokens).
- `d_model`: The dimensionality of the model.
- `nhead`: The number of attention heads.
- `num_layers`: The number of encoder layers.

**Forward Pass:**
```python
def forward(self, src, src_mask=None, src_key_padding_mask=None):
```
- The forward pass begins by embedding the input tokens and applying positional encoding. It then passes the sequence through multiple encoder layers, where attention and memory retrieval take place. Finally, the model outputs the predicted token for each position in the sequence.

---

#### **4. `PositionalEncoding`**
The **PositionalEncoding** class adds positional information to the input embeddings, which is essential for the model to understand the order of the tokens in the sequence.

**Mathematical Formula:**
The positional encoding vector $ \mathbf{PE}(t)$ for position $(t)$ and dimension $(i)$ is calculated as:
$$
\mathbf{PE}(t, 2i) = \sin\left( t / 10000^{2i/d_{\text{model}}} \right)
$$
$$
\mathbf{PE}(t, 2i+1) = \cos\left( t / 10000^{2i/d_{\text{model}}} \right)
$$
Where $(t)$ is the position of the token in the sequence and $ d_{\text{model}} $ is the dimensionality of the model.

---

### **Summary of Key Features**

1. **Titans Memory Module:** The core of the Titans model, which enables the model to dynamically retrieve and update memory entries based on the input sequence. This allows the model to effectively handle long-term dependencies.

2. **Attention Mechanism with Memory Integration:** The model combines traditional self-attention with memory, allowing it to consider both short-term and long-term dependencies across sequences.

3. **Enhanced Long-Term Dependency Handling:** By incorporating a memory module, the Titans Transformer can learn and utilize long-term context, addressing the challenge of capturing long-range dependencies in sequences.

4. **Scalability and Versatility:** The Titans Transformer is scalable, handling sequences of varying lengths, and is versatile for tasks across different domains, including language modeling, time-series prediction, and more.

---
---

The code you provided seems like a full implementation for training and evaluating the **Titans Transformer** model on a subset of the **WikiText-2** dataset. I'll break down the main components of the code and clarify any necessary points for better understanding:

### Key Components of the Code:

1. **Data Preparation (`get_data` function)**:
   - **Dataset Loading:** The dataset is loaded from Hugging Face using the `load_dataset` method with the `"wikitext"` dataset, specifically the `"wikitext-2-raw-v1"` subset.
   - **Tokenization:** The `AutoTokenizer` is used to tokenize the text data, and the `pad_token` is set to the `eos_token`.
   - **Subset Selection:** A subset of the data is selected for training (1000 samples), and smaller subsets are used for validation and testing (1/10th of the training size).
   - **Tensor Conversion:** The input IDs are converted to PyTorch tensors for processing.
   - **Vocabulary Size:** The size of the vocabulary is obtained from the tokenizer.

2. **Batching (`batchify` function)**:
   - This function splits the dataset into batches. The input tensor is reshaped into a format that is suitable for model training, with the data arranged in a batch-first format.

3. **Evaluation (`evaluate` function)**:
   - This function is used to evaluate the model on validation or test data. It calculates the total loss by passing the input data through the model and comparing the output to the targets using the `CrossEntropyLoss` criterion.
   
4. **Training Loop (`train_model` function)**:
   - **Hyperparameters:** Defines batch size, number of epochs, model dimensions, and other hyperparameters.
   - **Device Setup:** The model is placed on a GPU if available, otherwise, it uses the CPU.
   - **Model Initialization:** The model is initialized using the Titans Transformer class (presumably defined elsewhere).
   - **Optimization:** The optimizer is set to Adam, and a learning rate scheduler (`StepLR`) is used to decay the learning rate after each epoch.
   - **Training and Evaluation:** The model is trained for a set number of epochs, with the validation loss evaluated after each epoch. If the validation loss improves, the model is saved. The test loss is evaluated after training completes.

### Points of Clarification:

1. **Model Definition (`TitansTransformer`)**:
   - The code references a `TitansTransformer` class, but this class is not defined in the provided code. You will need to define this class, which should incorporate the memory module and attention mechanism based on the **Titans Transformer** paper.
   - The model should include the components for handling long-term dependencies, as mentioned in the paper, such as memory updating and retrieval during each attention operation.

2. **Memory Module**:
   - If you're following the **Titans Transformer** architecture, ensure that you include the memory module, which interacts with the attention mechanism.
   - The `TitansMemoryModule` will handle memory retrieval and updates, as discussed in the original paper.

3. **Saving Model Checkpoints**:
   - The model state is saved after each epoch if the validation loss improves (`torch.save(model.state_dict(), 'titans_transformer_model.pt')`). This ensures that the best model (based on validation loss) is retained.

4. **Optimization and Scheduler**:
   - The optimizer used is Adam with a learning rate of 0.0001. The learning rate scheduler decays the learning rate by a factor of 0.95 after each epoch.

5. **Evaluation during Training**:
   - The validation loss is evaluated after every epoch and printed to monitor the model’s progress. After training, the model's performance is tested on a held-out test set.

### To-Do Before Running the Code:

- **Define `TitansTransformer`:** Implement the Titans Transformer model class, which will include the self-attention mechanism, memory handling, and feedforward layers as described in the **Titans** paper.
  
- **Check Dataset Availability:** Ensure the `wikitext` dataset is available via Hugging Face. The `"wikitext-2-raw-v1"` dataset should work, but you can explore other datasets based on your requirements.

```
| epoch   1 | batch   0 | loss 11.05
| epoch   1 | batch 100 | loss  4.72
| epoch   1 | batch 200 | loss  3.27
| epoch   1 | batch 300 | loss  2.55
| epoch   1 | batch 400 | loss  2.26
| epoch   1 | batch 500 | loss  2.05
| epoch   1 | batch 600 | loss  1.89
| epoch   1 | batch 700 | loss  1.81
| epoch   1 | batch 800 | loss  1.75
| epoch   1 | batch 900 | loss  1.76
| epoch   1 | batch 1000 | loss  1.79
| epoch   1 | batch 1100 | loss  1.72
| epoch   1 | batch 1200 | loss  1.72
| epoch   1 | batch 1300 | loss  1.64
| epoch   1 | batch 1400 | loss  1.62
| epoch   1 | batch 1500 | loss  1.58
| epoch   1 | batch 1600 | loss  1.57
| epoch   1 | batch 1700 | loss  1.55
| epoch   1 | batch 1800 | loss  1.53
| epoch   1 | batch 1900 | loss  1.53
-----------------------------------------------------------------------------------------
| end of epoch   1 | time: 108.01s | valid loss  0.14
-----------------------------------------------------------------------------------------
| epoch   2 | batch   0 | loss  3.25
| epoch   2 | batch 100 | loss  1.53
| epoch   2 | batch 200 | loss  1.62
| epoch   2 | batch 300 | loss  1.41
| epoch   2 | batch 400 | loss  1.37
| epoch   2 | batch 500 | loss  1.32
| epoch   2 | batch 600 | loss  1.27
| epoch   2 | batch 700 | loss  1.27
| epoch   2 | batch 800 | loss  1.27
| epoch   2 | batch 900 | loss  1.32
| epoch   2 | batch 1000 | loss  1.38
| epoch   2 | batch 1100 | loss  1.35
| epoch   2 | batch 1200 | loss  1.38
| epoch   2 | batch 1300 | loss  1.32
| epoch   2 | batch 1400 | loss  1.32
| epoch   2 | batch 1500 | loss  1.29
| epoch   2 | batch 1600 | loss  1.30
| epoch   2 | batch 1700 | loss  1.30
| epoch   2 | batch 1800 | loss  1.28
| epoch   2 | batch 1900 | loss  1.30
---------------------------------------------------------------------------
| end of epoch   2 | time: 108.31s | valid loss  0.15
-----------------------------------------------------------------------------------------
| epoch   3 | batch   0 | loss  3.29
| epoch   3 | batch 100 | loss  1.50
| epoch   3 | batch 200 | loss  1.60
| epoch   3 | batch 300 | loss  1.40
| epoch   3 | batch 400 | loss  1.36
| epoch   3 | batch 500 | loss  1.31
| epoch   3 | batch 600 | loss  1.26
| epoch   3 | batch 700 | loss  1.26
| epoch   3 | batch 800 | loss  1.26
| epoch   3 | batch 900 | loss  1.31
| epoch   3 | batch 1000 | loss  1.37
| epoch   3 | batch 1100 | loss  1.34
| epoch   3 | batch 1200 | loss  1.37
| epoch   3 | batch 1300 | loss  1.31
| epoch   3 | batch 1400 | loss  1.31
| epoch   3 | batch 1500 | loss  1.28
| epoch   3 | batch 1600 | loss  1.29
| epoch   3 | batch 1700 | loss  1.29
| epoch   3 | batch 1800 | loss  1.27
| epoch   3 | batch 1900 | loss  1.29
---------------------------------------------------------------------------
| end of epoch   3 | time: 108.19s | valid loss  0.15
---------------------------------------------------------------------------
=========================================================================================
| End of training | test loss  0.12
=========================================================================================
```

---
---

This code is a benchmark for comparing the performance of the **Standard Transformer** and the **Titans Transformer** on the **WikiText-2** dataset. The benchmarking process includes measuring inference time and perplexity for varying sequence lengths. The results are then plotted and saved as an image.

Here’s a breakdown of the code:

### Key Components:

1. **Positional Encoding (`PositionalEncoding` class)**:
   - Implements the positional encoding mechanism used in the Transformer model, where sine and cosine functions are used to encode the relative positions of tokens in a sequence.
   - It’s applied to the input embeddings before they are passed into the transformer layers.

2. **Standard Transformer (`StandardTransformer` class)**:
   - Implements a standard transformer model with positional encoding, an embedding layer, and a transformer encoder block.
   - The model consists of:
     - **Embedding Layer:** Converts token IDs into vectors of fixed dimension (`d_model`).
     - **Positional Encoding:** Adds information about token positions.
     - **Transformer Encoder:** Performs self-attention and transformations.
     - **Linear Output Layer:** Maps the transformer output to the vocabulary space.

3. **Data Loading (`get_data` function)**:
   - Loads the **WikiText-2** dataset from Hugging Face and tokenizes the text data using the GPT-2 tokenizer.
   - The data is split into training, validation, and test sets, and returned as tensors.

4. **Benchmarking (`benchmark_models` function)**:
   - Tests both the **Standard Transformer** and **Titans Transformer** on the test data for varying sequence lengths (`128, 256, 512, 1024`).
   - For each sequence length:
     - **Inference Time:** Measures how long the model takes to process a batch of data.
     - **Perplexity:** Computes the perplexity of the model, which is a measure of how well the model predicts the next token in a sequence.
   - The results (time and perplexity) for both models are stored and returned.

5. **Plotting Benchmark Results (`plot_benchmark_results` function)**:
   - Uses `matplotlib` to plot the benchmark results:
     - **Inference Time vs Sequence Length**
     - **Perplexity vs Sequence Length**
   - The plots are saved as `benchmark_results.png`.

6. **Main Execution (`__main__`)**:
   - Runs the benchmarking function, collects results, and then plots and saves them.

### Key Notes:
1. **Titans Transformer Model**:
   - The code assumes that you already have a **TitansTransformer** class defined. This model should be similar to the standard transformer but include additional features, such as memory mechanisms or specialized attention mechanisms. Make sure the Titans model is correctly implemented and imported.
   
2. **Data Handling**:
   - The tokenizer from Hugging Face's `gpt2` model is used, and the dataset is tokenized and padded to a maximum sequence length of 512.
   
3. **Memory and Performance**:
   - The benchmarking function tracks performance for different sequence lengths to observe how both models scale.
   - The `torch.cuda.empty_cache()` is called before each test to clear the GPU memory.

4. **Results Plotting**:
   - Results are plotted and saved to a PNG file (`benchmark_results.png`), which includes both inference time and perplexity.

![](https://raw.github.com/codewithdark-git/codewithdark.dev/4d9982308f2a31c9da22c70552a4bb81ae7792d9/public/benchmark_results.png)

----
----


### **Note: Implementation Disclaimer**

This implementation is my personal attempt to experiment with the **Titans Transformer** architecture, and I am currently in the learning stage. While I have tried to follow the ideas presented in the original paper, this code may contain errors or inefficiencies, and may not fully align with the exact specifications or optimizations described in the paper.

I encourage you to refer to the original **Titans Transformer** paper for an accurate understanding of the proposed architecture and its intended implementations. This code is meant for educational and experimental purposes, and further refinement is needed for production-level results.


```
[GitHub Repo](https://github.com/codewithdark-git/titans-transformer.git)
```

```
![LinkedIn](https://linkedin.com/in/codewithdark)
```
```
[Join GitHub Community](https://github.com/XCollab)
```

Thank you for your understanding, and feel free to provide feedback or suggestions for improvement!

---