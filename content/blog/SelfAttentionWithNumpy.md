---
title: "Build Self-Attention in Numpy"
date: "2025-01-14"
readTime: "10 min"
categories: ["Machine Learning", "Python", "Scratch", "Deep Learning", "SelfAttention"]
---

Class that implements the scaled dot-product multi-head self-attention mechanism, similar to the previous implementation but with **NumPy** instead of PyTorch. Let's break down the key parts of this code and explain the functionality with clarity.

![Self-Attention](https://raw.github.com/codewithdark-git/Transformers/2791fcd40efcf29d9b0dd796e20078a9d2005f86/neural-self-attention-cover-picture-1536x1151.png)

### Key Components:
The `SelfAttention` class is responsible for computing attention scores based on input embeddings, then applying multi-head attention, and projecting the output back to the original embedding space.

#### 1. **Self-Attention Mechanism Overview**:
The main idea behind self-attention is that each position in the input sequence attends to all other positions in the sequence (including itself) to compute a weighted sum of values (V) based on the similarity between queries (Q) and keys (K).

For **multi-head attention**, we split the input embeddings into multiple "heads", compute attention for each head independently, and then combine them to form the final output.

#### 2. **Mathematical Formulation**:

1. **Query, Key, and Value (Q, K, V)**:
   - **Queries (Q)**, **Keys (K)**, and **Values (V)** are vectors computed from the input embeddings using linear projections.
 
$$
Q = XW_Q, \quad K = XW_K, \quad V = XW_V
$$

   Where $(X)$ is the input matrix, and $(W_Q)$, $(W_K)$, and $(W_V)$ are learnable weight matrices (in this code, they are initialized randomly).

2. **Scaled Dot-Product Attention**:
   Attention scores are computed by taking the dot product of the query $(Q)$ and key $(K)$, scaled by the square root of the dimension of the keys:

$$
\text{attn-scores} = \frac{QK^T}{\sqrt{d_k}}
$$

   Where $(d_k)$ is the dimensionality of the key vectors (i.e., `head_dim`).

3. **Softmax for Attention Weights**:
   The attention scores are passed through a softmax function to normalize them into attention weights:

$$
\text{attn-weights} = \text{softmax}(\text{attn-scores})
$$

4. **Weighted Sum**:
   The attention weights are used to compute the weighted sum of the values (V):

$$
\text{attn-output} = \text{attn-weights} \times V
$$

5. **Multi-Head Attention**:
   Multiple attention heads are computed in parallel, and their outputs are concatenated and projected back to the original embedding space:

$$
\text{multi-head-output} = \text{Concat}(head-1, head-2, ..., head-h)W_O
$$

   Where $(W_O)$ is the output projection matrix.

#### 3. **Detailed Code Breakdown**:

```python
import numpy as np

class SelfAttention:
    def __init__(self, embed_dim, num_heads=1):
        """
        Initializes the SelfAttention module.
        
        Args:
            embed_dim (int): Dimensionality of the input/output embeddings.
            num_heads (int): Number of attention heads.
        """
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Initialize weight matrices for Q, K, V, and output projection
        self.q_proj = np.random.randn(embed_dim, embed_dim).astype(np.float32)  # Query projection matrix
        self.k_proj = np.random.randn(embed_dim, embed_dim).astype(np.float32)  # Key projection matrix
        self.v_proj = np.random.randn(embed_dim, embed_dim).astype(np.float32)  # Value projection matrix
        self.out_proj = np.random.randn(embed_dim, embed_dim).astype(np.float32)  # Output projection matrix
    
    def softmax(self, x, axis=-1):
        """Compute softmax along the specified axis."""
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))  # Numerical stability trick
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)  # Softmax normalization
    
    def forward(self, embeddings):
        """
        Forward pass for self-attention.
        
        Args:
            embeddings (np.ndarray): Input tensor of shape (batch_size, seq_len, embed_dim).
        
        Returns:
            np.ndarray: Contextual embeddings of shape (batch_size, seq_len, embed_dim).
        """
        batch_size, seq_len, _ = embeddings.shape
        
        # Compute Q, K, V by applying the projection matrices to the input embeddings
        Q = np.dot(embeddings, self.q_proj)  # (batch_size, seq_len, embed_dim)
        K = np.dot(embeddings, self.k_proj)  # (batch_size, seq_len, embed_dim)
        V = np.dot(embeddings, self.v_proj)  # (batch_size, seq_len, embed_dim)
        
        # Reshape Q, K, V for multi-head attention and transpose for batch-wise operations
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        # Scaled dot-product attention
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.head_dim)  # (batch_size, num_heads, seq_len, seq_len)
        attn_weights = self.softmax(scores, axis=-1)  # (batch_size, num_heads, seq_len, seq_len)
        
        # Compute the attention output by multiplying attention weights with V
        attn_output = np.matmul(attn_weights, V)  # (batch_size, num_heads, seq_len, head_dim)
        
        # Concatenate heads and reshape back to (batch_size, seq_len, embed_dim)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.embed_dim)
        
        # Final linear projection to get the final output embeddings
        output = np.dot(attn_output, self.out_proj)  # (batch_size, seq_len, embed_dim)
        
        return output
```

### Detailed Explanation:

1. **Initialization (`__init__` Method)**:
   - The number of heads (`num_heads`) and embedding dimension (`embed_dim`) are provided as input.
   - The projection matrices for Query $(Q)$, Key $(K)$, and Value $(V)$ are initialized randomly with the shape `(embed_dim, embed_dim)` using NumPy’s `randn`.
   - We also initialize the output projection matrix (`out_proj`) to combine the multi-head outputs.

2. **Softmax Function**:
   - The `softmax` function computes the softmax over a specified axis, which normalizes the attention scores into a probability distribution.

3. **Forward Pass (`forward` Method)**:
   - The `embeddings` input tensor has a shape of `(batch_size, seq_len, embed_dim)`, where `batch_size` is the number of sequences in a batch, `seq_len` is the sequence length, and `embed_dim` is the embedding dimension of each element.
   - The `embeddings` are projected into Q, K, and V using matrix multiplication (`np.dot`).
   - We reshape Q, K, and V to handle multiple attention heads. After reshaping, the tensor has the shape `(batch_size, num_heads, seq_len, head_dim)`.
   - The scaled dot-product attention is computed by taking the dot product between Q and K, dividing by the square root of the head dimension (`sqrt(head_dim)`), and then applying the softmax function to compute the attention weights.
   - The attention weights are then used to compute the weighted sum of the values $(V)$ to get the final attention output.
   - Finally, the multi-head outputs are concatenated, reshaped back to `(batch_size, seq_len, embed_dim)`, and a final linear transformation is applied using `out_proj`.

### Key Concepts:
- **Multi-Head Attention**: This mechanism allows the model to focus on different parts of the sequence using multiple attention heads. Each head computes its own attention, and the results are concatenated and passed through a linear transformation.
  
- **Scaled Dot-Product Attention**: This is the core of the attention mechanism, where the similarity between queries and keys is computed to determine how much focus should be given to each value $(V)$.


1. **Text Tokenization and Word Embedding Generation:**
   - It tokenizes the input text into words (converts the text to lowercase and splits it by spaces).
   - Then it converts these tokens into numerical indices based on a given vocabulary mapping.
   - For each token, it retrieves the corresponding word embedding from a randomly generated embedding matrix.

Let's break it down step by step:

### Step 1: Simple Text Tokenizer and Word Embeddings

```python
def text_to_word_embeddings(text, vocab, embedding_dim=128):
    """
    Converts text to word embeddings.
    
    Args:
        text (str): Input text.
        vocab (dict): Vocabulary mapping words to indices.
        embedding_dim (int): Dimensionality of the word embeddings.
    
    Returns:
        np.ndarray: Word embeddings of shape (1, seq_len, embedding_dim).
    """
    tokens = text.lower().split()  # Tokenize the input text into words (lowercase and split by spaces).
    seq_len = len(tokens)  # Calculate the length of the sequence (number of tokens).
    
    # Initialize a random embedding matrix of shape (vocab_size, embedding_dim).
    embedding_matrix = np.random.randn(len(vocab), embedding_dim).astype(np.float32)
    
    # Convert the tokens into their corresponding indices using the vocab dictionary.
    token_indices = [vocab[token] for token in tokens if token in vocab]
    
    # Use the token indices to get the corresponding word embeddings from the embedding matrix.
    word_embeddings = embedding_matrix[token_indices]  # Shape: (seq_len, embedding_dim)
    
    # Add a batch dimension to the embeddings to make the shape (1, seq_len, embedding_dim).
    return word_embeddings[np.newaxis, :, :]
```

#### **Function Breakdown:**

- **Input Arguments:**
  - `text`: A string input text that we want to convert to word embeddings.
  - `vocab`: A dictionary mapping words to their indices (this is a predefined vocabulary).
  - `embedding_dim`: The dimension of the word embeddings (default is 128).

- **Steps:**
  1. **Tokenization:** The input text is converted to lowercase and split by spaces into tokens.
  2. **Embedding Matrix:** A random matrix of shape `(vocab_size, embedding_dim)` is created, where `vocab_size` is the number of words in the vocabulary. Each row of this matrix will represent a word's embedding vector.
  3. **Token Indices:** For each token in the text, the corresponding index from the vocabulary is found and stored in `token_indices`.
  4. **Retrieve Embeddings:** Using the token indices, the corresponding embeddings from the embedding matrix are retrieved. The shape of `word_embeddings` becomes `(seq_len, embedding_dim)`, where `seq_len` is the number of tokens in the text.
  5. **Add Batch Dimension:** The embeddings are reshaped to include a batch dimension, resulting in a tensor of shape `(1, seq_len, embedding_dim)`.

### Step 2: Define Vocabulary and Test Data

```python
vocab = {"the": 0, "cat": 1, "sat": 2, "on": 3, "mat": 4}
text = "The cat sat on the mat"
tokens = text.lower().split()
```

- **Vocabulary (`vocab`)**: A simple dictionary mapping each word in the sentence to a unique index (for example, `"the"` is mapped to 0, `"cat"` to 1, etc.).
- **Test Data (`text`)**: The input text `"The cat sat on the mat"` is provided to test the function. The `tokens` variable holds the list of words split from the text.

### Step 3: Convert Text to Word Embeddings

```python
word_embeddings = text_to_word_embeddings(text, vocab)
```

- The function `text_to_word_embeddings()` is called with the input text and the vocabulary, and it returns the word embeddings corresponding to the input text.
- The resulting `word_embeddings` will have the shape `(1, seq_len, embedding_dim)`, where `seq_len` is the number of tokens in the sentence, and `embedding_dim` is 128 by default.

### Example Output:

For the text `"The cat sat on the mat"`, the resulting `word_embeddings` array will have the shape `(1, 6, 128)`, assuming each word is mapped to a 128-dimensional embedding vector. Each word in the sentence is replaced by its corresponding embedding from the randomly initialized `embedding_matrix`.

### Summary:

- **Tokenization** converts the input sentence into tokens.
- **Vocabulary** maps each word to an index.
- **Word Embeddings** are randomly generated and indexed based on the vocabulary.
- The final output includes these embeddings with a batch dimension, suitable for input to downstream neural networks.

![Embeddings](https://raw.github.com/codewithdark-git/Transformers/2791fcd40efcf29d9b0dd796e20078a9d2005f86/download.png)

## Step 4: **Plotting Comparing Original vs. Contextualized Embeddings**:
visualizes the difference between original (non-contextualized) word embeddings and contextualized word embeddings using **PCA (Principal Component Analysis)** for dimensionality reduction, followed by plotting using `matplotlib`. The visualization shows how the positions of words in embedding space change from their original (static) representations to their contextualized representations (e.g., after processing through a transformer model).

#### **Function Breakdown:**

### `plot_embeddings(original, contextual, tokens)`

- **Inputs:**
  - `original`: A numpy array or matrix representing the original (non-contextualized) word embeddings. The shape is `(seq_len, embedding_dim)`.
  - `contextual`: A numpy array or matrix representing the contextualized word embeddings (after processing through a contextual model like BERT). The shape is also `(seq_len, embedding_dim)`.
  - `tokens`: A list of strings representing the words in the sentence. Its length should match the number of rows in the `original` and `contextual` matrices.

- **Outputs:**
  - A 2D plot that compares the positions of the original embeddings versus the contextual embeddings. Words are shown in blue for original embeddings and red for contextual embeddings. Arrows are drawn to show how each word's position has shifted in the embedding space.

### **Steps in the Function:**

1. **Dimensionality Reduction Using PCA:**
   ```python
   pca = PCA(n_components=2)
   original_2d = pca.fit_transform(original)
   contextual_2d = pca.transform(contextual)
   ```
   - `PCA` is used to reduce the high-dimensional embeddings (e.g., 128 or 300 dimensions) to 2 dimensions for visualization purposes.
   - `fit_transform(original)` reduces the dimensionality of the `original` embeddings, and `transform(contextual)` applies the same transformation to the `contextual` embeddings.

2. **Plotting the Original Word Embeddings (in Blue):**
   ```python
   for i, token in enumerate(tokens):
       plt.scatter(original_2d[i, 0], original_2d[i, 1], color='blue', label='Original' if i == 0 else "", alpha=0.7)
       plt.text(original_2d[i, 0], original_2d[i, 1], f"{token} (O)", fontsize=10, color='blue')
   ```
   - A scatter plot is created for the original embeddings. Each point is colored blue and labeled with the token and `(O)` to indicate it is the original (static) embedding.

3. **Plotting the Contextualized Word Embeddings (in Red):**
   ```python
   for i, token in enumerate(tokens):
       plt.scatter(contextual_2d[i, 0], contextual_2d[i, 1], color='red', label='Contextualized' if i == 0 else "", alpha=0.7)
       plt.text(contextual_2d[i, 0], contextual_2d[i, 1], f"{token} (C)", fontsize=10, color='red')
   ```
   - A scatter plot is created for the contextualized embeddings. Each point is colored red and labeled with the token and `(C)` to indicate it is the contextualized embedding.

4. **Drawing Arrows Between Original and Contextualized Embeddings:**
   ```python
   plt.arrow(
       original_2d[i, 0], original_2d[i, 1],
       contextual_2d[i, 0] - original_2d[i, 0], contextual_2d[i, 1] - original_2d[i, 1],
       head_width=0.1, head_length=0.15, fc='gray', ec='gray', alpha=0.6
   )
   ```
   - For each token, an arrow is drawn from the position of the original embedding to the position of the contextualized embedding. The arrow shows how the word's representation changes contextually.

5. **Setting Up Plot Labels, Title, and Display:**
   ```python
   plt.title("Word Embeddings vs. Contextualized Embeddings")
   plt.xlabel("PCA Component 1")
   plt.ylabel("PCA Component 2")
   plt.legend()
   plt.grid(True)
   plt.show()
   ```
   - The plot is labeled with the title "Word Embeddings vs. Contextualized Embeddings".
   - The axes are labeled according to the two PCA components (Component 1 and Component 2).
   - A legend is displayed to indicate which points are original embeddings and which are contextualized.
   - The grid is enabled to improve readability of the plot.
   - Finally, `plt.show()` displays the plot.

### **Visual Output:**
- **Blue Points**: Represent the original (non-contextualized) embeddings, showing how words are positioned in a fixed embedding space.
- **Red Points**: Represent the contextualized embeddings, showing how words' positions change based on context (e.g., in a sentence).
- **Arrows**: Show the transformation of each word's embedding from the original to the contextualized space.

### **Example Use Case:**
- **Original Embeddings**: A word like "bank" will have the same representation regardless of its context (financial institution or riverside).
- **Contextualized Embeddings**: The word "bank" would have different embeddings depending on the context (financial institution vs. riverside).

### **Benefits of Using PCA for Visualization:**
- Reducing the dimensionality to 2D allows for easy visualization of how words are represented in embedding spaces.
- The arrows indicate the change in representation when the model processes the words in different contexts.


### **Step 5: Initialize Self-Attention and Compute Contextual Embeddings**

1. **Setting Hyperparameters:**
   ```python
   embed_dim = 128
   num_heads = 4
   ```
   - `embed_dim`: This is the dimension of the embeddings used in the self-attention model. Each word will be represented in a 128-dimensional space.
   - `num_heads`: The number of attention heads used in the multi-head attention mechanism. In this case, we use 4 attention heads, meaning the attention mechanism will learn 4 different attention scores simultaneously.

2. **Initializing the Self-Attention Layer:**
   ```python
   self_attention = SelfAttention(embed_dim, num_heads)
   ```
   - Here, an instance of the `SelfAttention` class is created with the specified `embed_dim` and `num_heads`.
   - This layer is designed to compute attention scores and produce contextual embeddings based on the input word embeddings.

3. **Computing the Contextual Embeddings:**
   ```python
   contextual_embeddings = self_attention.forward(word_embeddings)
   ```
   - The `forward` method of the `SelfAttention` class is called on the `word_embeddings` generated in the previous step.
   - This computes the **contextualized embeddings** for each token, where each word’s embedding is updated based on its context in the sentence.

   ### **What Happens in the `forward` Method:**

   - **Input**: The method takes the `word_embeddings` as input, which has the shape `(1, seq_len, embed_dim)`, where:
     - `1` is the batch size (since we have one sentence).
     - `seq_len` is the length of the input sentence (in this case, 5 words: "the", "cat", "sat", "on", "mat").
     - `embed_dim` is 128, the dimensionality of each word’s embedding.
   
   - **Computing Q, K, V (Query, Key, and Value):**
     ```python
     Q = np.dot(embeddings, self.q_proj)  # (batch_size, seq_len, embed_dim)
     K = np.dot(embeddings, self.k_proj)  # (batch_size, seq_len, embed_dim)
     V = np.dot(embeddings, self.v_proj)  # (batch_size, seq_len, embed_dim)
     ```
     - Queries (Q), Keys (K), and Values (V) are computed by multiplying the input embeddings with the respective learned weight matrices (`q_proj`, `k_proj`, and `v_proj`).

   - **Splitting into Multiple Attention Heads:**
     - The embeddings for Q, K, and V are reshaped and transposed to allow each attention head to operate independently:
       ```python
       Q = Q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
       K = K.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
       V = V.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
       ```
       - After reshaping, each head focuses on a different part of the information, providing a richer representation.

   - **Scaled Dot-Product Attention:**
     ```python
     scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.head_dim)  # (batch_size, num_heads, seq_len, seq_len)
     attn_weights = self.softmax(scores, axis=-1)  # (batch_size, num_heads, seq_len, seq_len)
     ```
     - The attention scores are computed by performing a scaled dot-product between Q and K. The scores are divided by the square root of the head dimension (`sqrt(self.head_dim)`) to stabilize gradients.
     - The scores are passed through a **softmax** to obtain the attention weights, which represent how much attention each word should pay to other words in the sequence.

   - **Computing Attention Output:**
     ```python
     attn_output = np.matmul(attn_weights, V)  # (batch_size, num_heads, seq_len, head_dim)
     ```
     - The attention output is computed by multiplying the attention weights with the Values (V), aggregating information from the sequence based on the attention mechanism.

   - **Concatenating Attention Heads:**
     ```python
     attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.embed_dim)
     ```
     - The output from all attention heads is concatenated together to form the final attention output.

   - **Final Linear Projection:**
     ```python
     output = np.dot(attn_output, self.out_proj)  # (batch_size, seq_len, embed_dim)
     ```
     - A final linear projection is applied to the concatenated attention output, mapping it back to the original embedding space (`embed_dim`), resulting in the final **contextualized embeddings**.

---

### **Key Takeaways:**
- **Self-Attention** allows each word to attend to all other words in the sequence, generating a new, context-dependent embedding for each word.
- **Multi-Head Attention** enables the model to capture different aspects of the context by using multiple attention heads, each focusing on different parts of the input.
- **Contextualized Embeddings** are generated by updating the word embeddings based on the surrounding words, which is key in capturing the meaning of words in different contexts (e.g., “bank” in "river bank" vs. “bank” in "financial bank").

---

## Reference:

- Code Reference: [github.com/codeeithdark-git/transformers](https://github.com/codewithdark-git/Transformers/blob/2791fcd40efcf29d9b0dd796e20078a9d2005f86/self_Attention.ipynb)
- Linkedin: [linkedin.com/in/codewithdark](https://www.linkedin.com/in/codewithdark)



