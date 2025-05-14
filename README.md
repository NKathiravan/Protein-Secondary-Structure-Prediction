# 🧬 Protein Secondary Structure Prediction using Transformer & Hybrid Quantum-Classical Models

## Overview

This project focuses on **predicting the secondary structure of proteins** (in Q8 format) using deep learning. We explore both classical Transformer-based architectures and quantum-enhanced variants to classify each amino acid residue into one of the 8 standard secondary structure classes.

---

## 🧪 Task Description

### 🎯 Goal:
To classify each amino acid in a protein sequence into one of the 8 secondary structure labels (Q8 classification).

---

## 📂 Dataset Preprocessing

- **Tokenization**: Protein sequences were tokenized character-wise using a custom `CharTokenizer`, mapping each amino acid to a unique integer.
- **Label Encoding**: Secondary structure labels (Q8 format) were encoded using a `label_map` dictionary.
- **Padding & Truncation**: 
  - Sequences were padded or truncated to a **fixed length of 350 tokens**.
  - Attention masks were generated to ignore padded tokens during loss computation.
- **Train-Validation Split**: Dataset split 80:20 using PyTorch's `random_split`.

---

## 🧠 Model Architectures

### 1️⃣ ProtBERT + MLP Model

- **Embedding**: Sequences passed through **pre-trained ProtBERT** to obtain contextual amino acid embeddings.
- **MLP Head**: Fully connected layers with decreasing hidden sizes.
- **Output**: Final predictions made via softmax over 8 classes (Q8 format).

### 2️⃣ Transformer from Scratch (Encoder-Only)

- **Input Embedding**: Amino acid tokens mapped to 128-dimensional vectors (`d_model = 128`).
- **Positional Encoding**: Fixed sinusoidal encodings added to retain positional information.
- **Encoder Layer**:
  - **Multi-Head Attention** with 8 heads (`n_heads = 8`)
  - **Feed-Forward Network** with ReLU activation
  - **LayerNorm + Dropout** after attention and FFN
- **Output Layer**: Linear layer → softmax over 8 classes

---

## 🔬 Quantum Hybridization (Optional)

We explore replacing classical Transformer components with **quantum-inspired modules**:

| Classical Component        | Quantum Equivalent                   |
|---------------------------|--------------------------------------|
| Input Embedding           | Quantum Encoding (Angle/Amplitude)  |
| Self-Attention Mechanism  | Quantum Self-Attention               |
| Feedforward Network       | Quantum Feedforward Layer           |
| Post-Attention Normalization | Quantum Normalization/Re-Encoding |

---

## ⚙️ Training Configuration

- **Loss Function**: Cross-entropy (per token), padded positions masked using attention masks.
- **Optimizer**: Adam (`lr = 1e-4`)
- **Batch Size**: 32
- **Epochs**: 25
- **Metrics**: Accuracy, Precision, Recall, F1-score on validation set

---

## 📈 Results (Prototype Phase)

- The model is currently in prototype phase and being evaluated for performance.
- Preliminary results show promise for combining contextualized embeddings and quantum layers.

---

## 📌 Future Work

- Expand quantum module integration (currently in experimental phase)
- Fine-tune ProtBERT + Transformer on larger datasets
- Explore Q3 classification alongside Q8
- Visualize secondary structure predictions with 3D mapping

---

## 🛠️ Technologies Used

- Python, PyTorch
- HuggingFace Transformers (ProtBERT)
- Qiskit / PennyLane (for quantum extensions)
- NumPy, Scikit-learn, Matplotlib

---

## 🤝 Contributions

Feel free to fork and contribute to improvements or add support for Q3 classification. All ideas for better hybrid architectures or evaluation techniques are welcome!

---
