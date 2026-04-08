# Homework 2: Multimodal Fusion and Contrastive Learning

## Project Goal

Build and evaluate multimodal fusion models and a contrastive learning alignment model for ABO₃ perovskite property prediction (band gap, formation energy), using crystal graph and text embeddings extracted from the HW1 dataset.

## Tasks and Implementations


### Unimodal Baselines on AV-MNIST
Trained CNN-based unimodal models on the AV-MNIST benchmark (audio spectrograms + digit images):
- **Audio CNN**: best accuracy **41.65%** (3-layer conv + dropout 0.15)
- **Image CNN**: best accuracy **64.67%** (3-layer conv + dropout 0.1)

Image significantly outperforms audio, motivating multimodal fusion to leverage complementary signals.

### Early Fusion on AV-MNIST
Implemented early fusion (concatenate encoder outputs → shared MLP) and discussed causes of training stagnation (learning rate, initialization, overfitting) and alternative fusion strategies.

### Four Fusion Methods on Perovskite Dataset
Extracted unimodal embeddings (CGCNN graph embeddings + `all-MiniLM-L6-v2` text embeddings, both projected to `dim=32`), then implemented and compared four fusion methods for formation energy prediction:

| Fusion Method | MAE (eV) | RMSE (eV) | R² | Params | Memory (MB) | Time (s) |
|---|---|---|---|---|---|---|
| Early Fusion | 0.745 | 1.070 | 0.627 | 16,641 | 0.063 | 1.86 |
| **Late Fusion** | **0.682** | **0.975** | **0.690** | 25,092 | 0.096 | 1.37 |
| Tensor Fusion | 0.771 | 1.159 | 0.562 | 46,497 | 0.177 | 0.96 |
| LMF Fusion | 0.787 | 1.172 | 0.552 | 7,585 | 0.029 | 1.57 |

**Late Fusion** achieved the best validation performance (MAE 0.682 eV, R² 0.69) with softmax-weighted combination of per-modality MLP predictions (hidden dims 128→64, dropout 0.1, Adam lr=1e-3, early stopping patience=20).

Tensor and LMF fusion underperformed — Tensor Fusion overfits with its high-dimensional interaction terms on the small dataset, while LMF underfits due to reduced capacity. Late Fusion strikes the best balance for this small-data, high-level-embedding regime.

### Contrastive Learning and Cross-Modal Alignment
Trained a CLIP-style contrastive model to align graph and text embeddings in a shared latent space (cross-entropy loss over in-batch negatives):
- **Graph → Original Text retrieval**: model successfully retrieves the correct paired text.
- **Graph → New text queries** (space group, local site geometry, A-site element): partial success — correct answer often appears in top-5 but not reliably top-1; negative cosine similarities observed for hard queries.
- **Post-alignment visualization** (t-SNE): most failures in top-1 retrieval occur for structurally similar materials with nearly identical embeddings; top-5 retrieval is near-perfect, confirming that failures arise from embedding proximity rather than incorrect alignment.


## Key Takeaways
- Late fusion outperforms early and higher-order fusions on the small perovskite dataset — high-capacity interaction methods (TFN, LMF) are data-hungry and do not pay off with ~500 samples.
- Precomputed, compressed unimodal embeddings already carry much of the cross-modal information; a lightweight weighted combination suffices for regression.
- Contrastive alignment enables meaningful cross-modal retrieval even without explicit labels, but performance degrades for hard fine-grained queries (space group) where the text vocabulary is too coarse.
- LMF is the most parameter- and memory-efficient option and is worth revisiting with richer embeddings or more data.
