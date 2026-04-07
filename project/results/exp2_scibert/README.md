# Exp 2: SciBERT Baseline (Text-only)

## Overview

SciBERT-based regression baseline for band gap prediction using only Robocrystallographer-generated text descriptions as input. Two variants are compared: frozen encoder (MLP head only) and fine-tuned (last 2 BERT layers + MLP head).

## Setup

- **Model**: `allenai/scibert_scivocab_uncased` (110M params) + MLP head (768 -> 256 -> 1)
- **Input**: robocrys_text -> SciBERT [CLS] embedding (768-dim) -> MLP -> scalar
- **Target**: band_gap (eV)
- **Data**: text_ok=True samples only (493 total)

### Hyperparameters

| Param | Value |
|-------|-------|
| max_seq_len | 256 |
| hidden_dim | 256 |
| dropout | 0.1 |
| batch_size | 16 |
| lr | 2e-5 |
| optimizer | AdamW (weight_decay=0.01) |
| early stopping | patience=10 |

## Data Split (text_ok only)

| Split | Samples |
|-------|---------|
| Train | 383 |
| Val   | 54 |
| Test  | 56 |

## Results

| Variant | Trainable Params | MAE (eV) | RMSE (eV) | R² | Training Time |
|---------|-----------------|----------|-----------|------|---------------|
| SciBERT-frozen | 197K | 1.2152 | 1.4326 | 0.1867 | 132.6s |
| SciBERT-finetune | 14.96M | 0.7507 | 1.0609 | 0.5539 | 71.2s |

### Cross-comparison with Exp 1

| Model | Modality | MAE ↓ | RMSE ↓ | R² ↑ |
|-------|----------|-------|--------|------|
| CGCNN | Structure | **0.6557** | **0.9686** | **0.6197** |
| SciBERT-frozen | Text | 1.2152 | 1.4326 | 0.1867 |
| SciBERT-finetune | Text | 0.7507 | 1.0609 | 0.5539 |

## Analysis

### Frozen Encoder (R²=0.19)

- Frozen SciBERT's [CLS] embedding is a general-purpose scientific text representation, not tuned for property prediction. The MLP head alone cannot extract sufficient band gap information from this fixed representation.
- R²=0.19 is low but nonzero, indicating that the text does carry some property-relevant signal even without fine-tuning.
- Val MAE was still slowly decreasing at epoch 50 (1.60 -> 1.03), suggesting the model had not fully converged but was hitting a fundamental ceiling.

### Fine-tuned Encoder (R²=0.55)

- Fine-tuning the last 2 BERT layers dramatically improves performance: R² jumps from 0.19 to 0.55. This proves that SciBERT can learn to extract material property information from Robocrystallographer descriptions when given task-specific supervision.
- The model converges quickly (best at ~epoch 15, early stop at 23) despite having 15M trainable parameters on only 383 training samples. This suggests the pretrained representations provide a strong initialization.
- Fine-tune is actually faster (71s) than frozen (133s) because early stopping kicks in earlier.

### Structure vs Text

- CGCNN (R²=0.62) > SciBERT-finetune (R²=0.55): Structure modality is stronger, as expected. GNN directly encodes precise atomic coordinates and bond geometry, while text is a lossy semantic compression of the same structure.
- However, the gap is relatively small (0.07 in R²). This suggests that Robocrystallographer text captures most of the structure's predictive information, and potentially captures different aspects (e.g., global descriptors like "tilted octahedra", "corner-sharing" that require multi-hop reasoning in GNN).

## Significance for the Project

This establishes the ideal baseline configuration for multimodal fusion:
- Structure: strong but imperfect (R²=0.62)
- Text: weaker but independently valuable (R²=0.55)
- If the two modalities encode complementary information, fusion (Exp 3) should exceed R²=0.62.

## Output Files

```
results/exp2_scibert/
├── README.md
├── frozen/
│   ├── best_model.pt
│   ├── results.json
│   ├── training_history.csv
│   └── test_predictions.csv
└── finetune/
    ├── best_model.pt
    ├── results.json
    ├── training_history.csv
    └── test_predictions.csv
```

## Run Commands

```bash
cd /home/gridsan/wouyang/MMA/MMA_proj/perovskite-pvk

# Frozen encoder (MLP head only)
python scripts/run_exp2_scibert.py --freeze

# Fine-tune last 2 layers
python scripts/run_exp2_scibert.py
```
