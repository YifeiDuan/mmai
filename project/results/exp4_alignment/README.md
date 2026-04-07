# Exp 4: Contrastive Alignment (CLIP-style)

## Overview

Explore whether CLIP-style contrastive alignment between crystal structure and text modalities can improve downstream band gap prediction. Two-stage approach: (1) align structure and text encoders into a shared embedding space via InfoNCE loss, (2) train a regression head on the aligned embeddings.

## Method

### Stage 1: Contrastive Alignment Pretraining

```
CIF  → CGCNN  → proj(128→128) → z_s  ─┐
                                        ├─ InfoNCE: maximize cos(z_s_i, z_t_i), minimize cos(z_s_i, z_t_j≠i)
Text → SciBERT → proj(768→128) → z_t  ─┘
```

- Symmetric InfoNCE loss (CLIP-style)
- Learnable temperature parameter
- Separate learning rates: projection heads at 5e-4, encoders at 5e-5

### Stage 2: Downstream Regression

Two variants tested:
- **Frozen**: Freeze aligned encoders, train only a regression MLP (66K params)
- **FiLM finetune**: Use aligned weights as initialization, finetune encoder + FiLM head (15M params)

## Stage 1 Results: Alignment Quality

| Metric | Start | Final |
|--------|-------|-------|
| Retrieval Acc (val) | 1.9% (random) | **52.8%** |
| Positive Pair Cosine Similarity | 0.036 | **0.879** |
| Temperature | 0.070 | 0.082 |
| Training Time | — | 281s |

The model successfully learns to align structure and text embeddings: same-material pairs reach cosine similarity ~0.88, and the model correctly retrieves the matching text for >50% of structures (vs 1.9% random baseline on 54 val samples).

## Stage 2 Results: Regression

| Model | Trainable Params | MAE (eV) ↓ | RMSE (eV) ↓ | R² ↑ |
|-------|-----------------|-----------|------------|------|
| Aligned → Concat (frozen) | 66K | 0.7905 | 1.0876 | 0.5312 |
| Aligned → FiLM (finetune) | 15.1M | 0.5880 | 0.9288 | 0.6582 |

### Cross-comparison with Exp 1-3

| Model | Approach | MAE ↓ | R² ↑ |
|-------|----------|-------|------|
| CGCNN (Exp 1) | Structure only | 0.656 | 0.620 |
| SciBERT-frozen (Exp 2) | Text only, frozen | 1.215 | 0.187 |
| SciBERT-finetune (Exp 2) | Text only, finetune | 0.751 | 0.554 |
| Exp 3 Concat Fusion | End-to-end fusion | 0.583 | 0.639 |
| Exp 3 Gated Fusion | End-to-end fusion | 0.550 | 0.664 |
| **Exp 3 FiLM Fusion** | **End-to-end fusion** | **0.509** | **0.674** |
| Exp 4 Aligned-frozen | Align → frozen MLP | 0.791 | 0.531 |
| Exp 4 Aligned-FiLM | Align → finetune FiLM | 0.588 | 0.658 |

## Analysis

### What Worked

1. **Alignment itself is successful**: 52.8% retrieval accuracy and 0.88 cosine similarity show the model genuinely learned to match structures with their text descriptions.

2. **Aligned frozen embeddings >> Frozen SciBERT**: With the same setup (frozen encoder + MLP), aligned embeddings (R²=0.53) vastly outperform frozen SciBERT (R²=0.19). The alignment process produces a more information-dense representation in 128 dimensions than SciBERT's generic 768-dim [CLS] embedding.

### What Didn't Work

1. **Alignment did not improve over end-to-end fusion**: Exp 4 Aligned-FiLM (R²=0.658) < Exp 3 FiLM (R²=0.674). The contrastive pretraining step did not provide a better initialization than directly training the fusion model end-to-end.

2. **Projection bottleneck**: Alignment compresses CGCNN (128-d) and SciBERT (768-d) into a shared 128-d space. This loses information that the end-to-end FiLM model retains by working with the full 128+768 dimensional encoder outputs.

3. **Objective mismatch**: InfoNCE optimizes for "distinguishing different materials", not "predicting band gap". The alignment may discard features that are irrelevant for material identity but useful for property regression (e.g., subtle bond length variations that don't change the overall structure description but affect band gap).

### Why — Small Data Regime

With only 383 training samples:
- InfoNCE has very few negatives per batch (batch_size=128 → only 127 negatives)
- Two-stage training accumulates noise: alignment errors propagate to regression
- End-to-end training is more sample-efficient since all gradient signal directly optimizes the final objective

Contrastive alignment is expected to shine with larger datasets (>10K samples) where:
- More negatives improve the contrastive signal
- The aligned space enables zero-shot transfer and cross-modal retrieval
- Pretrained representations generalize better to unseen compositions

### Joint Training Attempt (Failed)

We also tried joint training (L = 0.5 * L_InfoNCE + 0.5 * L_regression) with soft contrastive masking. This completely failed (val retrieval acc = 5.6%) because the regression loss dominated the contrastive signal, preventing meaningful alignment from forming. This confirms that alignment and regression are somewhat competing objectives on small data.

## Significance for the Project

This experiment provides important negative results:

1. **End-to-end fusion > two-stage alignment** on small datasets — the FiLM fusion model (Exp 3) remains our best approach.
2. **Alignment has potential value for larger datasets** — the 52.8% retrieval accuracy shows the concept works, and the aligned-frozen result (R²=0.53 with only 66K params) demonstrates the space is meaningful.
3. **Motivates future work**: scaling the dataset (relaxing space group filters: 592 → 2000+ samples) and revisiting alignment with more data.

## Output Files

```
results/exp4_alignment/
├── README.md
├── stage1/
│   ├── best_model.pt              # Aligned encoder checkpoint
│   ├── results.json
│   ├── training_history.csv
│   └── val_embeddings.npz         # For t-SNE visualization
├── stage2_concat_frozen/
│   ├── best_model.pt
│   ├── results.json
│   ├── training_history.csv
│   └── test_predictions.csv
└── stage2_film_finetune/
    ├── best_model.pt
    ├── results.json
    ├── training_history.csv
    └── test_predictions.csv
```

## Run Commands

```bash
cd /home/gridsan/wouyang/MMA/MMA_proj/perovskite-pvk

# Stage 1: Contrastive alignment
python scripts/run_exp4_align.py

# Stage 2: Downstream regression
python scripts/run_exp4_regress.py --mode concat --freeze-encoders
python scripts/run_exp4_regress.py --mode film
```
