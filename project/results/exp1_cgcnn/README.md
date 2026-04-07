# Exp 1: CGCNN Baseline (Structure-only)

## Overview

CGCNN (Crystal Graph Convolutional Neural Network) baseline for band gap prediction using only crystal structure (CIF files) as input. This serves as the structure-only baseline for our multimodal fusion study.

## Setup

- **Model**: CGCNN (Xie & Grossman, PRL 2018)
- **Input**: CIF -> crystal graph (nodes=atoms, edges=bonds within 8A radius, max 12 neighbors)
- **Target**: band_gap (eV)
- **Parameters**: 80,193

### Hyperparameters

| Param | Value |
|-------|-------|
| atom_fea_len | 64 |
| n_conv | 3 |
| h_fea_len | 128 |
| batch_size | 64 |
| lr | 1e-3 |
| optimizer | Adam (weight_decay=1e-4) |
| scheduler | ReduceLROnPlateau (patience=15, factor=0.5) |
| early stopping | patience=30 |

## Data Split

| Split | Samples |
|-------|---------|
| Train | 468 |
| Val   | 63 |
| Test  | 61 |

## Results

| Metric | Value |
|--------|-------|
| Test MAE | 0.6557 eV |
| Test RMSE | 0.9686 eV |
| Test R² | 0.6197 |
| Best Val MAE | 0.5956 eV |
| Training Time | 40.1s (V100) |

## Training Dynamics

- Model converges quickly: R² jumps from -0.89 (epoch 1) to 0.69 (epoch 10).
- After epoch 10, validation MAE stops improving while training loss continues to decrease (0.49 -> 0.07), indicating **overfitting**.
- Early stopping triggered at epoch 51; best checkpoint is around epoch 10.
- Overfitting is expected: 80K parameters trained on only 468 samples.

## Interpretation

- **MAE = 0.66 eV**: On average, the model's band gap prediction is off by 0.66 eV. For context, CGCNN on full Materials Project (~46K samples) achieves MAE ~0.39 eV. Our dataset is ~100x smaller, so the gap is expected.
- **R² = 0.62**: The structure-only model explains 62% of the variance in band gap. This confirms that crystal structure carries substantial predictive information, but leaves ~38% unexplained.
- **RMSE (0.97) >> MAE (0.66)**: The large gap suggests a few outlier samples with high prediction error. These are candidates for error analysis (see `test_predictions.csv`).

## Significance for the Project

This baseline establishes the **structure-only performance ceiling** at R²=0.62. If the multimodal fusion model (Exp 3: CGCNN + SciBERT) achieves a meaningfully higher R², it demonstrates that machine-generated text descriptions contain complementary information beyond what graph convolutions can extract.

## Output Files

| File | Description |
|------|-------------|
| `best_model.pt` | Best model checkpoint (selected by val MAE) |
| `results.json` | Test metrics summary |
| `training_history.csv` | Per-epoch train loss, val MAE/RMSE/R², learning rate |
| `test_predictions.csv` | Per-sample test predictions and errors for error analysis |

## Run Command

```bash
cd /home/gridsan/wouyang/MMA/MMA_proj/perovskite-pvk
python scripts/run_exp1_cgcnn.py
```
