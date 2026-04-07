# Exp 3: Multimodal Fusion (CGCNN + SciBERT)

## Overview

Multimodal fusion for band gap prediction, combining crystal structure (CGCNN) and machine-generated text descriptions (SciBERT). Three fusion strategies are compared: Concatenation, Gated Fusion, and FiLM (Feature-wise Linear Modulation).

All fusion models initialize encoders from pretrained Exp 1 (CGCNN) and Exp 2 (SciBERT-finetune) checkpoints, then fine-tune end-to-end.

## Data Split (text_ok only)

| Split | Samples |
|-------|---------|
| Train | 383 |
| Val   | 54 |
| Test  | 56 |

## Fusion Strategies

### 1. Concatenation (baseline)
```
h_struct (128-d) ──┐
                   ├── Concat (896-d) → MLP → prediction
h_text   (768-d) ──┘
```
Simplest approach. No interaction between modalities before the MLP head.

### 2. Gated Fusion
```
h_s' = LN(proj(h_struct))   ──┐
                               ├── g = σ(W·[h_s'; h_t'])
h_t' = LN(proj(h_text))     ──┘
                               
fused = LN(g·h_s' + (1-g)·h_t') + 0.5·(h_s'+h_t')  → MLP → prediction
```
Learns a per-sample, per-dimension gate to weight the two modalities. Includes LayerNorm for stable training, gate bias initialized to 0 (balanced start), and a residual skip connection.

### 3. FiLM (Feature-wise Linear Modulation)
```
h_s' = LN(proj(h_struct))
h_t' = LN(proj(h_text))

(γ, β) = FiLM_gen(h_text)        # text generates modulation params
h_mod  = LN(γ·h_s' + β) + h_s'   # modulate structure + residual

fused = [h_mod; h_t'] → MLP → prediction
```
Text acts as a "controller" that rescales and shifts structure features. Intuition: text provides high-level semantic context ("distorted perovskite", "corner-sharing octahedra") that adjusts the interpretation of precise structural features. Includes:
- Two-layer FiLM generator with zero initialization (starts as identity transform)
- Residual connection to preserve original structure signal
- Text shortcut: h_text is also directly concatenated to the head input

## Results

### Full Comparison (all experiments)

| Model | Modality | MAE (eV) ↓ | RMSE (eV) ↓ | R² ↑ | Training Time |
|-------|----------|-----------|------------|------|---------------|
| CGCNN (Exp 1) | Structure | 0.6557 | 0.9686 | 0.6197 | 40.1s |
| SciBERT-frozen (Exp 2) | Text | 1.2152 | 1.4326 | 0.1867 | 132.6s |
| SciBERT-finetune (Exp 2) | Text | 0.7507 | 1.0609 | 0.5539 | 71.2s |
| Concat Fusion | Struct+Text | 0.5834 | 0.9546 | 0.6389 | 173.2s |
| Gated Fusion | Struct+Text | 0.5500 | 0.9203 | 0.6644 | 147.7s |
| **FiLM Fusion** | **Struct+Text** | **0.5094** | **0.9067** | **0.6742** | 229.1s |

### Key Findings

1. **All fusion models outperform both single-modality baselines**, confirming that structure and text encode complementary information about band gap.

2. **FiLM is the best fusion strategy**: MAE=0.509 eV, a **22% reduction** over CGCNN-only (0.656 eV) and a **32% reduction** over SciBERT-finetune (0.751 eV). R² improves from 0.620 to 0.674.

3. **Fusion strategies ranked**: FiLM > Gated > Concat. More expressive fusion mechanisms that allow modality interaction (rather than simple concatenation) consistently perform better.

4. **FiLM's asymmetric design works well**: Using text to modulate structure (rather than the reverse) aligns with the intuition that text provides high-level semantic context while structure provides precise geometric detail. The text "interprets" the structure.

### Training Dynamics

| Fusion | Best Val MAE | Early Stop Epoch | Stability |
|--------|-------------|-----------------|-----------|
| Concat | 0.512 | 58 | Stable, steady improvement |
| Gated  | 0.543 | 52 | Some oscillation, LayerNorm helps |
| FiLM   | 0.505 | 83 | Most oscillation early, but trains longest and finds best solution |

- FiLM trains the longest (83 epochs) despite having the most parameters, suggesting the zero-initialized FiLM generator starts as identity and gradually learns meaningful modulation.
- All models show some val-test gap due to small dataset size (54 val / 56 test samples).

### Ablation: Pretrained vs From-scratch

All results above use pretrained encoder initialization. Training from scratch (`--no-pretrained`) is expected to perform worse given the small dataset, as the encoders would need to learn both representation and fusion simultaneously.

## Significance for the Project

These results validate the core hypothesis:

> Machine-generated structural text descriptions (Robocrystallographer) contain information complementary to what graph neural networks extract from crystal structures, and multimodal fusion consistently improves band gap prediction.

The progression from simple concatenation to FiLM suggests that **how** modalities are fused matters — not just whether they are combined. This motivates further exploration of cross-attention and contrastive alignment strategies in the final report.

## Output Files

```
results/exp3_fusion/
├── README.md
├── concat/
│   ├── best_model.pt
│   ├── results.json
│   ├── training_history.csv
│   └── test_predictions.csv
├── gated/
│   ├── best_model.pt
│   ├── results.json
│   ├── training_history.csv
│   └── test_predictions.csv
└── film/
    ├── best_model.pt
    ├── results.json
    ├── training_history.csv
    └── test_predictions.csv
```

## Run Commands

```bash
cd /home/gridsan/wouyang/MMA/MMA_proj/perovskite-pvk

# Concat (baseline)
python scripts/run_exp3_fusion.py

# Gated
python scripts/run_exp3_fusion.py --fusion gated

# FiLM
python scripts/run_exp3_fusion.py --fusion film

# Without pretrained encoders (ablation)
python scripts/run_exp3_fusion.py --fusion film --no-pretrained
```
