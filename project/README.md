# Geo-Text Alignment: Multimodal Perovskite Property Prediction

**Synergizing Crystal Graph and Generated Descriptions for Explainable Perovskite Property Prediction**

Course project for Multimodal AI (MAS.S60 / 6.S985), Spring 2026.

## Key Idea

Materials property prediction typically relies on either graph neural networks (structure) or NLP models (text) in isolation. We combine both: **CGCNN** encodes crystal structure from CIF files, while **SciBERT** encodes machine-generated text descriptions from [Robocrystallographer](https://github.com/hackingmaterials/robocrys). Multimodal fusion consistently outperforms single-modality baselines on band gap prediction of ABO3 perovskite oxides.

## Results Summary

| Model | Modality | MAE (eV) | R² |
|-------|----------|----------|-----|
| CGCNN | Structure | 0.656 | 0.620 |
| SciBERT (frozen) | Text | 1.215 | 0.187 |
| SciBERT (finetune) | Text | 0.751 | 0.554 |
| Concat Fusion | Struct + Text | 0.583 | 0.639 |
| Gated Fusion | Struct + Text | 0.550 | 0.664 |
| **FiLM Fusion** | **Struct + Text** | **0.509** | **0.674** |
| Aligned + FiLM | Struct + Text | 0.588 | 0.658 |

**Best model (FiLM Fusion) reduces MAE by 22% over structure-only baseline**, confirming that machine-generated text descriptions contain information complementary to graph neural networks.

## Dataset

- **Source**: Materials Project API
- **Scope**: 592 ABO3 perovskite oxides (493 with successful text generation)
- **Modalities**: Crystal structure (CIF) + Robocrystallographer text descriptions
- **Targets**: Band gap (eV), Formation energy (eV/atom)
- **Split**: 80/10/10 (train/val/test), deterministic hash-based

Data is included in this repository (`data/`, ~2.6 MB).

## Project Structure

```
perovskite-pvk/
├── configs/                # Experiment configs (YAML)
│   ├── exp0_eda.yaml
│   ├── exp1_cgcnn.yaml
│   ├── exp2_scibert.yaml
│   ├── exp3_fusion.yaml
│   └── exp4_alignment.yaml
├── data/
│   ├── interim/            # Filtered & split tables
│   └── processed/          # Final dataset + CIF files
├── src/
│   ├── data/               # Data loading, crystal graph, text dataset
│   ├── models/             # CGCNN, SciBERT, Fusion (Concat/Gated/FiLM), Alignment
│   ├── evaluation/         # Metrics
│   └── visualization/      # EDA plotting functions
├── scripts/                # Experiment entry points
├── results/                # Outputs (figures, CSVs, JSONs — checkpoints excluded)
│   ├── exp0_eda/           # EDA figures & summary stats
│   ├── exp1_cgcnn/         # CGCNN baseline results
│   ├── exp2_scibert/       # SciBERT baseline results (frozen + finetune)
│   ├── exp3_fusion/        # Fusion results (concat / gated / film)
│   └── exp4_alignment/     # Contrastive alignment results
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Exp 0: Exploratory Data Analysis
python scripts/run_exp0_eda.py

# Exp 1: CGCNN baseline (structure-only)
python scripts/run_exp1_cgcnn.py

# Exp 2: SciBERT baseline (text-only)
python scripts/run_exp2_scibert.py --freeze    # frozen encoder
python scripts/run_exp2_scibert.py             # finetune last 2 layers

# Exp 3: Multimodal Fusion (requires Exp 1 & 2 checkpoints)
python scripts/run_exp3_fusion.py                    # concat
python scripts/run_exp3_fusion.py --fusion gated     # gated
python scripts/run_exp3_fusion.py --fusion film       # FiLM (best)

# Exp 4: Contrastive Alignment
python scripts/run_exp4_align.py                                      # stage 1: alignment
python scripts/run_exp4_regress.py --mode concat --freeze-encoders    # stage 2: frozen regression
python scripts/run_exp4_regress.py --mode film                         # stage 2: FiLM finetune
```

**Note**: SciBERT requires `USE_TF=0` prefix if TensorFlow conflicts with protobuf:
```bash
USE_TF=0 python scripts/run_exp2_scibert.py
```

## Experiments

| Exp | Description | Script | Results |
|-----|-------------|--------|---------|
| 0 | EDA & Visualization | `run_exp0_eda.py` | [results/exp0_eda/](results/figures/exp0_eda/) |
| 1 | CGCNN baseline (structure-only) | `run_exp1_cgcnn.py` | [results/exp1_cgcnn/](results/exp1_cgcnn/) |
| 2 | SciBERT baseline (text-only) | `run_exp2_scibert.py` | [results/exp2_scibert/](results/exp2_scibert/) |
| 3 | Multimodal Fusion (Concat/Gated/FiLM) | `run_exp3_fusion.py` | [results/exp3_fusion/](results/exp3_fusion/) |
| 4 | Contrastive Alignment (CLIP-style) | `run_exp4_align.py` / `run_exp4_regress.py` | [results/exp4_alignment/](results/exp4_alignment/) |

Each experiment folder contains a `README.md` with detailed analysis.

## Key Findings

1. **Multimodal fusion consistently outperforms single-modality baselines.** FiLM Fusion achieves MAE=0.509 eV (vs 0.656 eV structure-only), a 22% improvement.

2. **Fusion strategy matters.** FiLM > Gated > Concat. Allowing text to modulate structure features (FiLM) works better than simple concatenation, suggesting the two modalities play asymmetric roles.

3. **Machine-generated text carries real predictive signal.** SciBERT-finetune alone reaches R²=0.55, proving that Robocrystallographer descriptions encode meaningful structure-property relationships.

4. **Contrastive alignment works but doesn't beat end-to-end fusion on small data.** Alignment achieves 52.8% cross-modal retrieval accuracy, but the two-stage approach (align then regress) underperforms direct fusion due to projection bottleneck and data scarcity.

## Next Steps

- **Cross-Attention Fusion**: Token-level interaction between structure and text
- **Expand dataset**: Relax space group filters (592 → 2000+ samples)
- **Inverse design agent**: Use the fusion model as a surrogate for compositional optimization
- **Explainability**: Attention visualization to identify which text tokens drive predictions
