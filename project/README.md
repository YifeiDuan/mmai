# Geo-Text-Vision (GTV): Trimodal Perovskite Property Prediction

**Geo-Text-Vision Alignment for Perovskite Property Prediction**

Course project for Multimodal AI (MAS.S60 / 6.S985), Spring 2026.

## Key Idea

We propose **GTV**, a three-modality framework for ABO₃ perovskite band gap prediction. Each crystal is represented from three complementary views:

1. **Structure (CGCNN)** — crystal graph encoding precise local geometry from CIF files
2. **Text (SciBERT)** — machine-generated natural-language descriptions from [Robocrystallographer](https://github.com/hackingmaterials/robocrys), encoding coordination environments and octahedral tilting motifs
3. **Image (ResNet18)** — rendered 224×224 ball-and-stick PNGs capturing global crystal symmetry and topology

Text conditions the structure graph via asymmetric **FiLM** modulation; image features concatenate at the fusion head. Multi-task learning with a metallicity auxiliary head provides free regularization. An **LLM-in-the-loop inverse design agent** (Qwen2.5-VL) reuses the trained ensemble as an oracle to search for perovskites with target band gaps.

## Results Summary (5-seed ensemble on test set)

| Model | Modality | Ens. MAE (eV) ↓ | Seed σ | Δ vs CGCNN |
|-------|----------|-----------------|--------|------------|
| SciBERT (finetune) | Text only | 0.6566 | 0.022 | −10.9% |
| CGCNN | Structure only | 0.5921 | 0.023 | 0% |
| FiLM | Struct + Text | 0.5173 | 0.024 | +12.6% |
| MT-FiLM warm start | Struct + Text | 0.5388 | 0.016 | +9.0% |
| Cross-Attention | Struct + Text | 0.5395 | 0.034 | +8.9% |
| ALBEF init | Struct + Text | 0.4990 | 0.038 | +15.7% |
| MT-FiLM cold start | Struct + Text | 0.4906 | 0.056 | +17.1% |
| **GTV 3-modality** | **Struct + Text + Image** | **0.4862** | 0.042 | **+17.9%** |

**GTV reduces MAE by 18% over the structure-only CGCNN baseline**, and 26% over text-only SciBERT. Semantic content (real embeddings vs. random noise) accounts for ~half the total gain; architectural capacity explains the other half.

### Modality Ablation

| Configuration | Test MAE (eV) | vs. Full Model |
|---------------|---------------|----------------|
| Full model (struct + text + image) | 0.4862 | — |
| Text ablated (struct + rand + image) | 0.5137 | +5.7% |
| Image ablated (struct + text + rand) | 0.5332 | +9.7% |
| Both ablated (struct + rand + rand) | 0.5394 | +10.9% |
| CGCNN only (struct) | 0.5921 | +21.8% |

## Dataset

- **Source**: Materials Project API
- **Scope**: 592 ABO₃ perovskite oxides, filtered to 6 common space groups (Pnma, Pm3m, R3c, I4/mcm, P4mm, R3c), spanning 62 A-site and 70 B-site elements
- **Modalities**: Crystal structure (CIF) + Robocrystallographer text (493/592 successful) + rendered dual-view PNG images
- **Targets**: Band gap (eV, 0–6.09 eV; mean 1.61 ± 1.71 eV), metallicity label (216 metals, 376 non-metals)
- **Split**: 383/54/56 train/val/test, deterministic SHA-256 hash-based

Data is included in this repository (`data/`, ~2.6 MB).

## Project Structure

```
project/
├── configs/                # Experiment configs (YAML)
│   ├── exp0_eda.yaml
│   ├── exp1_cgcnn.yaml
│   ├── exp2_scibert.yaml
│   ├── exp3_fusion.yaml
│   └── exp4_alignment.yaml
├── data/
│   ├── interim/            # Filtered & split tables (parquet)
│   └── processed/          # Final multimodal dataset + CIF files + PNGs
├── src/
│   ├── data/               # Data loading: crystal graph, text, image, fusion datasets
│   ├── models/             # CGCNN, SciBERT, FiLM fusion, cross-attention, image encoder,
│   │                       #   multitask fusion, three-modality GTV, alignment
│   ├── evaluation/         # Metrics, multi-seed ensemble
│   ├── inverse_design/     # Oracle, proposer (Qwen2.5-VL), agent loop, substitution utils
│   └── visualization/      # EDA plotting
├── scripts/                # Experiment entry points
│   ├── run_exp0_eda.py
│   ├── run_exp1_cgcnn.py
│   ├── run_exp2_scibert.py
│   ├── run_exp3_fusion.py          # Concat / Gated / FiLM
│   ├── run_exp4_align.py           # Contrastive alignment (ALBEF)
│   ├── run_exp4_regress.py
│   ├── run_exp5_multitask_film.py  # MT-FiLM (warm/cold start)
│   ├── run_exp6_crossattn.py       # Cross-attention fusion
│   ├── run_exp7_albef_slurm.sh     # ALBEF init + MT-FiLM
│   ├── run_exp8_three_modality.py  # GTV: struct + text + image
│   ├── ensemble_seeds.py           # 5-seed ensemble averaging
│   ├── render_cifs_to_png.py       # Crystal image rendering (ASE)
│   ├── run_inverse_design.py       # LLM-in-the-loop inverse design agent
│   └── build_ablation_table.py
├── results/                # Outputs (figures, CSVs, JSONs — checkpoints excluded)
│   ├── exp1_cgcnn/
│   ├── exp2_scibert/
│   ├── exp3_fusion/
│   ├── exp4_alignment/
│   └── ablation_table.md
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Exp 0: Exploratory Data Analysis
python scripts/run_exp0_eda.py

# Exp 1: CGCNN baseline (structure-only, 5-seed)
python scripts/run_exp1_cgcnn.py

# Exp 2: SciBERT baseline (text-only)
python scripts/run_exp2_scibert.py --freeze    # frozen encoder
python scripts/run_exp2_scibert.py             # finetune last 2 layers

# Exp 3: Bimodal Fusion (Struct + Text)
python scripts/run_exp3_fusion.py                    # concat
python scripts/run_exp3_fusion.py --fusion gated     # gated
python scripts/run_exp3_fusion.py --fusion film      # FiLM

# Exp 4: Contrastive Alignment (ALBEF-style)
python scripts/run_exp4_align.py                     # stage 1: alignment
python scripts/run_exp4_regress.py --mode film       # stage 2: FiLM finetune

# Exp 5/5b: Multi-task FiLM (warm/cold start)
python scripts/run_exp5_multitask_film.py --init warm
python scripts/run_exp5_multitask_film.py --init cold

# Exp 6: Cross-attention Fusion
python scripts/run_exp6_crossattn.py

# Exp 8: GTV Trimodal (best model)
python scripts/run_exp8_three_modality.py

# Ensemble 5 seeds
python scripts/ensemble_seeds.py --exp exp8

# Inverse Design Agent
python scripts/run_inverse_design.py --target 2.0
```

**Note**: SciBERT requires `USE_TF=0` prefix if TensorFlow conflicts with protobuf:
```bash
USE_TF=0 python scripts/run_exp2_scibert.py
```

## Experiments

| Exp | Description | Key Result |
|-----|-------------|------------|
| 0 | EDA & Visualization | Band gap bimodal (metals at 0 eV); SciBERT t-SNE shows partial property clustering |
| 1 | CGCNN baseline (structure-only) | 0.5921 eV MAE |
| 2 | SciBERT baseline (text-only, finetune) | 0.6566 eV MAE |
| 3 | FiLM fusion (Struct + Text) | 0.5173 eV MAE (+12.6%) |
| 4 | Contrastive ALBEF alignment | Reduces σ from 0.056 → 0.038 without mean gain |
| 5 (warm) | Multi-task FiLM, warm start | 0.5388 eV — warm start is a saddle point for joint loss |
| 5b (cold) | Multi-task FiLM, cold start | **0.4906 eV** (+17.1%) |
| 6 | Cross-attention fusion | 0.5395 eV — overfits at this dataset scale |
| 7 | ALBEF init + MT-FiLM | 0.4990 eV, σ=0.038 — more stable deployment |
| 8 | **GTV: Struct + Text + Image** | **0.4862 eV** (+17.9%), best overall |

## Key Findings

1. **Trimodal fusion consistently outperforms all unimodal baselines.** GTV achieves 18% MAE improvement over structure-only CGCNN and 26% over text-only SciBERT.

2. **Asymmetric FiLM beats cross-attention at this scale.** Text conditioning the structure (not vice versa) reflects the correct inductive bias: geometry is ground truth, text is semantic gloss. Cross-attention's extra capacity overfits on the 383-sample training set.

3. **Cold start strictly outperforms warm start for multi-task training.** A checkpoint optimal under single-task loss is a saddle under the joint loss; 9.5% degradation from warm start.

4. **Vision modality contributes beyond architectural capacity.** The image ablation incurs a 9.7% MAE penalty despite using a frozen ImageNet prior — the visual branch captures global symmetry cues (space group, unit cell geometry) not encoded by graph message passing or text descriptions.

5. **Semantic content and architectural capacity both matter.** Random-noise ablation decomposes the total gain: 8.9% from extra parameters alone, 10.9% from real semantic embeddings replacing noise.

6. **5-seed ensembling is essential on small test sets.** Single-seed σ ≈ 0.05 eV on the 56-sample test set can produce false SoTA claims; ensemble averaging closes a 5.6% discrepancy.

7. **ALBEF initialization reduces variance without improving mean.** σ drops from 0.056 → 0.038 (Exp 5b → Exp 7), yielding a more reliable deployment model even when mean MAE is 2% worse.

## Inverse Design Agent

The LLM-in-the-loop agent reuses the 5-seed MT-FiLM ensemble as a forward oracle. Qwen2.5-VL-3B-Instruct proposes (A, B) element substitutions in JSON; the oracle scores candidates in ~55 seconds for 75 evaluations across three band gap targets (1.0, 2.0, 3.0 eV) without any retraining.

| Target (eV) | Best Candidate | Pred ± std (eV) | \|Error\| (eV) |
|-------------|----------------|-----------------|----------------|
| 1.0 | CeGeO₃ (mp-19269) | 0.787 ± 0.205 | 0.213 |
| 2.0 | NaPbO₃ (mp-756464) | 1.925 ± 0.325 | **0.075** |
| 3.0 | LaNbO₃ (mp-1180739) | 2.573 ± 0.537 | 0.427 |

## Future Directions

1. **Scale the dataset**: Relax space-group filters (592 → 2000+ samples); include double perovskites (A₂BB'O₆)
2. **Domain-specific visual encoder**: Replace frozen ImageNet-ResNet18 with SE(3)-equivariant 3D encoder
3. **Smarter inverse loop**: Crystal generative models + DFT-in-the-loop validation
4. **Multi-property prediction**: Jointly predict formation energy, stability, and defect tolerance
