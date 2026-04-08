# Homework 1: Multimodal Data Preparation for Perovskite Property Prediction

## Project Goal

Predict key physical properties (band gap, formation energy) of ABO₃ perovskite materials by jointly leveraging crystal structure data and auto-generated textual descriptions.

## Tasks and Implementations

### Data Extraction
Implemented and ran the full data pipeline:
- **CIF extraction**: Queried the Materials Project API (`mp-api`) and filtered candidates by stoichiometry, perovskite space groups (Pm-3m, Pnma, R-3c, I4/mcm, P4mm, R3c), and energy above hull < 0.1 eV, yielding 592 clean samples split 468/63/61 (train/val/test).
- **Text generation**: Used Robocrystallographer to generate natural-language structural descriptions for each CIF. ~16.7% of structures failed due to disorder or partial occupancies; fast mode (no mineral prototype matching) was used for stability.

### Visualization
Explored the dataset with multiple visualization techniques:
- **t-SNE** of combined categorical (space group, A/B sites) and numerical (band gap, formation energy, energy above hull) features — full dataset and random sample subsets.
- **Bar plot** of space group distribution.
- **Violin and scatter plots** of target property distributions, stratified by space group.
- **t-SNE of text embeddings** (via `all-MiniLM-L6-v2` SentenceTransformer) colored by space group, revealing that structural similarity is partially captured in the text latent space.

### Evaluation Metrics
Selected and implemented regression metrics for model evaluation:
- **MAE** — physically interpretable (eV units), consistent with Matbench leaderboard convention, robust to outliers.
- **R²** — scale-invariant, useful for comparing models across tasks.
- **RMSE** — penalizes large errors more heavily, important for catching catastrophic mispredictions.
- Rejected MAPE (undefined at zero, unstable for near-zero targets), MSE (reserved for training loss), and classification metrics (not applicable to regression).

## Key Takeaways
- Building a clean multimodal materials dataset requires significant domain-specific filtering; raw API data is noisy and structurally inconsistent.
- Text auto-generation from structure is imperfect (~17% failure rate) and introduces a missing-modality challenge that must be handled at training time.
- t-SNE of text embeddings shows partial clustering by space group, suggesting the text modality captures meaningful structural patterns and is not simply redundant with composition.
- MAE + R² + RMSE together give a comprehensive picture of regression performance that aligns with prior literature and supports both interpretability and sensitivity to outliers.
