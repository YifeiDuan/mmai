# Modality Ablation (5-seed ensemble) on band_gap test set

| # | Method | Modalities | n seeds | Single-seed MAE (mean ± std) | Best seed MAE | **Ensemble MAE** | Ensemble R² |
|---|---|---|---|---|---|---|---|
| 1 | Exp 1 CGCNN (struct only) | struct | 5 | 0.6451 ± 0.0232 | 0.6152 | **0.5921** | 0.6815 |
| 2 | Exp 2 SciBERT FT (text only) | text | 5 | 0.6771 ± 0.0224 | 0.6510 | **0.6566** | 0.6235 |
| 3 | Exp 3 FiLM (struct + text) | struct+text | 5 | 0.5379 ± 0.0237 | 0.5125 | **0.5173** | 0.7039 |
| 4 | Exp 5 MT-FiLM (struct + text + MT, FiLM warm) | struct+text+MT (warm) | 5 | 0.5520 ± 0.0158 | 0.5267 | **0.5388** | 0.6906 |
| 5 | Exp 5b MT-FiLM (struct + text + MT, no warm) | struct+text+MT | 5 | 0.5508 ± 0.0562 | 0.4984 | **0.4906** | 0.7385 |
| 6 | Exp 6 CrossAttn (struct + text via cross-attn) | struct+text+MT (cross-attn) | 5 | 0.5742 ± 0.0340 | 0.5369 | **0.5395** | 0.6565 |
| 7 | Exp 7 ALBEF (align then fuse) | struct+text+MT (align init) | 5 | 0.5209 ± 0.0375 | 0.4891 | **0.4990** | 0.7215 |
| 8 | Exp 8 3-modality (struct + text + image) | struct+text+image+MT | 5 | 0.5011 ± 0.0415 | 0.4413 | **0.4862** | 0.7134 |
