# Homework 3: Vision-Language Models and LoRA Fine-Tuning

## Project Goal

Explore and fine-tune a Vision-Language Model (VLM) for a domain-specific task: classifying the crystal space group of ABO₃ perovskite materials from rendered unit cell images, using `Qwen2.5-VL-3B-Instruct`.

## Tasks and Implementations

### Dataset Preparation
Converted crystal structure CIF files from the HW1 dataset into 2D unit cell images for VLM input. Applied a **stratified 80/20 train/test split** by space group to preserve class distribution across the five space groups (Pm-3m, Pnma, R-3c, I4/mcm, P4mm), since Pnma accounts for ~1/3 of samples while the rarest group covers only ~5%.

### Baseline Inference (Pretrained Model)
Ran the pretrained `Qwen2.5-VL-3B-Instruct` (no fine-tuning) on 4 held-out crystal images, testing 6 question phrasings (direct query, domain-specified, 5-option MCQ, 2-option MCQ, with/without non-greedy decoding):
- Without options: model generates arbitrary space group symbols (0% accuracy).
- With 2 options + greedy decoding: uniform answer across all images (trivially 50%).
- **Best**: 2-option question + non-greedy decoding → **50% accuracy**, but still not conditioned on image content.

### Prompt Engineering
Tested 8 system prompt configurations (minimalist domain spec, step-by-step reasoning, with/without options, few-shot examples):
- **Best result**: 25% accuracy with step-by-step instruction + 5 options.
- Few-shot examples did not help — model picked the first listed option or generated random answers.
- Conclusion: prompt engineering alone is insufficient for this highly domain-specific task; fine-tuning is necessary.

### LoRA Fine-Tuning
Fine-tuned `Qwen2.5-VL-3B-Instruct` with LoRA on perovskite crystal images, sweeping 11 hyperparameter configurations across epochs, learning rate, batch size, gradient accumulation, sequence length, image resolution, and LoRA rank/alpha/dropout/target modules.

**Best configuration** (val loss = 0.5017):
```
NUM_EPOCHS=15, LR=1e-3, BSZ_PER_DEV=1, GRAD_ACCUM=1
MAX_SEQ_LEN=512, SHORTEST_EDGE=288
LORA_R=4, LORA_ALPHA=8, LORA_DROPOUT=0.05
LORA_TARGET=["q_proj", "k_proj", "v_proj", "o_proj"]
```

Most impactful hyperparameters: **batch size** (BSZ=10 → val loss >1.59, far fewer update steps) and **learning rate** (lr=1e-5 → very slow convergence even at 40 epochs). Image resolution increase (288→448) slightly degraded performance.

### Post-Training Evaluation
Compared pretrained vs. fine-tuned model on the same held-out images:

| Setting | Accuracy |
|---|---|
| Pretrained, no options, greedy | 0% (random symbols) |
| Pretrained, 5 options, greedy | 0% (uniform wrong answer) |
| Fine-tuned, no options, greedy | 50% (all predict Pnma) |
| **Fine-tuned, no options, non-greedy (temp=1.15)** | **50% (diverse predictions)** |
| Fine-tuned, 5 options, non-greedy | 50% |

- Fine-tuning eliminated random symbol generation; the model now produces valid space group outputs.
- The model learned to ingest visual features (correct on training-set minority groups), but the imbalanced dataset introduced a strong **Pnma bias** (P(Pnma)≈0.59 for a Pm-3m sample).
- **Non-greedy decoding** (temperature > 1) partially corrected the majority-class bias by flattening the output distribution.

## Key Takeaways
- Pretrained VLMs cannot handle highly domain-specific visual tasks (perovskite crystallography) without fine-tuning; prompt engineering alone yields at most marginal gains.
- LoRA fine-tuning significantly improves the model's ability to process domain-specific images, but class imbalance in small datasets introduces label bias toward the majority class.
- Non-greedy decoding is an effective lightweight correction for majority-class bias after fine-tuning, without requiring retraining.
- Batch size and learning rate are the most critical training hyperparameters; larger batches reduce effective update frequency and hurt convergence far more than LoRA rank or dropout choices.
