# Homework 4: Reinforcement Learning for Vision-Language Models (GRPO)

## Project Goal

Explore **Group Relative Policy Optimization (GRPO)** for aligning a Vision-Language Model (VLM) with reward signals, applied to the same perovskite space group classification task from HW3. Train `Qwen3-VL-2B-Instruct` using RL instead of supervised fine-tuning (SFT), and compare the two paradigms.

## Tasks and Implementations

### Part 1: Reading & Reflection

Analyzed two key papers (DeepSeekMath and The Illustrated GRPO) and answered conceptual questions on:
- **GRPO vs PPO**: GRPO replaces the learned critic with a group-relative baseline, halving memory cost. The group mean reward serves as the baseline; when all completions score equally, advantages are zero and no update occurs. Tradeoff: higher variance baseline and costly group sampling.
- **Reward design**: Rule-based rewards are interpretable and hackable in predictable ways (correct answer, wrong reasoning); learned reward models are richer but susceptible to distribution shift and annotation noise. KL divergence to a reference model regularizes both.
- **SFT vs. GRPO**: SFT maximizes log-likelihood of gold answers (imitates data distribution); GRPO generates its own completions and reinforces relatively better ones. GRPO is preferred when rewards are verifiable and the model must explore beyond the training distribution.

### Part 2: Implementing GRPO Advantage Computation

Implemented `compute_grpo_advantage(rewards, group_ids, response_mask)` from scratch:
- Groups rewards by prompt ID, computes group mean and std
- Normalizes per-sample advantage: `A_i = (r_i − mean) / (std + ε)`
- Supports Dr. GRPO variant (mean-only, no std scaling)
- Broadcasts scalar advantages to token-level via `response_mask`
- Edge cases: single-sample groups get zero advantage; uniform rewards give zero gradient

All 5 unit tests passed (basic grouping, single-sample, masking, Dr. GRPO variant, uniform rewards).

### Part 3: Reward Functions

Implemented two rule-based reward functions for the `GRPOTrainer` API:
- **`accuracy_reward`**: 1.0 if extracted answer matches ground truth, 0.0 otherwise
- **`format_reward`**: 1.0 if completion contains `Answer:` marker, 0.0 otherwise

The format reward provides a learning signal even before accuracy improves, preventing zero-reward collapse early in training.

### Part 4: GRPO Training on Perovskite Images

Reused the perovskite crystal image dataset from HW3 (60 samples, 5 space groups: Pm-3m, Pnma, R-3c, I4/mcm, P4mm). The model must classify space group from a rendered unit cell image.

**Hyperparameter exploration** (8 configurations, sweeping `num_generations`, `max_completion_length`, `max_steps`, `beta`):

| Configuration | Key Change | Observation |
|---|---|---|
| Default | G=2, len=256, steps=100 | Low reward signals; model rarely hits `Answer:` |
| MAX_COMPLETION_LENGTH++ | len=1024 | Necessary for multi-step reasoning to reach `Answer:` |
| NUM_GENERATIONS++ | G=4 | More diverse completions → better baseline estimate |
| MAX_STEPS-- | steps=20 | Insufficient training; no convergence |
| w. Options, Default | G=2, len=256 + 5 options in prompt | Format reward emerges first |
| **w. Options, len=1024** | G=2, len=1024 + options | Best; format compliance saturates, accuracy improves |
| w. Options, G=4 | G=4, len=512 + options | Good; more stable reward curves |
| w. Options, beta=0.1 | KL penalty enabled | Slightly slower convergence but less drift |

**Best configuration**:

| Parameter | Value |
|-----------|-------|
| `num_generations` | 4 |
| `max_completion_length` | 512 |
| `max_steps` | 50 |
| `learning_rate` | 1e-5 |
| `epsilon` | 0.2 |
| `temperature` | 0.9 |
| `beta` | 0.0 |

Model: `Qwen3-VL-2B-Instruct` with LoRA (r=16, α=32, target: q/k/v/o_proj).

### Part 5: Post-Training Evaluation

Evaluated the GRPO-trained model on 4 held-out perovskite images:
- **Format compliance**: 100% (vs. 0% for pretrained base model) — the model reliably outputs `Answer:` and step-by-step reasoning
- **Accuracy**: ~50%, comparable to the best SFT-trained model from HW3
- Minority space groups show improved accuracy relative to SFT, which was biased toward the Pnma majority class

## Key Takeaways

- **Options in prompts are essential** for domain-specific tasks. Without a constrained answer set, the model almost never produces a correct answer in a zero-reward regime, so no learning signal can emerge.
- **`max_completion_length` must be large enough** for the reasoning chain to reach the `Answer:` marker; the default 256 tokens caused training to stall with zero format reward.
- **Increasing `num_generations`** (G=4 vs G=2) provides a better group baseline estimate and improves reward variance, leading to more stable updates.
- **GRPO converges slower than SFT** on this highly domain-specific task because it must explore correct answers before any reward signal appears, whereas SFT sees ground truth immediately.
- **GRPO shows less majority-class bias than SFT**: RL reinforces relatively correct completions regardless of class frequency, partially correcting the Pnma bias introduced by SFT's imitating the imbalanced training distribution.
