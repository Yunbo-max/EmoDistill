# EmoDistill — EMNLP Submission Code (CRAD)

Minimal release of **EmoDistill**: an offline-RL pipeline that distills emotional
negotiation skills into a 7B small language model creditor.
Three stages: **IQL** emotion selector → **LoRA-SFT** expression imitation →
**JPO** (Judge Policy Optimization) refinement.

This repo reproduces the headline **CRAD** (debt-collection) row of the paper.

---

## 1. Install

```bash
pip install -r requirements.txt
cp .env.template .env
# then edit .env and fill DASHSCOPE_API_KEYS=sk-...
```

Tested on Python 3.10, PyTorch 2.4, peft 0.12, transformers 4.44.

## 2. Hardware

1× NVIDIA RTX 4090 (24 GB) is sufficient for the full pipeline.
Qwen-Plus debtor/judge are API-served (no extra GPU).

## 3. End-to-end on CRAD (4 stages)

### Stage A — Stochastic emotion sweep (~30 min, API only)
Generate the offline training corpus: 20 held-out scenarios × 28 GoEmotions × 3 iterations.

```bash
python -m experiments.run_random_emotion_sweep \
    --dataset_type debt \
    --scenarios 20 --offset 80 --iterations 3 \
    --max_dialog_len 30 \
    --model_creditor qwen-plus --model_debtor qwen-plus \
    --concurrency 6 --seed 42 \
    --out_dir results/sweep_crad
```

Output: `results/sweep_crad/debt_<ts>/random_sweep_<ts>.json`

### Stage B — Per-turn LLM judge (~10 min, API only)
Add a 1–10 judge score to every creditor utterance. This becomes the JPO reward.

```bash
python -m EmoDistill.judge_scorer_v2 \
    --in_json  results/sweep_crad/debt_<ts>/random_sweep_<ts>.json \
    --out_json results/sweep_crad/debt_<ts>/random_sweep_judged.json \
    --judge_model qwen-plus --concurrency 6
```

### Stage C — Train the three EmoDistill components (~50 min, 1 GPU)

**C.1 — IQL value network** on per-turn judge rewards (5–8 min)
```bash
python -m experiments.run_iql \
    --dataset_path results/sweep_crad/debt_<ts>/random_sweep_judged.json \
    --out_dir results/iql_crad \
    --n_steps 20000 --batch_size 256 \
    --hidden_dim 256 --lr 3e-4 \
    --expectile 0.7 --beta 3.0 \
    --normalize_reward --seed 42
```

**C.2 — LoRA-SFT** on the top-K% advantage-filtered turns (~15 min)
```bash
python -m experiments.run_lora_train \
    --sweep_dir results/sweep_crad/debt_<ts> \
    --scenario_type debt --top_k_percent 0.10 \
    --base_model Qwen/Qwen2.5-7B-Instruct \
    --lora_r 16 --lora_alpha 32 \
    --epochs 1 --batch_size 1 --grad_accum 16 \
    --lr 1e-4 --max_seq_len 1024 \
    --out_dir results/sft_crad
```

**C.3 — JPO refinement** on top of the SFT init (~45 min)
```bash
python -m experiments.run_grpo_train \
    --sweep_dir results/sweep_crad/debt_<ts> \
    --scenario_type debt \
    --reward_field judge_score \
    --base_model Qwen/Qwen2.5-7B-Instruct \
    --sft_adapter results/sft_crad/run_<ts>/adapter_final \
    --reference_mode init_snapshot \
    --kl_beta 0.04 --clip_eps 0.2 \
    --epochs 1 --batch_size 1 --grad_accum 16 \
    --lr 5e-6 --warmup_ratio 0.03 \
    --max_seq_len 1024 \
    --output_dir results/jpo_crad
```

### Stage D — Held-out evaluation (~12 min)
20 scenarios × 1 iter; reports Success / Outcomes / Utility / Rounds.

```bash
python -m experiments.run_hierarchical_eval \
    --iql_ckpt    results/iql_crad/iql_<ts>.pt \
    --lora_adapter results/jpo_crad/run_<ts>/adapter_final \
    --base_model Qwen/Qwen2.5-7B-Instruct \
    --dataset_type debt \
    --scenarios 20 --iterations 1 --offset 80 \
    --debtor_model qwen-plus \
    --max_dialog_len 30 --concurrency 4 \
    --seed 42 \
    --out_dir results/eval_crad
```

Final headline reported in `results/eval_crad/debt_<ts>/hierarchical_eval_<ts>.json`:
fields `success_rate`, `mean_savings_ratio`, `mean_rounds`, and per-episode breakdown.

---

## 4. Cost estimate (Qwen-Plus, May 2026 list price)

| Stage | Estimate |
|---|---|
| A. Sweep (~56k API calls) | ~$23 |
| B. Judge (~28k API calls) | ~$7 |
| C. Training (GPU only) | $0 |
| D. Eval (~3.6k debtor calls) | ~$1 |
| **Total CRAD** | **~$31** |

DashScope's 50%-off Batch tier halves stages A+B (~$15 total).

## 5. Repo layout

```
.
├── README.md
├── requirements.txt
├── .env.template
├── data/
│   └── credit_recovery_scenarios.csv          # 100 CRAD scenarios
├── EmoDistill/                                # Core library
│   ├── grpo_train.py    JPO inner trainer
│   ├── sft_train.py     SFT inner trainer
│   ├── iql.py           IQL policy + value net
│   ├── judge_scorer_v2.py
│   ├── dashscope_wrapper.py
│   ├── lora_negotiator.py / negotiator_new.py
│   ├── reward_v4.py     time-weighted shaped reward
│   ├── emotions.py      28-emotion GoEmotions taxonomy
│   ├── lora_data.py / grpo_data.py            # training-data extractors
│   └── ...
├── baselines/                                    # IQL policy + vanilla baseline
├── experiments/                               # Entry-point scripts
│   ├── run_random_emotion_sweep.py
│   ├── run_iql.py
│   ├── run_lora_train.py
│   ├── run_grpo_train.py
│   └── run_hierarchical_eval.py
├── llm/                                       # Prompt templates
└── utils/                                     # Scenario preprocessing
```

## 6. Citation

```bibtex
@inproceedings{emodistill2026,
  title  = {EmoDistill: Offline Emotion Skill Distillation for Language Model Agents},
  author = {Anonymous},
  booktitle = {EMNLP},
  year   = {2026}
}
```
