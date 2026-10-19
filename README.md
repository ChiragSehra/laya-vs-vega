# laya-vega — an independent head-to-head of two open-source "System 1" decision engines

This project benchmarks **[vegaml](https://github.com/NandhaKishorM/vegaml)** against
**[laya](https://github.com/NandhaKishorM/laya)** — two open-source alternatives to TypeSafe's
closed **Jev** decision model, both by the same author ([NandhaKishorM](https://github.com/NandhaKishorM)) —
on a third-party dataset neither has published numbers on: the
**[Apache Jira Issue Tracking Dataset](https://zenodo.org/records/5665896)** (CC-BY-4.0).

Everything runs **locally on an Apple-silicon Mac (MPS), CPU fallback, no GPU rental, no API keys.**

## Why

Both projects publish benchmark tables — but only against each other's siblings and against closed
Jev. To our knowledge there is **no published independent comparison of laya vs vegaml on the same
splits**. This repo is one. It is small (n=100 per cell) but fully scripted and reproducible.

## What the two engines are

| | [laya](https://github.com/NandhaKishorM/laya) | [vegaml](https://github.com/NandhaKishorM/vegaml) |
|---|---|---|
| idea | non-autoregressive typed decisions over a BERT-style encoder (ModernBERT / mmBERT), trained with RL against proper scoring rules | a frozen causal LM is *perception only*; a 14.3M "physics engine" settles a particle in a 64-d energy landscape where each candidate answer is a Gaussian well |
| readout | one forward pass, logits → typed answers | damped Hamiltonian dynamics → settling well + physics-conditioned temperature |
| extras | router (3 checkpoints, 100+ languages), shortlist/tournament for many options | calibrated probabilities, conformal sets, abstain + unbound flags, 73k-token context, images |
| licence | Apache 2.0 | Apache 2.0 |

## Datasets

| file | source | notes |
|---|---|---|
| `data/jira_issues.parquet` | [Zenodo 5665896](https://zenodo.org/records/5665896) | 1,013,964 Apache Jira issues, 647 projects, streamed from BSON (`parse_jira.py`), MD5-verified |
| `data/support_tickets.parquet` | [Tobi-Bueck/customer-support-tickets](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets) | 61,765 tickets, en+de |
| `data/emotion_*.parquet` | [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion) | real tweets, 6 emotions |

## Results

### Main: Apache Jira — `issuetype` (6-way choice) and `priority` (5-level ordinal)

834,559 usable issues. **Issue-disjoint split**: SHA-1-bucketed train/eval, zero overlap.
**Project-disjoint split**: train and eval issues come from **different Apache projects** —
the model never sees a ticket from the eval team during fitting.

| issuetype accuracy | issue-disjoint | project-disjoint |
|---|---|---|
| **vegaml** (fit on 120 labels, `mode="auto"`) | **0.71** | **0.75** |
| laya (zero-shot, English checkpoint) | 0.64 | 0.63 |

| priority accuracy (±1 level = partial credit) | issue-disjoint | project-disjoint |
|---|---|---|
| **vegaml** | **0.92** (0.64 exact) | **0.90** (0.55 exact) |
| laya | 0.91 (0.44 exact) | 0.87 (0.48 exact) |

### Earlier runs (support tickets + emotion) — `results/summary.json`

- Ticket triage, zero-shot: type 0.70, priority±1 0.93, queue 0.18 — and the **abstain flag fired
  on 58/60 of the queue decisions it was bad at** and only 8/60 of the type decisions it was good
  at. The calibration machinery refuses to stand behind exactly the answers that deserve refusal.
- Fitting a head on 120 labelled tickets took queue accuracy 0.18 → 0.70.
- Emotion on real tweets: 0.42 zero-shot (vegaml's own docs report 0.313 for the base engine).

### Cost side

| | laya | vegaml |
|---|---|---|
| setup | load checkpoint (~86 s) | load checkpoint (~167 s first run) |
| 100 two-question evals | 53–69 s, **no training** | ~430 s fit (120 examples) + eval |
| when to pick | zero labels, fastest path | you have a few hundred labels and want exact ordinal sense + abstain |

## Honest caveats

1. **n=100 per cell** → roughly ±5 pp; treat differences under ~10 pp as suggestive.
2. **Not parity**: vegaml was fitted on 120 labels, laya ran zero-shot. A `laya.finetune` arm is
   the obvious follow-up (his repo ships an MPS script).
3. Raw Jira data includes real-world **spam issues** (pharma etc.); we did not filter them.
4. vegaml's TTT readout drops the abstain flag when a fitted head serves the answer — his own
   documented trade-off, observed here too.

## Reproduce

```bash
./.venv/bin/python -m venv .venv            # Python 3.11
.venv/bin/pip install vegaml laya pandas pyarrow pymongo torch torchvision

# 1. get + parse the Jira dump (765 MB, MD5-checked)
curl -L -o data/issues.bson.gz https://zenodo.org/api/records/5665896/files/issues.bson.gz/content
.venv/bin/python parse_jira.py              # md5 must be c285290f3b4e0843d6cf3ea05da74a53

# 2. head-to-head, both engines, both splits
HF_HOME=$PWD/.hf-cache .venv/bin/python jira_h2h.py --engine vegaml --n-train 120 --n-eval 100
HF_HOME=$PWD/.hf-cache .venv/bin/python jira_h2h.py --engine laya  --n-eval 100

# 3. the earlier demo (tickets + emotion)
HF_HOME=$PWD/.hf-cache .venv/bin/python demo_triage.py --n-tickets 60 --n-emotion 100
HF_HOME=$PWD/.hf-cache .venv/bin/python fit_eval.py --train-n 120 --eval-n 60
```

All result JSON lives in `results/`.

## Files

| file | what it does |
|---|---|
| [parse_jira.py](parse_jira.py) | streams BSON → flat parquet, no MongoDB needed |
| [jira_h2h.py](jira_h2h.py) | clean-split vegaml/laya benchmark on Jira |
| [demo_triage.py](demo_triage.py) | support-ticket triage + emotion benchmark |
| [fit_eval.py](fit_eval.py) | before/after `vegaml.fit()` on the weak decision |
| [results/](results/) | every number in this README, as JSON |

## Credits & licence

All model work is [NandhaKishorM](https://github.com/NandhaKishorM)'s — [laya](https://github.com/NandhaKishorM/laya)
and [vegaml](https://github.com/NandhaKishorM/vegaml), both Apache 2.0. Dataset by Diamantopoulos &
Symeonidis, CC-BY-4.0. The benchmark code and numbers in this repo: MIT — independent evaluation
is exactly what open decision models are for.
