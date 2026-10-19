"""Practical demo of NandhaKishorM's vegaml on real public datasets.

1. Support-ticket triage (Tobi-Bueck/customer-support-tickets):
   for each ticket decide queue (10-way), priority (5-level), type (4-way)
   in one call, with calibrated probabilities and abstain flags.
2. Emotion benchmark (dair-ai/emotion, real tweets): 6-way accuracy vs labels,
   comparable to the number in vegaml's own fine-tuning docs (0.542 smoke test).

Usage: .venv/bin/python demo_triage.py [--n-tickets 60] [--n-emotion 100]
"""
import argparse
import json
import os
import pathlib
import time

import pandas as pd

ROOT = pathlib.Path(__file__).parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

QUEUES = [
    "Technical Support", "Product Support", "Customer Service", "IT Support",
    "Billing and Payments", "Returns and Exchanges",
    "Service Outages and Maintenance", "Sales and Pre-Sales",
    "Human Resources", "General Inquiry",
]
PRIORITIES = ["very_low", "low", "medium", "high", "critical"]
TYPES = ["Incident", "Request", "Problem", "Change"]
EMOTIONS = ["sadness", "joy", "love", "anger", "fear", "surprise"]

QUESTIONS = {
    "queue": {
        "type": "choice",
        "instructions": "Which support queue should handle this ticket?",
        "criteria": {
            "Technical Support": "product faults, bugs, configuration problems",
            "Product Support": "product features, usage questions, compatibility",
            "Customer Service": "general account and service matters, complaints",
            "IT Support": "internal IT, access, credentials, infrastructure",
            "Billing and Payments": "invoices, charges, refunds, subscriptions",
            "Returns and Exchanges": "returns, exchanges, replacements, shipping",
            "Service Outages and Maintenance": "outages, downtime, scheduled maintenance",
            "Sales and Pre-Sales": "purchasing, pricing, product comparisons",
            "Human Resources": "staffing, internal HR matters",
            "General Inquiry": "anything that fits none of the other queues",
        },
    },
    "priority": {
        "type": "score",
        "instructions": "How urgent is this ticket?",
        "criteria": PRIORITIES,
    },
    "type": {
        "type": "choice",
        "instructions": "What kind of ticket is this?",
        "criteria": {
            "Incident": "a sudden disruption or outage affecting service",
            "Request": "an inquiry, feature request or ask for help",
            "Problem": "a recurring fault or error the user faces",
            "Change": "a request to modify configuration, plan or settings",
        },
    },
    "emotion": {
        "type": "choice",
        "instructions": "Which emotion does this text express?",
        "criteria": {e: e for e in EMOTIONS},
    },
}


def run_tickets(v, n):
    df = pd.read_parquet(DATA / "support_tickets.parquet")
    df = df[(df.language == "en") & df.body.notna()].reset_index(drop=True)
    # stratify sample by queue
    per = max(1, n // df.queue.nunique())
    sample = df.groupby("queue").sample(n=per, random_state=7).reset_index(drop=True)
    rows, abstains = [], {"queue": 0, "priority": 0, "type": 0}
    t0 = time.time()
    for i, r in sample.iterrows():
        state = {"subject": str(r.subject), "body": str(r.body)[:4000]}
        out = v.decide(state, QUESTIONS)
        row = {"subject": str(r.subject)[:60]}
        for q in ("queue", "priority", "type"):
            a = out["answers"][q]
            pred = a["choice"] if q != "priority" else PRIORITIES[int(round(a["score"]))]
            gold = r[q]
            if q == "priority":
                gold = PRIORITIES.index(gold)
                ok = abs(int(round(a["score"])) - gold) <= 1  # adjacent = half credit
                exact = int(round(a["score"])) == gold
            else:
                ok = exact = pred == gold
            abstains[q] += bool(a["abstain"])
            row[q] = {"pred": pred, "gold": gold, "ok": bool(ok), "exact": bool(exact),
                      "conf": round(a["confidence"], 3), "abstain": bool(a["abstain"])}
        rows.append(row)
    dt = time.time() - t0
    summary = {"n": len(rows), "seconds": round(dt, 1),
               "ms_per_ticket": round(1000 * dt / len(rows)),
               **{f"{q}_acc": round(sum(r[q]["ok"] for r in rows) / len(rows), 3) for q in ("queue", "type")},
               "priority_acc_adj1": round(sum(r["priority"]["ok"] for r in rows) / len(rows), 3),
               "priority_acc_exact": round(sum(r["priority"]["exact"] for r in rows) / len(rows), 3),
               **{f"{q}_abstain": abstains[q] for q in abstains}}
    (RESULTS / "tickets_detailed.json").write_text(json.dumps(rows, indent=1))
    return summary, rows


def run_emotion(v, n):
    df = pd.read_parquet(DATA / "emotion_test.parquet").sample(n=min(n, len(pd.read_parquet(DATA / "emotion_test.parquet"))), random_state=7)
    qs = {"emotion": QUESTIONS["emotion"]}
    ok, conf_right = 0, 0
    lat = []
    for _, r in df.iterrows():
        t0 = time.time()
        out = v.decide(str(r.text), qs)
        lat.append(time.time() - t0)
        a = out["answers"]["emotion"]
        ok += a["choice"] == EMOTIONS[int(r.label)]
        conf_right += (a["choice"] == EMOTIONS[int(r.label)]) == (a["confidence"] == max(a["probs"].values()))
    return {"n": len(df), "accuracy": round(ok / len(df), 3),
            "median_ms": round(1000 * sorted(lat)[len(lat) // 2])}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-tickets", type=int, default=60)
    ap.add_argument("--n-emotion", type=int, default=100)
    ap.add_argument("--skip", default="")
    args = ap.parse_args()

    import vegaml
    v = vegaml.load()
    summary = {"model": "vega-U (Qwen3.5-0.8B reader), engine readout, zero-shot"}
    if "tickets" not in args.skip:
        s, _ = run_tickets(v, args.n_tickets)
        summary["support_tickets"] = s
        print("tickets:", json.dumps(s, indent=1))
    if "emotion" not in args.skip:
        e = run_emotion(v, args.n_emotion)
        summary["emotion"] = e
        print("emotion:", json.dumps(e, indent=1))
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
