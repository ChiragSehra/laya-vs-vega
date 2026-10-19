"""Does vegaml.fit() on a small labelled sample fix the weak queue decision?

Train on `--train-n` labelled tickets (the dataset's own queue labels),
re-evaluate on the SAME held-out sample the baseline demo used.
"""
import argparse, json, pathlib, time
import pandas as pd

from demo_triage import DATA, RESULTS, QUESTIONS

root = pathlib.Path(__file__).parent


def main(n_train, n_eval):
    import vegaml
    v = vegaml.load()

    df = pd.read_parquet(DATA / "support_tickets.parquet")
    df = df[(df.language == "en") & df.body.notna()].reset_index(drop=True)
    per = max(1, (n_train + n_eval) // df.queue.nunique())
    pool = df.groupby("queue").sample(n=per, random_state=7).reset_index(drop=True)
    train = pool.sample(n=n_train, random_state=11)
    eval_df = pool.sample(n=n_eval, random_state=7)  # same sample as the baseline run

    # 1. fit a TTT head for the queue question on the dataset's own labels
    examples = [({"subject": str(r.subject), "body": str(r.body)[:4000]}, {"queue": r.queue})
                for _, r in train.iterrows()]
    t0 = time.time()
    report = v.fit(examples, {"queue": QUESTIONS["queue"]})
    print("fit in %.0fs:" % (time.time() - t0), json.dumps(report, indent=1))

    # 2. re-run the evaluation, auto mode (uses the head where it earned it)
    rows = []
    for _, r in eval_df.iterrows():
        out = v.decide({"subject": str(r.subject), "body": str(r.body)[:4000]}, QUESTIONS)
        a = out["answers"]["queue"]
        rows.append({"pred": a["choice"], "gold": r.queue, "ok": a["choice"] == r.queue,
                     "readout": a["readout"], "abstain": bool(a.get("abstain", False)),
                     "conf": round(a["confidence"], 3)})
    summary = {
        "n_eval": len(rows),
        "queue_acc_zero_shot_baseline": 0.183,
        "queue_acc_after_fit": round(sum(r["ok"] for r in rows) / len(rows), 3),
        "readouts_used": pd.Series([r["readout"] for r in rows]).value_counts().to_dict(),
        "abstains": sum(r["abstain"] for r in rows),
    }
    (RESULTS / "fit_queue_eval.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-n", type=int, default=120)
    ap.add_argument("--eval-n", type=int, default=60)
    a = ap.parse_args()
    main(a.train_n, a.eval_n)
