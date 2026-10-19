"""Clean-split vegaml vs laya on the Apache Jira dataset.

Splits are disjoint at the ISSUE level (random hash split, no overlap) and we
also report a harder PROJECT-level split (train projects != eval projects) to
test generalization to unseen teams.

Decisions (labels come from the dataset itself):
  - issuetype: choice {Bug, Improvement, Task, New Feature, Test, Wish}
  - priority:  score 0-4 over [Trivial, Minor, Major, Critical, Blocker]

Usage: .venv/bin/python jira_h2h.py [--n-train 120] [--n-eval 100] [--engine both]
"""
import argparse
import hashlib
import json
import pathlib
import time

import pandas as pd

from demo_triage import DATA, RESULTS

ISSUE_TYPES = ["Bug", "Improvement", "Task", "New Feature", "Test", "Wish"]
PRIORITIES = ["Trivial", "Minor", "Major", "Critical", "Blocker"]

QUESTIONS = {
    "issuetype": {
        "type": "choice",
        "instructions": "What kind of software issue is this?",
        "criteria": {
            "Bug": "a defect: code behaves incorrectly or breaks",
            "Improvement": "an enhancement to existing functionality",
            "Task": "a work item, chore or maintenance activity",
            "New Feature": "brand-new functionality that did not exist",
            "Test": "adding or improving tests",
            "Wish": "a user wish or suggestion, low formality",
        },
    },
    "priority": {
        "type": "score",
        "instructions": "How severe is this issue?",
        "criteria": PRIORITIES,
    },
}


def bucket(key, frac, salt):
    """Stable pseudo-random assignment of an issue to [0, frac)."""
    h = hashlib.sha1(f"{salt}:{key}".encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF < frac


def load_jira(min_len=60):
    df = pd.read_parquet(DATA / "jira_issues.parquet")
    df = df[df.issuetype.isin(ISSUE_TYPES) & df.priority.isin(PRIORITIES)]
    df = df[df.summary.notna()]
    df["text"] = (df.summary.fillna("") + "\n" + df.description.fillna("")).str.slice(0, 4000)
    df = df[df.text.str.len() >= min_len].reset_index(drop=True)
    return df


def make_split(df, salt, disjoint_on="key"):
    if disjoint_on == "project":
        projects = sorted(df.project.dropna().unique())
        train_projects = {p for p in projects if bucket(p, 0.8, salt)}
        return df[df.project.isin(train_projects)], df[~df.project.isin(train_projects)]
    is_train = df.key.map(lambda k: bucket(k, 0.8, salt))
    return df[is_train], df[~is_train]


def score_row(q, pred, gold):
    if q == "priority":
        g = PRIORITIES.index(gold)
        return abs(int(round(pred)) - g) <= 1, int(round(pred)) == g
    return pred == gold, pred == gold


def run_vegaml(train, ev, n_train, n_eval):
    import vegaml
    v = vegaml.load()
    tr = train.sample(n=min(n_train, len(train)), random_state=3)
    te = ev.sample(n=min(n_eval, len(ev)), random_state=3)
    examples = [({"text": r.text}, {"issuetype": r.issuetype, "priority": PRIORITIES.index(r.priority)})
                for _, r in tr.iterrows()]
    t0 = time.time()
    fit_report = v.fit(examples, QUESTIONS)
    fit_s = round(time.time() - t0)
    rows = []
    for _, r in te.iterrows():
        out = v.decide({"text": r.text}, QUESTIONS)
        row = {}
        for q in ("issuetype", "priority"):
            a = out["answers"][q]
            pred = a["choice"] if q == "issuetype" else a["score"]
            ok, exact = score_row(q, pred, r[q])
            row[q] = {"ok": bool(ok), "exact": bool(exact), "readout": a.get("readout")}
        rows.append(row)
    n = len(rows)
    return {
        "n_train": len(tr), "n_eval": n, "fit_seconds": fit_s,
        "fit_report": fit_report,
        "issuetype_acc": round(sum(r["issuetype"]["ok"] for r in rows) / n, 3),
        "priority_acc_adj1": round(sum(r["priority"]["ok"] for r in rows) / n, 3),
        "priority_acc_exact": round(sum(r["priority"]["exact"] for r in rows) / n, 3),
        "readouts": pd.Series([r["issuetype"]["readout"] for r in rows]).value_counts().to_dict(),
    }


def run_laya(train, ev, n_train, n_eval):
    import laya
    agent = laya.load("convaiinnovations/laya")
    tr = train.sample(n=min(n_train, len(train)), random_state=3)
    te = ev.sample(n=min(n_eval, len(ev)), random_state=3)
    questions = {
        "issuetype": dict(QUESTIONS["issuetype"], type="choice"),
        "priority": dict(QUESTIONS["priority"], type="score",
                         criteria=[c for c in PRIORITIES]),
    }
    t0 = time.time()
    agent.predict_batch([{"text": r.text} for _, r in tr.head(8).iterrows()], questions)  # warm-up
    rows = []
    t0 = time.time()
    for _, r in te.iterrows():
        out = agent.predict({"text": r.text}, questions)
        row = {}
        for q in ("issuetype", "priority"):
            a = out["answers"][q]
            pred = a["choice"] if q == "issuetype" else a["score"]
            ok, exact = score_row(q, pred, r[q])
            row[q] = {"ok": bool(ok), "exact": bool(exact)}
        rows.append(row)
    n = len(rows)
    return {
        "n_eval": n, "predict_seconds": round(time.time() - t0),
        "issuetype_acc": round(sum(r["issuetype"]["ok"] for r in rows) / n, 3),
        "priority_acc_adj1": round(sum(r["priority"]["ok"] for r in rows) / n, 3),
        "priority_acc_exact": round(sum(r["priority"]["exact"] for r in rows) / n, 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=120)
    ap.add_argument("--n-eval", type=int, default=100)
    ap.add_argument("--engine", default="vegaml", choices=["vegaml", "laya"])
    args = ap.parse_args()

    df = load_jira()
    print("usable issues:", len(df))
    out = {}
    for split_name, on in (("issue-disjoint", "key"), ("project-disjoint", "project")):
        train, ev = make_split(df, salt="jira-" + on, disjoint_on=on)
        assert not set(train.key) & set(ev.key), "split leaked!"
        fn = run_vegaml if args.engine == "vegaml" else run_laya
        out[split_name] = fn(train, ev, args.n_train, args.n_eval)
        print(split_name, "->", json.dumps(out[split_name], indent=1, default=str))
    p = RESULTS / f"jira_h2h_{args.engine}.json"
    p.write_text(json.dumps(out, indent=1, default=str))
    print("saved", p)


if __name__ == "__main__":
    main()
