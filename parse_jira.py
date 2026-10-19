"""Stream-parse the Apache Jira issues BSON dump into a flat parquet.

No MongoDB needed: gzip stream -> bson.decode_file_iter.
Usage: .venv/bin/python parse_jira.py
"""
import gzip
import itertools

import bson
import pandas as pd

SRC = "data/issues.bson.gz"


def field(doc, *path):
    d = doc
    for p in path:
        if not isinstance(d, dict):
            return None
        d = d.get(p)
    if isinstance(d, dict) and "name" in d:
        d = d["name"]
    if isinstance(d, dict) and "value" in d:
        d = d["value"]
    return d


def main(limit=None):
    rows = []
    with gzip.open(SRC, "rb") as f:
        for doc in itertools.islice(bson.decode_file_iter(f), limit):
            rows.append({
                "key": doc.get("issuekey") or doc.get("key"),
                "project": doc.get("projectname"),
                "issuetype": field(doc, "issuetype"),
                "priority": field(doc, "priority"),
                "status": field(doc, "status"),
                "summary": (doc.get("summary") or "")[:500],
                "description": (doc.get("description") or "")[:4000],
                "components": field(doc, "components"),
                "created": field(doc, "created"),
            })
    df = pd.DataFrame(rows)
    print(df.shape)
    print(df.head(3).to_string(max_colwidth=70))
    for c in ("issuetype", "priority", "status"):
        print(c, "->", df[c].value_counts().head(12).to_dict())
    print("projects:", df.project.nunique(), df.project.value_counts().head(10).to_dict())
    df.to_parquet("data/jira_issues.parquet")
    print("saved data/jira_issues.parquet")


if __name__ == "__main__":
    main(limit=2_000_000)
