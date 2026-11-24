"""Separate extractor misses from model misses.

The study scores a fixed pipeline: prompt, model, deterministic extractor. A
capable model answering a plain question writes an explanation with several
codes in it, and the extractor can pick the wrong one. This script measures
how much of each strategy's error is that kind of miss, by asking whether the
label appears anywhere in the raw response (an upper bound on what any
extractor could recover) and whether the first code in the response is right.

Usage::

    python scripts/parse_gap.py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from icdprompt.infer import MODEL_IDS  # noqa: E402

CODE = re.compile(r"\b([A-Z][0-9]{2})(?:\.[0-9]{1,2})?\b")


def codes_in(text: object) -> list[str]:
    """All category-shaped tokens in a response, in order."""
    return CODE.findall(str(text)) if isinstance(text, str) else []


def load_full_text(cache_dir: Path) -> dict[tuple[str, str, str], str]:
    """Map ``(model, template, record_id)`` to the complete cached response.

    The results file keeps only the first few hundred characters of each
    response, and long explanations put their code after that cut.
    """
    import json

    texts: dict[tuple[str, str, str], str] = {}
    for path in cache_dir.glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        texts[(payload["model"], payload["template"], payload["record_id"])] = payload.get("text") or ""
    return texts


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="*", default=list(MODEL_IDS))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    frames = []
    for model in args.models:
        path = REPO_ROOT / "results" / model / "predictions.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
        else:
            print(f"no predictions for {model}; skipping", file=sys.stderr)
    if not frames:
        print("nothing to analyse", file=sys.stderr)
        return 1
    df = pd.concat(frames, ignore_index=True)
    full = load_full_text(REPO_ROOT / "results" / "raw")
    text = [full.get((m, t, r), raw) for m, t, r, raw in zip(df.model, df.template, df.record_id, df.raw_output)]
    mentioned = pd.Series(text, index=df.index).map(codes_in)
    df["label_mentioned"] = [t in c for t, c in zip(df.true_code, mentioned)]
    df["first_correct"] = [bool(c) and c[0] == t for t, c in zip(df.true_code, mentioned)]
    df["n_mentioned"] = mentioned.map(len)

    out = (
        df.groupby(["model", "split", "template"])
        .agg(
            n=("record_id", "size"),
            accuracy=("is_correct", "mean"),
            label_mentioned=("label_mentioned", "mean"),
            first_code_correct=("first_correct", "mean"),
            codes_per_response=("n_mentioned", "mean"),
        )
        .reset_index()
    )
    out["extractor_gap"] = out["label_mentioned"] - out["accuracy"]
    out = out.sort_values(["split", "model", "accuracy"])
    target = args.out or REPO_ROOT / "results" / "parse_gap.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(target, index=False)
    with pd.option_context("display.width", 200):
        print(out.round(2).to_string(index=False))
    print(f"\nwrote {target.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
