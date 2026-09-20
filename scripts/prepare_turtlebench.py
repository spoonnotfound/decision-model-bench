"""Prepare a pinned upstream dataset. This script does not call a model."""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

REVISION = "6547a5f0cb7097292b900b256d5e824a280acd33"
ROOT = Path(__file__).resolve().parents[1]


def prepare(language="en"):
    base = f"https://raw.githubusercontent.com/mazzzystar/TurtleBench/{REVISION}/data/{language}/"
    sources = {}
    for name in ["stories.json", "cases.list"]:
        with urllib.request.urlopen(base + name, timeout=45) as response:
            sources[name] = response.read()
    stories = {s["title"]: s for s in json.loads(sources["stories.json"])}
    labels = {"T": "Correct", "F": "Incorrect", "N": "Unknown"}
    rows = []
    for i, line in enumerate(sources["cases.list"].decode().splitlines(), 1):
        if language == "en":
            guess, title, target = line.strip().split("\t|\t")
        else:
            guess, title, target = line.strip().replace(" ", "").split("\t")
            target = labels[target]
        story = stories[title]
        state = (
            (
                "Surface story:\n"
                + story["surface"]
                + "\n\nBottom story:\n"
                + story["bottom"]
                + "\n\nPlayer guess:\n"
                + guess
            )
            if language == "en"
            else (
                "汤面：\n" + story["surface"] + "\n\n汤底：\n" + story["bottom"] + "\n\n玩家猜测：\n" + guess
            )
        )
        rows.append({"id": str(i), "group": title, "state": state, "target": target})
    assert len(rows) == 1532
    data = ROOT / "tasks/data"
    data.mkdir(exist_ok=True)
    (data / f"turtlebench-{language}.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    provenance = {
        "upstream": "https://github.com/mazzzystar/TurtleBench",
        "revision": REVISION,
        "language": language,
        "sha256": {k: hashlib.sha256(v).hexdigest() for k, v in sources.items()},
        "modifications": "Normalized to JSONL; compact state rendering; T/F/N renamed in zh; no relabeling.",
        "license": "Apache-2.0; see LICENSES/TurtleBench.txt and NOTICE",
    }
    (data / f"turtlebench-{language}.provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Prepared {len(rows)} {language} cases at revision {REVISION}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", choices=["en", "zh"], default="en")
    prepare(parser.parse_args().language)
