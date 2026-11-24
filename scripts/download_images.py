"""Download Fakeddit images from the `image_url` column of a TSV file."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm


def fetch(row, out_dir: Path):
    path = out_dir / f"{row['id']}.jpg"
    if path.exists():
        return True
    try:
        r = requests.get(row["image_url"], timeout=10)
        if r.status_code == 200 and len(r.content) > 1000:
            path.write_bytes(r.content)
            return True
    except Exception:
        pass
    return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tsv", required=True)
    p.add_argument("--out", default="data/raw/images")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--workers", type=int, default=16)
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(a.tsv, sep="\t")
    df = df[df["hasImage"] == True].dropna(subset=["image_url"])  # noqa: E712
    if a.limit:
        df = df.head(a.limit)
    rows = df.to_dict("records")
    with ThreadPoolExecutor(a.workers) as ex:
        ok = list(tqdm(ex.map(lambda r: fetch(r, out), rows), total=len(rows)))
    print(f"Downloaded/available: {sum(ok)}/{len(rows)}")


if __name__ == "__main__":
    main()
