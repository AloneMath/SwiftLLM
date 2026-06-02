from __future__ import annotations

import argparse
from pathlib import Path

import requests
from tqdm import tqdm

BASE_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
MAX_SHARD = 6542


def shard_name(index: int) -> str:
    return f"shard_{index:05d}.parquet"


def download_file(url: str, out_path: Path, chunk_size: int = 1024 * 1024) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with tmp_path.open("wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=out_path.name) as pbar:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                pbar.update(len(chunk))

    tmp_path.replace(out_path)


def download_shard(index: int, out_dir: Path) -> None:
    name = shard_name(index)
    out_path = out_dir / name
    if out_path.exists():
        print(f"skip existing: {out_path}")
        return

    url = f"{BASE_URL}/{name}"
    print(f"download: {url}")
    download_file(url, out_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download Karpathy ClimbMix shards")
    p.add_argument("--out-dir", type=str, required=True, help="Output directory")
    p.add_argument("--num-train-shards", type=int, default=8, help="Number of leading train shards")
    p.add_argument("--workers", type=int, default=1, help="Reserved for future parallel download")
    p.add_argument("--no-val", action="store_true", help="Skip downloading the fixed validation shard")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)

    num_train = max(0, min(args.num_train_shards, MAX_SHARD))
    indices = list(range(num_train))
    if not args.no_val:
        indices.append(MAX_SHARD)

    print(f"target dir: {out_dir}")
    print(f"train shards: {num_train}")
    print(f"download count: {len(indices)}")

    for idx in indices:
        download_shard(idx, out_dir)

    print("done")


if __name__ == "__main__":
    main()
