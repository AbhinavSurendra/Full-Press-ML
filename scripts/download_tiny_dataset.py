from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import urllib.request
from pathlib import Path

import py7zr

TRACKING_INDEX_URL = (
    "https://github.com/linouk23/NBA-Player-Movements/raw/master/data/2016.NBA.Raw.SportVU.Game.Logs"
)
PBP_URL = (
    "https://github.com/sumitrodatta/nba-alt-awards/raw/main/Historical/PBP%20Data/2015-16_pbp.csv"
)


def fetch_listing_items() -> list[dict]:
    with urllib.request.urlopen(TRACKING_INDEX_URL) as response:
        text = response.read().decode("utf-8", errors="ignore")
    match = re.findall(r'{"items":*\[.*?\]', text, re.DOTALL)
    if not match:
        raise RuntimeError("Unable to parse tracking file listing from upstream GitHub page.")
    return json.loads(match[0] + "}")["items"]


def select_games(config: str, items: list[dict]) -> list[dict]:
    sample_sizes = {
        "tiny": 5,
        "small": 25,
        "medium": 100,
        "full": len(items),
    }
    random.seed(9)
    return random.sample(items, sample_sizes[config])


def download_file(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, output_path.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def extract_archive(archive_path: Path, output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with py7zr.SevenZipFile(archive_path, mode="r") as archive:
        names = archive.getnames()
        archive.extractall(path=output_dir)
    return names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="tiny", choices=["tiny", "small", "medium", "full"])
    parser.add_argument("--output-dir", default="data/raw/tiny")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    archives_dir = output_dir / "archives"
    games_dir = output_dir / "games"
    output_dir.mkdir(parents=True, exist_ok=True)

    items = fetch_listing_items()
    selected_games = select_games(args.config, items)

    manifest: list[dict[str, str]] = []
    for game in selected_games:
        name = game["name"][:-3]
        archive_url = f"{TRACKING_INDEX_URL}/{name}.7z"
        archive_path = archives_dir / f"{name}.7z"
        extract_dir = games_dir / name
        download_file(archive_url, archive_path)
        extracted_files = extract_archive(archive_path, extract_dir)
        manifest.append(
            {
                "game_name": name,
                "archive_path": str(archive_path),
                "extract_dir": str(extract_dir),
                "extracted_files": ",".join(extracted_files),
            }
        )
        print(f"downloaded {name}")

    pbp_path = output_dir / "2015-16_pbp.csv"
    download_file(PBP_URL, pbp_path)

    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "config": args.config,
                "num_games": len(selected_games),
                "games": manifest,
                "pbp_path": str(pbp_path),
            },
            handle,
            indent=2,
        )

    print(f"wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
