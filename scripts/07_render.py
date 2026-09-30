"""FAZ 4a — Datasetten video render et (model tekrar çalışmaz).

    python scripts/07_render.py data/matches/match --mode radar      # yayın + radar
    python scripts/07_render.py data/matches/match --mode voronoi    # yayın + saha kontrolü
    python scripts/07_render.py data/matches/match --radar-only      # sadece kuşbakışı
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fai.render import render_match  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("match_dir")
    ap.add_argument("--mode", choices=["radar", "voronoi", "none"], default="radar")
    ap.add_argument("--radar-only", action="store_true")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--video", default=None, help="meta.json'daki video yolu değiştiyse")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    suffix = f"{a.mode}{'_radaronly' if a.radar_only else ''}"
    out = a.out or str(Path(a.match_dir) / f"render_{suffix}.mp4")
    p = render_match(a.match_dir, out, a.mode, a.radar_only, a.max_frames, a.video)
    print(f"[ok] {p}")


if __name__ == "__main__":
    main()
