"""FAZ 4b — Datasetten istatistik çıkar (video/model gerekmez).

    python scripts/08_analyze.py data/matches/match

Çıktı: <match>/stats/
    players.csv        track bazında mesafe, maks/ort. hız, sprint sayısı
    possession.csv     frame bazında topa en yakın oyuncu/takım (proxy)
    team_shape.csv     frame x takım: merkez, derinlik, genişlik
    heatmap_team{0,1}.png
    summary.json
"""
import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fai.analysis import player_stats, possession_proxy, team_shape  # noqa: E402
from fai.dataio import load_match  # noqa: E402
from fai.pitch import PitchConfig  # noqa: E402
from fai.viz import Radar  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("match_dir")
    ap.add_argument("--min-seconds", type=float, default=5.0, help="bundan kısa track'ler özet tablodan çıkar")
    a = ap.parse_args()

    df, frames, meta = load_match(a.match_dir)
    out = Path(a.match_dir) / "stats"
    out.mkdir(exist_ok=True)
    fps = meta["fps"]

    ps = player_stats(df)
    ps = ps[ps["seconds_tracked"] >= a.min_seconds]
    ps.to_csv(out / "players.csv", float_format="%.2f")

    poss = possession_proxy(df, fps=fps)
    poss.to_csv(out / "possession.csv")
    held = poss["team_held"].dropna()
    share = held.value_counts(normalize=True).sort_index()

    shape = team_shape(df)
    shape.to_csv(out / "team_shape.csv", index=False, float_format="%.2f")

    radar = Radar(PitchConfig(**meta["pitch"]))
    for t in (0, 1):
        pts = df[(df["team"] == t) & (df["role"] == "player") & df["on_pitch"]][["xs", "ys"]].to_numpy()
        cv2.imwrite(str(out / f"heatmap_team{t}.png"), radar.heatmap(pts))

    summary = {
        "duration_s": round(meta["frames_processed"] / fps, 1),
        "homography_coverage": meta["homography_coverage"],
        "possession_proxy": {f"team{int(k)}": round(float(v), 3) for k, v in share.items()},
        "possession_unknown_frac": round(float(poss["team_held"].isna().mean()), 3),
        "team_avg_depth_m": shape.groupby("team")["depth_m"].mean().round(1).to_dict(),
        "team_avg_width_m": shape.groupby("team")["width_m"].mean().round(1).to_dict(),
        "top_distance": ps.head(5)[["team", "distance_m", "max_speed_kmh"]].round(1).reset_index().to_dict("records"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[ok] -> {out}")


if __name__ == "__main__":
    main()
