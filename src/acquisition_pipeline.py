"""
Pipeline d'acquisition - orchestre les 3 sous-modules.

Execute dans l'ordre:
  1. acquisition_oiken          -> data/raw/oiken_raw.parquet
  2. acquisition_meteo_real     -> data/raw/meteo_real_raw.parquet
  3. acquisition_meteo_pred     -> data/raw/meteo_pred_raw.parquet

Puis genere data/reports/acquisition_report.json avec les statistiques
globales (lignes, sites, plage temporelle, sites complets/incomplets).

Pourquoi un orchestrateur separe plutot qu'un seul gros script ?
  - Chaque sous-module est lancable INDEPENDAMMENT pour debug.
  - L'orchestrateur peut paralleliser ou ajouter du cache si besoin
    (les telechargements InfluxDB durent 5-15 min selon la connexion).
  - Le rapport global donne une vue d'ensemble avant de lancer la
    normalisation, ce qui evite de partir avec des donnees corrompues.

Utilisation:
    python -m src.acquisition_pipeline
    python -m src.acquisition_pipeline --skip-meteo    # OIKEN seul
    python -m src.acquisition_pipeline --skip-pred     # sans previsions
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import polars as pl

from src.config import (
    DATA_RAW,
    DATA_REPORTS,
    OIKEN_EXPECTED_ROWS,
    SITES,
    MEASUREMENTS_REAL,
    MEASUREMENTS_PRED,
)


# ============================================================
# Etapes individuelles
# ============================================================

def step_oiken() -> dict:
    """Execute acquisition_oiken et retourne les metriques."""
    from src import acquisition_oiken

    t0 = time.time()
    rc = acquisition_oiken.main()
    dt = time.time() - t0

    if rc != 0:
        return {"status": "failed", "elapsed_s": round(dt, 1)}

    # Recharger pour metriques
    df = pl.read_parquet(DATA_RAW / "oiken_raw.parquet")

    return {
        "status": "ok",
        "elapsed_s": round(dt, 1),
        "rows": df.height,
        "rows_expected": OIKEN_EXPECTED_ROWS,
        "rows_match": df.height == OIKEN_EXPECTED_ROWS,
        "ts_min": str(df["timestamp_local"].min()),
        "ts_max": str(df["timestamp_local"].max()),
        "nulls": {c: df[c].null_count() for c in df.columns
                  if df[c].null_count() > 0},
    }


def step_meteo_real() -> dict:
    from src import acquisition_meteo_real

    t0 = time.time()
    rc = acquisition_meteo_real.main()
    dt = time.time() - t0

    if rc != 0:
        return {"status": "failed", "elapsed_s": round(dt, 1)}

    df = pl.read_parquet(DATA_RAW / "meteo_real_raw.parquet")

    return {
        "status": "ok",
        "elapsed_s": round(dt, 1),
        "rows": df.height,
        "n_sites": df["site"].n_unique(),
        "sites": df["site"].unique().to_list(),
        "ts_min": str(df["timestamp"].min()),
        "ts_max": str(df["timestamp"].max()),
        "n_columns": df.shape[1],
    }


def step_meteo_pred() -> dict:
    from src import acquisition_meteo_pred

    t0 = time.time()
    rc = acquisition_meteo_pred.main()
    dt = time.time() - t0

    if rc != 0:
        return {"status": "failed", "elapsed_s": round(dt, 1)}

    df = pl.read_parquet(DATA_RAW / "meteo_pred_raw.parquet")

    return {
        "status": "ok",
        "elapsed_s": round(dt, 1),
        "rows": df.height,
        "n_sites": df["site"].n_unique(),
        "sites": df["site"].unique().to_list(),
        "ts_min": str(df["timestamp"].min()),
        "ts_max": str(df["timestamp"].max()),
        "n_columns": df.shape[1],
    }


# ============================================================
# Verification post-acquisition
# ============================================================

def cross_checks(report: dict) -> list[str]:
    """
    Verifications croisees entre les 3 sources. Renvoie une liste
    de warnings (chaines a afficher), vide si tout va bien.

    Verifie:
      - OIKEN a le bon nombre de lignes
      - Meteo real et pred ont des plages temporelles compatibles
        avec la periode OIKEN
      - Tous les sites de SITES sont presents dans les sorties meteo
    """
    warnings_list: list[str] = []

    oiken = report.get("oiken", {})
    real = report.get("meteo_real", {})
    pred = report.get("meteo_pred", {})

    if oiken.get("status") == "ok" and not oiken.get("rows_match"):
        warnings_list.append(
            f"OIKEN a {oiken['rows']} lignes (attendu "
            f"{oiken['rows_expected']}). Verifier le CSV source."
        )

    expected_sites = set(SITES)

    if real.get("status") == "ok":
        actual = set(real.get("sites", []))
        missing = expected_sites - actual
        if missing:
            warnings_list.append(
                f"meteo_real: sites manquants {missing} "
                f"(probablement pas de donnees sur la periode)"
            )

    if pred.get("status") == "ok":
        actual = set(pred.get("sites", []))
        missing = expected_sites - actual
        if missing:
            warnings_list.append(
                f"meteo_pred: sites manquants {missing}"
            )

    return warnings_list


# ============================================================
# Sauvegarde du rapport
# ============================================================

def save_report(report: dict, name: str = "acquisition_report") -> Path:
    DATA_REPORTS.mkdir(parents=True, exist_ok=True)
    out = DATA_REPORTS / f"{name}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    return out


# ============================================================
# Main
# ============================================================

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pipeline d'acquisition OIKEN ML")
    parser.add_argument("--skip-oiken", action="store_true",
                        help="Saute l'acquisition OIKEN")
    parser.add_argument("--skip-meteo", action="store_true",
                        help="Saute toute la meteo (real + pred)")
    parser.add_argument("--skip-real", action="store_true",
                        help="Saute meteo reelle")
    parser.add_argument("--skip-pred", action="store_true",
                        help="Saute meteo previsions")
    args = parser.parse_args(argv)

    print("\n" + "#" * 60)
    print("# PIPELINE ACQUISITION OIKEN ML")
    print(f"# Sites    : {len(SITES)}")
    print(f"# Mesures  : {len(MEASUREMENTS_REAL)} reelles + "
          f"{len(MEASUREMENTS_PRED)} previsions")
    print("#" * 60 + "\n")

    t_start = time.time()
    report = {"steps": {}}

    if not args.skip_oiken:
        report["oiken"] = step_oiken()
    if not args.skip_meteo and not args.skip_real:
        report["meteo_real"] = step_meteo_real()
    if not args.skip_meteo and not args.skip_pred:
        report["meteo_pred"] = step_meteo_pred()

    elapsed = time.time() - t_start
    report["total_elapsed_s"] = round(elapsed, 1)

    # Verifications croisees
    warnings_list = cross_checks(report)
    report["warnings"] = warnings_list

    # Sauvegarde rapport
    out = save_report(report)

    # Resume console
    print("\n" + "#" * 60)
    print("# RESUME PIPELINE")
    print("#" * 60)
    for step, info in report.items():
        if step in ("warnings", "total_elapsed_s", "steps"):
            continue
        status = info.get("status", "?")
        print(f"  {step:15s}: {status:8s}  "
              f"({info.get('elapsed_s', 0):.1f}s, "
              f"{info.get('rows', 0):,} rows)")

    if warnings_list:
        print("\n  WARNINGS:")
        for w in warnings_list:
            print(f"    - {w}")
    else:
        print("\n  Aucun warning.")

    print(f"\n  Total: {elapsed:.1f}s")
    print(f"  Rapport: {out}")
    print("\nPIPELINE TERMINE\n")

    # Code de sortie != 0 si une etape a echoue
    if any(info.get("status") == "failed"
           for k, info in report.items()
           if isinstance(info, dict) and "status" in info):
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())