"""
Acquisition InfluxDB - PREVISIONS meteo COSMO-E.

Telecharge les 32 variables previsionnelles (8 grandeurs * 4 statistiques
ctrl/q10/q90/stde) pour le run "00" (lance a minuit UTC), pour chaque
site configure.

Pourquoi seulement le run "00" ?
  - Pour un forecast J+1 publie le matin (avant 06h00 locale), c'est le
    run le plus recent disponible : COSMO-E tourne a 00 UTC et publie
    vers 03 UTC (= 04-05 locale selon DST). Les runs ulterieurs (06,
    12, 18) seraient utiles pour de l'intraday, pas pour le J+1.
  - Si on voulait reentrainer en utilisant tous les runs comme
    augmentation de donnees, il faudrait ouvrir PRED_RUN. Pas le cas ici.

Particularites des previsions:
  - Pas natif tres different : COSMO-E publie des previsions par pas
    horaires (1h en general, 3h pour certaines grandeurs sur l'horizon
    long). On garde le pas natif ici, le rééchantillonnage est fait
    en normalisation.
  - Horizon : COSMO-E couvre +120h. Pour notre J+1 on n'utilise que
    les 24-48 premières heures, mais on stocke tout.

Sortie: data/raw/meteo_pred_raw.parquet
  Colonnes:
    timestamp                   (Datetime UTC)
    site                        (Utf8)
    pred_t_2m_ctrl, pred_t_2m_q10, pred_t_2m_q90, pred_t_2m_stde
    pred_glob_ctrl, ... (8 grandeurs * 4 variantes = 32 colonnes)

Utilisation:
    python -m src.acquisition_meteo_pred
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import polars as pl
from influxdb_client import InfluxDBClient
from influxdb_client.client.warnings import MissingPivotFunction

from src.config import (
    DATA_RAW,
    INFLUXDB_URL,
    INFLUXDB_TOKEN,
    INFLUXDB_ORG,
    INFLUXDB_BUCKET,
    INFLUX_TIMEOUT_MS,
    SITES,
    MEASUREMENTS_PRED,
    PRED_RUN,
    OIKEN_START,
    OIKEN_END,
)

warnings.simplefilter("ignore", MissingPivotFunction)


# ============================================================
# Construction de requetes Flux (avec filtre Prediction)
# ============================================================

def build_flux_query_pred(
    bucket: str,
    measurement: str,
    site: str,
    prediction: str,
    start: str,
    stop: str,
) -> str:
    """
    Comme build_flux_query_real mais avec un filtre supplementaire sur
    le tag Prediction (= identifiant du run du modele).
    """
    return f"""
from(bucket: "{bucket}")
  |> range(start: {start}T00:00:00Z, stop: {stop}T23:59:59Z)
  |> filter(fn: (r) => r._measurement == "{measurement}")
  |> filter(fn: (r) => r.Site == "{site}")
  |> filter(fn: (r) => r.Prediction == "{prediction}")
  |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
""".strip()


# ============================================================
# Telechargement
# ============================================================

def _short_name_pred(measurement: str) -> str:
    """
    Convertit 'PRED_T_2M_ctrl' -> 'pred_t_2m_ctrl'.

    Garde la structure complete (variable + statistique) en lowercase.
    """
    return measurement.lower()


def fetch_one_pred(
    query_api,
    measurement: str,
    site: str,
    prediction: str,
    start: str,
    stop: str,
) -> pl.DataFrame | None:
    short = _short_name_pred(measurement)

    query = build_flux_query_pred(
        INFLUXDB_BUCKET, measurement, site, prediction, start, stop
    )

    try:
        pdf = query_api.query_data_frame(query)
    except Exception as e:
        print(f"    [ERREUR Flux] {site} / {short}: {type(e).__name__}: {e}")
        return None

    if pdf is None or len(pdf) == 0 or pdf.empty:
        return None
    if "Value" not in pdf.columns:
        return None

    df = pl.from_pandas(pdf[["_time", "Value"]].rename(
        columns={"_time": "timestamp", "Value": short}
    ))
    df = df.with_columns(pl.lit(site).alias("site"))
    df = df.with_columns(pl.col(short).cast(pl.Float64))

    return df


def fetch_meteo_pred(
    sites: list[str],
    measurements: list[str],
    prediction: str = PRED_RUN,
    start: str = OIKEN_START,
    stop: str = OIKEN_END,
) -> tuple[pl.DataFrame, dict]:
    """
    Telecharge l'ensemble des combinaisons (site, mesure_pred).

    Meme structure que fetch_meteo_real, avec en plus le filtre Prediction.
    """
    client = InfluxDBClient(
        url=INFLUXDB_URL,
        token=INFLUXDB_TOKEN,
        org=INFLUXDB_ORG,
        timeout=INFLUX_TIMEOUT_MS,
    )
    query_api = client.query_api()

    logs: dict[str, dict] = {s: {} for s in sites}
    frames_by_site: dict[str, list[pl.DataFrame]] = {s: [] for s in sites}

    total = len(sites) * len(measurements)
    counter = 0

    for site in sites:
        for measurement in measurements:
            counter += 1
            short = _short_name_pred(measurement)
            t0 = time.time()
            df = fetch_one_pred(query_api, measurement, site, prediction,
                                start, stop)
            dt = time.time() - t0

            if df is None or df.height == 0:
                logs[site][short] = {"rows": 0, "elapsed_s": round(dt, 2),
                                     "ok": False}
                # Affichage condense pour eviter le bruit (32 mesures * 10 sites)
                if counter % 20 == 0 or counter == total:
                    print(f"  [{counter:>3}/{total}] {site:20s} / {short:25s}"
                          f" -> 0 rows")
                continue

            logs[site][short] = {"rows": df.height,
                                 "elapsed_s": round(dt, 2),
                                 "ok": True}
            frames_by_site[site].append(df)

            if counter % 20 == 0 or counter == total:
                print(f"  [{counter:>3}/{total}] {site:20s} / {short:25s}"
                      f" -> {df.height:>7,} rows ({dt:.1f}s)")

    client.close()

    # Fusion par site
    site_frames: list[pl.DataFrame] = []
    for site, fs in frames_by_site.items():
        if not fs:
            print(f"  [WARN] Aucune prevision pour le site '{site}' "
                  f"-> exclu")
            continue
        merged = fs[0]
        for nxt in fs[1:]:
            merged = merged.join(nxt, on=["timestamp", "site"], how="full",
                                 coalesce=True)
        site_frames.append(merged)

    if not site_frames:
        raise RuntimeError(f"Aucune prevision recuperee pour run '{prediction}'.")

    full = pl.concat(site_frames, how="diagonal_relaxed")
    full = full.sort(["site", "timestamp"])

    return full, logs


# ============================================================
# Sauvegarde
# ============================================================

def save_parquet(df: pl.DataFrame, name: str = "meteo_pred_raw") -> Path:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    out = DATA_RAW / f"{name}.parquet"
    df.write_parquet(out)
    return out


# ============================================================
# Point d'entree
# ============================================================

def main() -> int:
    print("=" * 60)
    print(f"ACQUISITION METEO PREVISIONS (run COSMO-E '{PRED_RUN}')")
    print("=" * 60)

    print(f"\nSites    : {len(SITES)} -> {SITES}")
    print(f"Mesures  : {len(MEASUREMENTS_PRED)} "
          f"(8 grandeurs * 4 statistiques)")
    print(f"Run      : '{PRED_RUN}' (00 UTC, dispo vers 03 UTC)")
    print(f"Periode  : {OIKEN_START} -> {OIKEN_END}")
    print(f"Total    : {len(SITES)*len(MEASUREMENTS_PRED)} requetes Flux")
    print()

    df, logs = fetch_meteo_pred(
        sites=SITES,
        measurements=MEASUREMENTS_PRED,
        prediction=PRED_RUN,
    )

    print("\n  --- Resume par site ---")
    for site, ms in logs.items():
        n_ok = sum(1 for v in ms.values() if v.get("ok", False))
        n_rows_total = sum(v["rows"] for v in ms.values())
        print(f"    {site:25s}: {n_ok:>2}/{len(MEASUREMENTS_PRED)} mesures, "
              f"{n_rows_total:>9,} rows total")

    print(f"\n  DataFrame final : {df.shape[0]:,} lignes, "
          f"{df.shape[1]} colonnes")
    print(f"  Sites presents  : {df['site'].n_unique()}")
    ts_min = df['timestamp'].min()
    ts_max = df['timestamp'].max()
    print(f"  Plage temporelle: {ts_min} -> {ts_max}")

    out = save_parquet(df, "meteo_pred_raw")
    print(f"\n  -> {out}")

    print("\nACQUISITION METEO PREVISIONS TERMINEE")
    return 0


if __name__ == "__main__":
    sys.exit(main())