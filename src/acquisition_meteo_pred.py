"""
Acquisition InfluxDB - PREVISIONS meteo COSMO-E (multi-lead).

Telecharge les 32 variables previsionnelles (8 grandeurs * 4 statistiques
ctrl/q10/q90/stde) pour chaque site et chaque lead time configure.

CHANGEMENT MAJEUR vs version precedente (single-lead) :
  - Ancien comportement : on filtrait sur Prediction="00" (qui s'est
    avere etre un LEAD TIME=0h, pas un identifiant de run).
    Consequence : -99999 systematique sur les 12 variables cumulees.
  - Nouveau comportement : on recupere tous les leads de
    config.PRED_LEAD_TIMES (par defaut "18" a "33") et on conserve
    une colonne 'lead_time' dans le parquet.
  - La selection finale d'un lead par instant se fait en normalisation.

OPTIMISATION REQUETE :
  Une seule requete Flux par couple (site, mesure) recupere tous les
  leads en une fois, grace a pivot(rowKey:["_time", "Prediction"], ...).
  Cela maintient le nombre de requetes a 320 (= 10 sites * 32 mesures)
  au lieu d'exploser a 5'120 (320 * 16 leads).

VOLUME DE SORTIE :
  ~30'000 timestamps horaires * 16 leads = ~480k lignes par couple.
  Total estime : ~150M lignes au global, 3-5 GB en parquet compresse.

Sortie: data/raw/meteo_pred_raw.parquet
  Colonnes:
    timestamp                   (Datetime UTC)
    lead_time                   (Int8, en heures, ex: 18, 19, ..., 33)
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

import pandas as pd
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
    PRED_LEAD_TIMES,
    METEO_START,
    METEO_END,
)

warnings.simplefilter("ignore", MissingPivotFunction)


# ============================================================
# Construction de requetes Flux (multi-lead)
# ============================================================

def build_flux_query_pred(
    bucket: str,
    measurement: str,
    site: str,
    lead_times: list[str],
    start: str,
    stop: str,
) -> str:
    """
    Requete Flux pour une mesure prevue, un site, et un ENSEMBLE de
    leads. La rowKey du pivot inclut "Prediction" pour conserver une
    ligne distincte par couple (instant cible, lead).

    contains() limite l'extraction cote serveur, ce qui evite de
    telecharger les leads non vouluss.
    """
    leads_str = ", ".join(f'"{lt}"' for lt in lead_times)
    return f"""
from(bucket: "{bucket}")
  |> range(start: {start}T00:00:00Z, stop: {stop}T23:59:59Z)
  |> filter(fn: (r) => r._measurement == "{measurement}")
  |> filter(fn: (r) => r.Site == "{site}")
  |> filter(fn: (r) => contains(value: r.Prediction, set: [{leads_str}]))
  |> pivot(rowKey:["_time", "Prediction"], columnKey: ["_field"], valueColumn: "_value")
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


def _flatten_query_dataframe(pdf) -> pd.DataFrame | None:
    """
    query_data_frame peut retourner soit un DataFrame, soit une liste
    de DataFrames si le resultat est partitionne en plusieurs tables
    Flux. On normalise en un seul DataFrame Pandas.
    """
    if pdf is None:
        return None
    if isinstance(pdf, list):
        if not pdf:
            return None
        # ignore_index=True pour eviter les index dupliques entre tables
        pdf = pd.concat(pdf, ignore_index=True)
    if pdf.empty:
        return None
    return pdf


def fetch_one_pred(
    query_api,
    measurement: str,
    site: str,
    lead_times: list[str],
    start: str,
    stop: str,
) -> pl.DataFrame | None:
    """
    Telecharge UNE mesure pour UN site, sur TOUS les leads demandes,
    en une seule requete Flux. Retourne un DataFrame Polars avec
    [timestamp, lead_time, site, <short_name>] ou None si vide.
    """
    short = _short_name_pred(measurement)

    query = build_flux_query_pred(
        INFLUXDB_BUCKET, measurement, site, lead_times, start, stop
    )

    try:
        pdf = query_api.query_data_frame(query)
    except Exception as e:
        print(f"    [ERREUR Flux] {site} / {short}: "
              f"{type(e).__name__}: {e}")
        return None

    pdf = _flatten_query_dataframe(pdf)
    if pdf is None:
        return None

    # Apres pivot avec rowKey=["_time","Prediction"], on attend au moins
    # ces 3 colonnes. Si l'une manque, c'est un cas degenere (mesure
    # vide ou _field different de "Value").
    required_cols = {"_time", "Prediction", "Value"}
    if not required_cols.issubset(set(pdf.columns)):
        return None

    df = pl.from_pandas(pdf[["_time", "Prediction", "Value"]].rename(
        columns={
            "_time": "timestamp",
            "Prediction": "lead_time",
            "Value": short,
        }
    ))

    # Cast: Prediction "01" -> 1 (Int8 suffit, max 45). strict=False
    # par precaution, mais toutes les valeurs devraient etre parsables.
    df = df.with_columns([
        pl.lit(site).alias("site"),
        pl.col("lead_time").cast(pl.Int8, strict=False),
        pl.col(short).cast(pl.Float64),
    ])

    return df


def fetch_meteo_pred(
    sites: list[str],
    measurements: list[str],
    lead_times: list[str] | None = None,
    start: str = METEO_START,
    stop: str = METEO_END,
) -> tuple[pl.DataFrame, dict]:
    """
    Telecharge l'ensemble des combinaisons (site, mesure) sur tous
    les leads. Une seule requete Flux par couple, grace au pivot
    multi-rowKey.

    Retour:
      - DataFrame fusionne: (timestamp, lead_time, site) + 32 colonnes
      - dict de logs (rows par couple, leads observes, duree, statut)
    """
    if lead_times is None:
        lead_times = PRED_LEAD_TIMES

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
            df = fetch_one_pred(query_api, measurement, site,
                                lead_times, start, stop)
            dt = time.time() - t0

            if df is None or df.height == 0:
                logs[site][short] = {
                    "rows": 0, "elapsed_s": round(dt, 2), "ok": False,
                }
                # Affichage condense (32 mesures * 10 sites = 320 lignes)
                if counter % 20 == 0 or counter == total:
                    print(f"  [{counter:>3}/{total}] {site:20s} / "
                          f"{short:25s} -> 0 rows")
                continue

            n_leads_obs = df["lead_time"].n_unique()
            n_ts_obs = df["timestamp"].n_unique()

            logs[site][short] = {
                "rows": df.height,
                "n_leads_observed": n_leads_obs,
                "n_timestamps": n_ts_obs,
                "elapsed_s": round(dt, 2),
                "ok": True,
            }
            frames_by_site[site].append(df)

            if counter % 20 == 0 or counter == total:
                print(f"  [{counter:>3}/{total}] {site:20s} / "
                      f"{short:25s} -> {df.height:>9,} rows "
                      f"({n_leads_obs} leads, {dt:.1f}s)")

    client.close()

    # ---- Fusion par site ----
    # Toutes les mesures d'un site sont jointes sur (timestamp, lead_time).
    # outer join pour qu'une mesure manquante a un (t, lead) ne fasse
    # pas perdre les autres mesures du meme couple.
    site_frames: list[pl.DataFrame] = []
    for site, fs in frames_by_site.items():
        if not fs:
            print(f"  [WARN] Aucune prevision pour le site '{site}' "
                  f"-> exclu")
            continue
        merged = fs[0]
        for nxt in fs[1:]:
            merged = merged.join(
                nxt,
                on=["timestamp", "lead_time", "site"],
                how="full",
                coalesce=True,
            )
        site_frames.append(merged)

    if not site_frames:
        raise RuntimeError(
            f"Aucune prevision recuperee pour les leads {lead_times}. "
            f"Verifier connectivite InfluxDB et configuration des leads."
        )

    # ---- Empilement vertical de tous les sites ----
    full = pl.concat(site_frames, how="diagonal_relaxed")
    full = full.sort(["site", "timestamp", "lead_time"])

    return full, logs


# ============================================================
# Sauvegarde
# ============================================================

def save_parquet(df: pl.DataFrame, name: str = "meteo_pred_raw") -> Path:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    out = DATA_RAW / f"{name}.parquet"
    # compression="zstd" (defaut Polars recent) compresse bien les
    # series temporelles repetitives. Aucune option a passer ici.
    df.write_parquet(out)
    return out


# ============================================================
# Point d'entree
# ============================================================

def main() -> int:
    print("=" * 60)
    print("ACQUISITION METEO PREVISIONS (multi-lead)")
    print("=" * 60)

    print(f"\nSites      : {len(SITES)} -> {SITES}")
    print(f"Mesures    : {len(MEASUREMENTS_PRED)} "
          f"(8 grandeurs * 4 statistiques)")
    print(f"Lead times : {len(PRED_LEAD_TIMES)} valeurs "
          f"-> '{PRED_LEAD_TIMES[0]}'..'{PRED_LEAD_TIMES[-1]}'")
    print(f"Periode    : {METEO_START} -> {METEO_END}")
    print(f"Total      : {len(SITES)*len(MEASUREMENTS_PRED)} requetes "
          f"Flux (1 par couple site*mesure, tous leads en une passe)")
    print()

    df, logs = fetch_meteo_pred(
        sites=SITES,
        measurements=MEASUREMENTS_PRED,
        lead_times=PRED_LEAD_TIMES,
    )

    print("\n  --- Resume par site ---")
    for site, ms in logs.items():
        n_ok = sum(1 for v in ms.values() if v.get("ok", False))
        n_rows_total = sum(v["rows"] for v in ms.values())
        print(f"    {site:25s}: {n_ok:>2}/{len(MEASUREMENTS_PRED)} "
              f"mesures, {n_rows_total:>12,} rows total")

    print(f"\n  DataFrame final : {df.shape[0]:,} lignes, "
          f"{df.shape[1]} colonnes")
    print(f"  Sites presents  : {df['site'].n_unique()}")
    n_leads_global = df["lead_time"].n_unique()
    leads_observed = sorted(df["lead_time"].unique().to_list())
    print(f"  Leads presents  : {n_leads_global} "
          f"({leads_observed[0]}..{leads_observed[-1]})")
    ts_min = df['timestamp'].min()
    ts_max = df['timestamp'].max()
    print(f"  Plage temporelle: {ts_min} -> {ts_max}")

    out = save_parquet(df, "meteo_pred_raw")
    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"\n  -> {out} ({size_mb:.1f} MB)")

    print("\nACQUISITION METEO PREVISIONS TERMINEE")
    return 0


if __name__ == "__main__":
    sys.exit(main())