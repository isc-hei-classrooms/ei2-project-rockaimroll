"""
Acquisition InfluxDB - mesures meteo REELLES (capteurs MeteoSuisse).

Telecharge les 9 variables physiques mesurees toutes les 10 minutes,
pour chacun des sites configures dans config.SITES.

Principe:
  - Une requete Flux par couple (site, mesure). Ca semble inefficace
    mais c'est en realite plus fiable qu'une grosse requete agregee:
    on peut tracer site par site les manques, et un site qui plante
    n'arrete pas tout le pipeline.
  - Les requetes lourdes peuvent excéder le timeout par défaut.
    On force INFLUX_TIMEOUT_MS depuis config.
  - Sauvegarde en format LONG: une ligne = (timestamp, site, [9 valeurs]).
    L'agregation des sites se fera dans normalization.py.

Sortie: data/raw/meteo_real_raw.parquet
  Colonnes:
    timestamp           (Datetime UTC, pas natif 10 min)
    site                (Utf8)
    temperature_2m      (Float64)
    pressure            (Float64)
    global_radiation    (Float64)
    gust_peak           (Float64)
    precipitation       (Float64)
    humidity            (Float64)
    sunshine_duration   (Float64)
    wind_direction      (Float64)
    wind_speed          (Float64)

Utilisation:
    python -m src.acquisition_meteo_real
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
    MEASUREMENTS_REAL,
    MEASUREMENT_SHORT_NAMES,
    OIKEN_START,
    OIKEN_END,
)

# La librairie influxdb-client emet un warning chaque fois qu'on n'utilise
# pas pivot() dans le SCRIPT (alors qu'on l'utilise dans la requete Flux).
# Faux positif, on tait.
warnings.simplefilter("ignore", MissingPivotFunction)


# ============================================================
# Construction de requetes Flux
# ============================================================

def build_flux_query_real(
    bucket: str,
    measurement: str,
    site: str,
    start: str,
    stop: str,
) -> str:
    """
    Construit une requete Flux pour une mesure reelle a un site donne.

    Le pipeline Flux:
      from(bucket)        -> selection bucket
      |> range            -> fenetre temporelle (UTC)
      |> filter _measurement = mesure
      |> filter Site = site
      |> pivot _field -> colonnes (le _field unique 'Value' devient col)

    Note: on UTILISE des f-strings ici. C'est sur car les valeurs
    (measurement, site) viennent de config.py controlee, pas d'input
    utilisateur. Si un jour ces valeurs etaient externes, il faudrait
    parametrer la requete (parametres Flux) pour eviter une injection.
    """
    return f"""
from(bucket: "{bucket}")
  |> range(start: {start}T00:00:00Z, stop: {stop}T23:59:59Z)
  |> filter(fn: (r) => r._measurement == "{measurement}")
  |> filter(fn: (r) => r.Site == "{site}")
  |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
""".strip()


# ============================================================
# Telechargement
# ============================================================

def fetch_one(
    query_api,
    measurement: str,
    site: str,
    start: str,
    stop: str,
) -> pl.DataFrame | None:
    """
    Telecharge une combinaison (mesure, site) et retourne un DataFrame
    Polars avec colonnes [timestamp, site, <nom_court>].

    Retourne None si vide ou erreur (a logger en aval).
    """
    col_short = MEASUREMENT_SHORT_NAMES.get(measurement, measurement.lower())

    query = build_flux_query_real(
        INFLUXDB_BUCKET, measurement, site, start, stop
    )

    try:
        # query_data_frame retourne du Pandas (seule API disponible).
        # Conversion vers Polars juste apres.
        pdf = query_api.query_data_frame(query)
    except Exception as e:
        print(f"    [ERREUR Flux] {site} / {col_short}: {type(e).__name__}: {e}")
        return None

    if pdf is None or len(pdf) == 0 or pdf.empty:
        return None

    # InfluxDB retourne au moins 'Value' apres pivot. On garde _time + Value,
    # on renomme.
    if "Value" not in pdf.columns:
        # Cas tres rare ou le pivot n'a pas marche (pas de _field "Value").
        # On l'observe en exploration mais pas dans nos vraies requetes.
        return None

    df = pl.from_pandas(pdf[["_time", "Value"]].rename(
        columns={"_time": "timestamp", "Value": col_short}
    ))

    # Ajout de la colonne site (constante sur toutes les lignes du frame)
    df = df.with_columns(pl.lit(site).alias("site"))

    # Cast explicite du type valeur en Float64 (Polars peut deduire Int64
    # si toutes les valeurs sont entieres, ce qui fait planter le concat).
    df = df.with_columns(pl.col(col_short).cast(pl.Float64))

    return df


def fetch_meteo_real(
    sites: list[str],
    measurements: list[str],
    start: str = OIKEN_START,
    stop: str = OIKEN_END,
) -> tuple[pl.DataFrame, dict]:
    """
    Telecharge l'ensemble des combinaisons (site, mesure).

    Retour:
      - DataFrame fusionne en format LONG (timestamp, site, valeurs)
      - dict de logs par requete (lignes recuperees, duree, erreurs)
    """
    client = InfluxDBClient(
        url=INFLUXDB_URL,
        token=INFLUXDB_TOKEN,
        org=INFLUXDB_ORG,
        timeout=INFLUX_TIMEOUT_MS,
    )
    query_api = client.query_api()

    # Structure: dict[site][mesure] = {"rows": int, "elapsed_s": float}
    logs: dict[str, dict] = {s: {} for s in sites}

    # On stocke un DataFrame par couple (site, mesure), puis on fait
    # un merge final en deux temps (par site puis empilement).
    frames_by_site: dict[str, list[pl.DataFrame]] = {s: [] for s in sites}

    total = len(sites) * len(measurements)
    counter = 0

    for site in sites:
        for measurement in measurements:
            counter += 1
            short = MEASUREMENT_SHORT_NAMES.get(measurement,
                                                measurement.lower())
            t0 = time.time()
            df = fetch_one(query_api, measurement, site, start, stop)
            dt = time.time() - t0

            if df is None or df.height == 0:
                logs[site][short] = {"rows": 0, "elapsed_s": round(dt, 2),
                                     "ok": False}
                print(f"  [{counter:>3}/{total}] {site:20s} / {short:18s}"
                      f" -> 0 rows ({dt:.1f}s)")
                continue

            logs[site][short] = {"rows": df.height,
                                 "elapsed_s": round(dt, 2),
                                 "ok": True}
            frames_by_site[site].append(df)
            print(f"  [{counter:>3}/{total}] {site:20s} / {short:18s}"
                  f" -> {df.height:>7,} rows ({dt:.1f}s)")

    client.close()

    # ---- Fusion par site (toutes mesures d'un site sur le meme timestamp) ----
    # outer join entre les mesures: si une mesure manque a un timestamp
    # donne, sa valeur est null mais les autres sont gardees.
    site_frames: list[pl.DataFrame] = []
    for site, fs in frames_by_site.items():
        if not fs:
            print(f"  [WARN] Aucune donnee pour le site '{site}' "
                  f"-> exclu de la sortie")
            continue
        merged = fs[0]
        for nxt in fs[1:]:
            merged = merged.join(nxt, on=["timestamp", "site"], how="full",
                                 coalesce=True)
        site_frames.append(merged)

    if not site_frames:
        raise RuntimeError("Aucune donnee meteo recuperee. "
                           "Verifier connexion InfluxDB et token.")

    # ---- Empilement vertical de tous les sites ----
    # diagonal_relaxed: tolere des colonnes manquantes (un site peut ne
    # pas avoir une mesure; les autres sites gardent leurs colonnes).
    full = pl.concat(site_frames, how="diagonal_relaxed")

    # Tri pour une serialisation deterministe
    full = full.sort(["site", "timestamp"])

    return full, logs


# ============================================================
# Sauvegarde
# ============================================================

def save_parquet(df: pl.DataFrame, name: str = "meteo_real_raw") -> Path:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    out = DATA_RAW / f"{name}.parquet"
    df.write_parquet(out)
    return out


# ============================================================
# Point d'entree
# ============================================================

def main() -> int:
    print("=" * 60)
    print("ACQUISITION METEO REELLE (InfluxDB)")
    print("=" * 60)

    print(f"\nSites    : {len(SITES)} -> {SITES}")
    print(f"Mesures  : {len(MEASUREMENTS_REAL)} (9 variables physiques)")
    print(f"Periode  : {OIKEN_START} -> {OIKEN_END}")
    print(f"Total    : {len(SITES)*len(MEASUREMENTS_REAL)} requetes Flux")
    print()

    df, logs = fetch_meteo_real(
        sites=SITES,
        measurements=MEASUREMENTS_REAL,
    )

    print("\n  --- Resume par site ---")
    for site, ms in logs.items():
        n_ok = sum(1 for v in ms.values() if v.get("ok", False))
        n_rows_total = sum(v["rows"] for v in ms.values())
        print(f"    {site:25s}: {n_ok:>2}/{len(MEASUREMENTS_REAL)} mesures, "
              f"{n_rows_total:>9,} rows total")

    print(f"\n  DataFrame final : {df.shape[0]:,} lignes, "
          f"{df.shape[1]} colonnes")
    print(f"  Sites presents  : {df['site'].n_unique()}")
    ts_min = df['timestamp'].min()
    ts_max = df['timestamp'].max()
    print(f"  Plage temporelle: {ts_min} -> {ts_max}")

    out = save_parquet(df, "meteo_real_raw")
    print(f"\n  -> {out}")

    print("\nACQUISITION METEO REELLE TERMINEE")
    return 0


if __name__ == "__main__":
    sys.exit(main())