"""
Acquisition OIKEN - lecture stricte du CSV source.

Principe: on FAIT LE MOINS POSSIBLE ici. On lit, on parse les types,
on signale les anomalies, on sauvegarde en Parquet brut. Toute
transformation semantique (timezone, imputation, fusion) est laissee
a normalization.py.

Pourquoi ?
  - Acquisition idempotente et reproductible: meme entree -> meme sortie.
  - Toute correction se fait en aval, ce qui permet de tracer
    quelle transformation a introduit quelle modification.
  - Si le CSV change, on ne touche qu'a ce module.

Sortie: data/raw/oiken_raw.parquet
  Colonnes:
    timestamp_local  (Datetime sans tz, ETIQUETTE locale Europe/Zurich)
    load             (Float64, valeur normalisee, peut contenir nulls)
    forecast_load    (Float64, baseline OIKEN, peut contenir #N/A -> null)
    pv_central_valais, pv_sion, pv_sierre, pv_remote (Float64, kWh)
    pv_total         (Float64, somme des 4 zones)

Utilisation:
    python -m src.acquisition_oiken
"""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import polars as pl

from src.config import (
    DATA_RAW,
    OIKEN_CSV_PATH,
    OIKEN_EXPECTED_ROWS,
)


# ============================================================
# Constantes locales au module
# ============================================================
# Mapping noms CSV -> noms Polars. Defini ici plutot que dans config.py
# car specifique au format OIKEN, susceptible d'evoluer si le CSV change.
COLUMN_RENAME = {
    "timestamp": "timestamp_local",
    "standardised load [-]": "load",
    "standardised forecast load [-]": "forecast_load",
    "central valais solar production [kWh]": "pv_central_valais",
    "sion area solar production [kWh]": "pv_sion",
    "sierre area production [kWh]": "pv_sierre",
    "remote solar production [kWh]": "pv_remote",
}

# Colonnes numeriques a parser (apres rename).
NUMERIC_COLUMNS = [
    "load",
    "forecast_load",
    "pv_central_valais",
    "pv_sion",
    "pv_sierre",
    "pv_remote",
]

# Marqueurs de valeur manquante observes dans le CSV.
NA_MARKERS = ["#N/A", "N/A", "", "NaN", "nan"]


# ============================================================
# Fonctions
# ============================================================

def detect_separator(path: Path) -> str:
    """
    Detecte le separateur CSV en regardant la premiere ligne.

    OIKEN exporte avec ';' (locale Excel europeenne) mais on supporte
    aussi ',' au cas ou un futur export change. Si l'en-tete contient
    les deux, ';' gagne (plus probable pour OIKEN).
    """
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline()
    if ";" in header:
        return ";"
    if "," in header:
        return ","
    raise ValueError(f"Separateur CSV non detecte dans {path}")


def load_oiken_csv(path: Path) -> pl.DataFrame:
    """
    Charge le CSV OIKEN en DataFrame Polars typé.

    Etapes:
      1. Detection separateur.
      2. Lecture en STRINGS (infer_schema_length=0) pour controler
         le parsing nous-memes -> evite les surprises sur les decimales
         et les marqueurs NA.
      3. Renommage des colonnes vers snake_case.
      4. Parsing du timestamp (format suisse DD.MM.YYYY HH:MM).
      5. Parsing des colonnes numeriques avec gestion des NA et virgule
         decimale eventuelle.
      6. Calcul de pv_total comme somme des 4 zones.
      7. Tri chronologique.

    On NE CONVERTIT PAS en UTC ici. Le timestamp reste en etiquette
    locale (datetime naif). La conversion se fait dans normalization.py
    qui gere proprement les transitions DST.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"CSV OIKEN introuvable: {path}\n"
            f"Depose le fichier oikendata.csv dans {DATA_RAW}/"
        )

    sep = detect_separator(path)
    print(f"  Separateur detecte: '{sep}'")

    # Lecture brute en strings. Bypass de l'inference de schema pour
    # eviter qu'une cellule '#N/A' transforme une colonne en utf8 puis
    # plante a l'agregation.
    df = pl.read_csv(
        path,
        separator=sep,
        infer_schema_length=0,
        null_values=NA_MARKERS,
    )

    # Verification que toutes les colonnes attendues sont la
    missing = set(COLUMN_RENAME) - set(df.columns)
    if missing:
        raise ValueError(
            f"Colonnes attendues manquantes dans {path}: {missing}\n"
            f"Colonnes trouvees: {df.columns}"
        )

    df = df.rename(COLUMN_RENAME)

    # ---- Parsing du timestamp ----
    # Format observe: "01.10.2022 00:15" ou "1.10.2022 0:15" (selon export).
    # On utilise le format strict, %d et %H acceptent 1 ou 2 chiffres.
    # Si le format change, l'erreur sera explicite (pas de cast silencieux).
    df = df.with_columns(
        pl.col("timestamp_local").str.strptime(
            pl.Datetime,
            format="%d.%m.%Y %H:%M",
            strict=True,
        )
    )

    # ---- Parsing des colonnes numeriques ----
    # Le CSV OIKEN utilise le point decimal (verifie sur l'echantillon).
    # On gere quand meme la virgule au cas ou un export change.
    # Les NA_MARKERS sont deja convertis en null par read_csv.
    for col in NUMERIC_COLUMNS:
        df = df.with_columns(
            pl.col(col)
            .str.replace(",", ".")  # virgule decimale eventuelle
            .cast(pl.Float64, strict=False)  # strict=False -> bad parse devient null
            .alias(col)
        )

    # ---- pv_total = somme des 4 zones ----
    # Si une seule zone est nulle a un instant t, pv_total devient null.
    # Comportement voulu: on prefere un null explicite a une somme partielle
    # qui sous-estimerait silencieusement la production.
    df = df.with_columns(
        (pl.col("pv_central_valais")
         + pl.col("pv_sion")
         + pl.col("pv_sierre")
         + pl.col("pv_remote")).alias("pv_total")
    )

    # ATTENTION : NE PAS TRIER PAR timestamp_local.
    # Le CSV OIKEN est deja chronologique avec une convention specifique:
    # lors du DST backward (octobre), les 4 doublons (02:15, 02:30, 02:45,
    # 03:00) sont espaces de 4 lignes dans l'ordre source (la 1ere occurrence
    # = CEST, puis 4 lignes plus tard la 2e = CET). Un sort par
    # timestamp_local rapprocherait ces doublons en les rendant adjacents,
    # ce qui CASSERAIT la reconstruction par grille UTC continue dans
    # normalize_oiken (cf bug observe: 21 desaccords au lieu de 6).
    #
    # On preserve donc l'ordre du fichier source. read_csv de Polars garantit
    # cet ordre. Si un futur export OIKEN n'etait plus chronologique, il
    # faudrait introduire une colonne row_idx avant tout tri.
    return df


def _DEPRECATED_sort_warning():
    """Marqueur pour rappeler que le sort par timestamp_local doit rester evite."""
    pass


def quality_report(df: pl.DataFrame) -> dict:
    """
    Genere un rapport de qualite synthetique sur le DataFrame OIKEN.

    Verifications:
      - Nombre de lignes vs attendu (105_120 = 3 ans en pas 15 min)
      - Nombre de nulls par colonne
      - Plage temporelle (min, max)
      - Doublons sur timestamp_local
      - Detection grossiere de discontinuites (sauts > 15 min)

    Retour: dict avec les metriques (sera persiste en JSON par le pipeline).
    """
    n_rows = df.height
    n_expected = OIKEN_EXPECTED_ROWS

    nulls_per_col = {col: df[col].null_count() for col in df.columns}

    ts_min = df["timestamp_local"].min()
    ts_max = df["timestamp_local"].max()

    # Doublons stricts sur le timestamp (en local, donc DST automne
    # va EN GENERER). C'est une info, pas une erreur ici.
    n_duplicates = n_rows - df["timestamp_local"].n_unique()

    # Detection des discontinuites: difference > 15min entre lignes
    # consecutives. NB: en heure locale, on s'attend a des sauts de
    # +1h chaque dimanche de mars (DST forward). Donc 3 sauts attendus.
    diffs = df["timestamp_local"].diff().drop_nulls()
    big_jumps = diffs.filter(diffs > timedelta(minutes=15)).len()

    report = {
        "n_rows": n_rows,
        "n_expected": n_expected,
        "n_rows_match_expected": n_rows == n_expected,
        "ts_min": str(ts_min),
        "ts_max": str(ts_max),
        "n_duplicates_local": n_duplicates,
        "n_jumps_gt_15min": big_jumps,
        "nulls_per_col": nulls_per_col,
    }
    return report


def print_report(report: dict) -> None:
    """Affiche le rapport en console de facon lisible."""
    print("\n  --- Rapport qualite OIKEN ---")
    match = "OK" if report["n_rows_match_expected"] else "ATTENTION"
    print(f"  Lignes        : {report['n_rows']:,} "
          f"(attendu {report['n_expected']:,}) [{match}]")
    print(f"  Plage         : {report['ts_min']} -> {report['ts_max']}")
    print(f"  Doublons local: {report['n_duplicates_local']} "
          f"(normal: ~12 si 3 DST automne dans la periode)")
    print(f"  Sauts > 15min : {report['n_jumps_gt_15min']} "
          f"(normal: ~3 si 3 DST printemps dans la periode)")
    print("  Nulls par colonne:")
    for col, n in report["nulls_per_col"].items():
        if n > 0:
            print(f"    {col:25s}: {n}")


def save_parquet(df: pl.DataFrame, name: str = "oiken_raw") -> Path:
    """Sauvegarde au format Parquet (compresse, typage preserve)."""
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    out = DATA_RAW / f"{name}.parquet"
    df.write_parquet(out)
    return out


# ============================================================
# Point d'entree
# ============================================================

def main() -> int:
    print("=" * 60)
    print("ACQUISITION OIKEN")
    print("=" * 60)

    print(f"\n[1/3] Lecture du CSV: {OIKEN_CSV_PATH}")
    try:
        df = load_oiken_csv(OIKEN_CSV_PATH)
    except FileNotFoundError as e:
        print(f"  ERREUR: {e}")
        return 1
    print(f"  Charge: {df.shape[0]:,} lignes, {df.shape[1]} colonnes")

    print("\n[2/3] Rapport qualite")
    report = quality_report(df)
    print_report(report)

    print("\n[3/3] Sauvegarde Parquet")
    out = save_parquet(df, "oiken_raw")
    print(f"  -> {out}")

    print("\nACQUISITION OIKEN TERMINEE")
    return 0


if __name__ == "__main__":
    sys.exit(main())