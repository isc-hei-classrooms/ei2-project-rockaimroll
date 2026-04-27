"""
Pipeline de normalisation - prend les 3 Parquet bruts et produit un
dataset propre, fusionne, en grille UTC reguliere de 15 min.

Etapes:
  1. OIKEN -> UTC (strategie HYBRIDE: reconstruction grille + verif local)
  2. Meteo real et pred -> garantie UTC
  3. Detection d'outliers (bornes physiques + sigma) -> nullification
  4. Resampling sur grille 15 min (real 10min, pred horaire)
  5. Agregation des sites (moyenne, mediane, ou moyenne circulaire selon
     la nature de la variable)
  6. Fusion OIKEN x meteo_real x meteo_pred sur timestamp UTC
  7. Rapport de qualite

Sortie:
  data/processed/dataset_normalized.parquet
  data/reports/normalization_report.json

Strategie OIKEN (hybride):
  - Le CSV OIKEN est en grille LOCALE Europe/Zurich, mais en pratique il
    est continu en UTC (105'120 lignes = 1095 jours * 96 quart-d'heures,
    les decalages DST forward et backward s'annulant exactement).
  - On reconstruit donc une grille UTC continue de N lignes a partir du
    premier timestamp local (= 01.10.2022 00:15 CEST = 30.09.2022 22:15 UTC).
  - On VERIFIE ensuite la coherence: pour chaque ligne, on convertit
    l'UTC reconstruit en local et on compare au timestamp_local du CSV.
  - Desaccords attendus: 1 par DST forward + 1 par DST backward, soit
    ~6 par 3 ans. Cause: la "ligne du saut" (UTC pile 01:00) correspond
    a 2 etiquettes locales possibles (avant/apres saut). OIKEN choisit
    le cote "avant-saut" (02:00 CET en mars, 03:00 CEST en octobre);
    zoneinfo choisit le cote "apres-saut" (03:00 CEST en mars, 02:00 CET
    en octobre). Convention differente, meme instant physique. Verifie
    sur le CSV reel: 6 desaccords detectes sur 3 ans.
  - Si le nombre de desaccords excede DST_MISMATCH_TOLERANCE, c'est
    qu'OIKEN a un trou dans la grille UTC -> bascule sur fallback.

Correction additionnelle DST sur forecast_load:
  - Diagnostic empirique (balayage MAE par shift): la baseline OIKEN
    est decalee de -1h en periode CET (hiver), shift=0 en periode CEST.
  - fix_forecast_load_dst_shift() applique un shift de +4 pas (=1h) sur
    forecast_load uniquement pour les lignes CET, en preservant les
    lignes CEST inchangees.

Utilisation:
    python -m src.normalization
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

from src.config import (
    DATA_RAW,
    DATA_PROCESSED,
    DATA_REPORTS,
    PHYSICAL_BOUNDS,
    LOAD_OUTLIER_SIGMA,
    PV_MIN_KWH,
    PV_MAX_KWH,
    TIMEZONE_OIKEN,
    TARGET_RESOLUTION,
    OIKEN_EXPECTED_ROWS,
)


# ============================================================
# Constantes locales : taxonomie des colonnes
# ============================================================
# La nature physique de chaque colonne meteo determine la methode
# d'agregation et de resampling appropriee.

# Variables instantanees: la valeur represente l'etat a l'instant t.
# Agregation par moyenne arithmetique sur les sites.
INSTANT_COLS = [
    "temperature_2m",
    "pressure",
    "humidity",
    "wind_speed",
    "global_radiation",
    "gust_peak",
]

# Variables cumulees sur 10 min : on perd peu d'information en faisant
# la moyenne plutot que la somme, MAIS la moyenne sous-estime quand on
# resample. Compromis pris ici: on les traite comme instantanees pour
# le resampling 10min->15min, en assumant un biais residuel acceptable.
# A raffiner si necessaire.
CUMULATIVE_COLS = [
    "precipitation",
    "sunshine_duration",
]

# Variables circulaires (angles): wind_direction. Necessite une moyenne
# circulaire (atan2(mean(sin), mean(cos))) sinon 350° et 10° donnent une
# moyenne de 180° au lieu de 0°.
CIRCULAR_COLS = [
    "wind_direction",
]

ALL_METEO_COLS = INSTANT_COLS + CUMULATIVE_COLS + CIRCULAR_COLS

# Limite d'interpolation: ne PAS interpoler des trous > 4 pas (= 1h)
# pour les valeurs continues. Au-dela, on laisse null et le modele gere.
# [Inference raisonnable] 1h est un compromis entre robustesse (eviter
# d'inventer des donnees sur 6h) et utilisabilite (combler les micro-trous).
INTERPOLATION_GAP_LIMIT = 4

# Tolerance sur les desaccords de label local OIKEN (DST forward + backward)
# 6 desaccords confirmes sur le CSV reel (3 ans). On accepte jusqu'a 10
# par securite, au-dela on bascule sur le fallback.
DST_MISMATCH_TOLERANCE = 10

# Nombre de pas de 15 min pour 1h (utilise par le DST shift fix).
DST_SHIFT_PAS = 4


# ============================================================
# 1. OIKEN -> UTC (strategie hybride)
# ============================================================

def reconstruct_oiken_utc(df: pl.DataFrame) -> pl.DataFrame:
    """
    Reconstruit les timestamps UTC en supposant une grille continue.

    Hypothese: le CSV OIKEN est continu en UTC (pas de quart d'heure
    manquant). C'est verifie par cross-check avec le label local en
    aval (verify_local_consistency).

    Le premier timestamp local est attendu en CEST (heure d'ete) car
    octobre 2022 est avant le DST backward (dernier dimanche d'octobre).

    Retour: meme DataFrame avec une colonne 'timestamp' (Datetime UTC).
    La colonne 'timestamp_local' est conservee pour la verification.
    """
    n = df.height
    if n == 0:
        raise ValueError("DataFrame OIKEN vide")

    # Premier timestamp local du CSV (suppose le premier en chronologique)
    first_local_naive = df["timestamp_local"][0]
    if not isinstance(first_local_naive, datetime):
        # Polars peut retourner soit datetime soit str selon version
        raise TypeError(
            f"timestamp_local n'est pas Datetime: {type(first_local_naive)}"
        )

    # Convertir en UTC en ATTACHANT le fuseau Europe/Zurich.
    # ZoneInfo gere automatiquement DST. On utilise fold=0 par defaut
    # (donc en cas d'ambiguite DST backward, on prend la 1ere occurrence
    # = CEST). Ici le 01.10 n'est pas un dimanche DST, donc non ambigu.
    zurich = ZoneInfo(TIMEZONE_OIKEN)
    first_local_aware = first_local_naive.replace(tzinfo=zurich)
    first_utc_aware = first_local_aware.astimezone(timezone.utc)
    first_utc_naive = first_utc_aware.replace(tzinfo=None)

    # Generer la grille UTC continue: N lignes en pas 15 min
    # end = first + (n-1) * 15 min ; on inclut end donc on a bien n lignes.
    last_utc_naive = first_utc_naive + timedelta(minutes=15) * (n - 1)

    grid_utc = pl.datetime_range(
        start=first_utc_naive,
        end=last_utc_naive,
        interval="15m",
        time_zone="UTC",
        eager=True,
    )

    if len(grid_utc) != n:
        raise RuntimeError(
            f"Grille UTC reconstruite a {len(grid_utc)} lignes, "
            f"attendu {n}. Probleme de pas de temps?"
        )

    # Attacher la grille au DataFrame
    df_utc = df.with_columns(grid_utc.alias("timestamp"))
    return df_utc


def verify_local_consistency(df: pl.DataFrame) -> dict:
    """
    Verifie que la conversion UTC -> local Europe/Zurich correspond
    aux labels locaux du CSV.

    Desaccords attendus: 1 par DST forward (instant du saut, etiquete
    "02:00 CET" dans le CSV mais converti en "03:00 CEST" par zoneinfo).

    Retour: dict avec
      - n_mismatches: nombre total de desaccords
      - tolerance: seuil
      - is_consistent: bool
      - first_mismatches: liste des 5 premiers (pour debug)
    """
    # Recalculer le label local depuis l'UTC reconstruit
    df_check = df.with_columns(
        pl.col("timestamp")
        .dt.convert_time_zone(TIMEZONE_OIKEN)
        .dt.replace_time_zone(None)  # naif pour comparaison directe
        .alias("recalc_local")
    )

    mismatches = df_check.filter(
        pl.col("timestamp_local") != pl.col("recalc_local")
    )
    n_mismatches = mismatches.height

    first_examples = []
    if n_mismatches > 0:
        sample = mismatches.head(5).select([
            "timestamp", "timestamp_local", "recalc_local"
        ])
        for row in sample.iter_rows(named=True):
            first_examples.append({
                "utc": str(row["timestamp"]),
                "csv_local": str(row["timestamp_local"]),
                "recalc_local": str(row["recalc_local"]),
            })

    return {
        "n_mismatches": n_mismatches,
        "tolerance": DST_MISMATCH_TOLERANCE,
        "is_consistent": n_mismatches <= DST_MISMATCH_TOLERANCE,
        "first_examples": first_examples,
    }


def fix_forecast_load_dst_shift(df: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    """
    Corrige le decalage de 1h de forecast_load en periode CET (hiver).

    Diagnostic confirme par balayage empirique du shift sur 5 saisons:
        shift_pas  shift_min  MAE_hiver  MAE_ete
                0         0m   0.2254    0.1616  <- ete optimal a shift=0
                4       +60m   0.1669    0.2359  <- hiver optimal a +1h

    L'optimum est shift=+4 pas (= +1h) en hiver, shift=0 en ete.
    Cela signifie que la prevision OIKEN pour l'instant i correspond
    en realite au moment i+4 en CET. Autrement dit, la baseline est
    placee 1h trop tot dans le CSV en periode CET.

    [Hypothese non confirmee] OIKEN genere probablement ses previsions
    avec une horloge interne ne respectant pas DST puis aligne les
    valeurs sur le timestamp_local du CSV. Consequence operationnelle:
    decalage saisonnier de 1h.

    Methode :
      - Detection CET/CEST via dt.dst_offset() (vectorise Polars):
          0s    -> CET (hiver), pas de DST applique
          3600s -> CEST (ete), DST applique
      - Pour les lignes CET : forecast_load[i] <- forecast_load[i-4]
        (shift(4) en Polars = decale les valeurs vers le bas)
      - Pour les lignes CEST : aucun changement

    Effet de bord aux transitions CEST -> CET (dimanche fin octobre):
    les 4 premieres lignes CET apres la transition heritent des 4
    dernieres valeurs CEST. C'est exactement ce qu'on veut puisque ces
    valeurs CEST etaient deja "1h en avance" dans le decalage OIKEN, et
    correspondent donc bien aux 4 premieres lignes CET.

    Effet de bord en debut de dataset: si les 4 premieres lignes du
    dataset sont en CET, elles deviennent null car shift(4) cherche des
    lignes inexistantes. Pour OIKEN dataset commencant en octobre CEST,
    ce cas n'arrive pas.

    A appliquer APRES interpolation des nulls existants pour ne pas
    shifter de nulls. Les eventuels nouveaux nulls crees seront filtres
    par features.py.

    Retour: (DataFrame corrige, log dict).
    """
    df_aug = df.with_columns([
        pl.col("timestamp")
          .dt.convert_time_zone(TIMEZONE_OIKEN)
          .dt.dst_offset()
          .dt.total_seconds()
          .alias("_dst_offset_s"),
        pl.col("forecast_load").shift(DST_SHIFT_PAS)
          .alias("_forecast_shifted_4"),
    ])

    n_cet = df_aug.filter(pl.col("_dst_offset_s") == 0).height
    n_cest = df_aug.filter(pl.col("_dst_offset_s") > 0).height
    n_null_before = df.filter(pl.col("forecast_load").is_null()).height

    df_fixed = df_aug.with_columns(
        pl.when(pl.col("_dst_offset_s") == 0)
          .then(pl.col("_forecast_shifted_4"))
          .otherwise(pl.col("forecast_load"))
          .alias("forecast_load")
    ).drop("_dst_offset_s", "_forecast_shifted_4")

    n_null_after = df_fixed.filter(
        pl.col("forecast_load").is_null()
    ).height

    log = {
        "n_cet_lines": int(n_cet),
        "n_cest_lines": int(n_cest),
        "shift_pas": int(DST_SHIFT_PAS),
        "n_null_before_shift": int(n_null_before),
        "n_null_after_shift": int(n_null_after),
        "n_null_added_by_shift": int(n_null_after - n_null_before),
    }
    return df_fixed, log


def fallback_oiken_utc(df: pl.DataFrame) -> pl.DataFrame:
    """
    Methode de repli si la reconstruction par grille echoue.

    Probleme du fallback naif (ancien code): replace_time_zone avec
    ambiguous="earliest" assigne CEST a TOUTES les occurrences des doublons
    DST backward, donc les deux occurrences se retrouvent au meme UTC et
    unique(keep="first") supprime les valeurs CET. Bug critique sur 4
    valeurs par DST backward (= 12 lignes perdues sur 3 ans).

    Fix: on numerote chaque doublon avec cum_count (par groupe de
    timestamp_local), puis on traite separement:
      - dup_idx == 0 (1ere occurrence)  -> ambiguous="earliest" (CEST)
      - dup_idx == 1 (2e occurrence)    -> ambiguous="latest"   (CET)
    Les deux sous-DataFrames sont concatenes, ce qui restitue les bons UTC
    distincts (00:15 CEST vs 01:15 CET pour le meme label local 02:15).

    LIMITATION RESIDUELLE: pour le label "03:00" du jour DST backward, ce
    label n'est pas ambigu en local (n'existe qu'en CET). Donc replace_time_zone
    le convertit toujours en UTC 02:00 (CET), peu importe ambiguous=earliest
    ou latest. Resultat: la "1ere occurrence 03:00" (= instant du saut, valeur
    CEST) est mal localisee a UTC 02:00 et collisionne avec la "2e occurrence"
    qui est la vraie valeur CET. unique(keep="first") garde la 1ere et perd
    la 2e. Soit 1 ligne perdue par DST backward (3 sur 3 ans).
    Cette limitation est intrinseque a replace_time_zone ; seule la
    reconstruction par grille UTC continue (strategie principale) la resout.

    Pour le DST forward, non_existent="null" supprime les lignes "fantomes"
    eventuelles. En pratique, le CSV OIKEN n'en a pas (verifie sur 3 ans).
    """
    # Numeroter chaque ligne au sein de son groupe de timestamp_local
    # cum_count() commence a 1, on retire 1 pour avoir 0-indexed.
    df = df.with_columns(
        (pl.col("timestamp_local").cum_count().over("timestamp_local") - 1)
        .alias("_dup_idx")
    )

    # Branche 1: premieres occurrences (CEST en cas d'ambiguite)
    df_first = df.filter(pl.col("_dup_idx") == 0).with_columns(
        pl.col("timestamp_local")
        .dt.replace_time_zone(
            TIMEZONE_OIKEN, ambiguous="earliest", non_existent="null"
        )
        .dt.convert_time_zone("UTC")
        .alias("timestamp")
    )

    # Branche 2: deuxiemes occurrences (CET, uniquement DST backward)
    df_second = df.filter(pl.col("_dup_idx") == 1).with_columns(
        pl.col("timestamp_local")
        .dt.replace_time_zone(
            TIMEZONE_OIKEN, ambiguous="latest", non_existent="null"
        )
        .dt.convert_time_zone("UTC")
        .alias("timestamp")
    )

    # Concatenation et nettoyage
    df_out = pl.concat([df_first, df_second]).drop("_dup_idx")
    df_out = df_out.filter(pl.col("timestamp").is_not_null())
    # Tri sur l'UTC final (different du tri par label local : aucun doublon ici)
    df_out = df_out.unique(subset=["timestamp"], keep="first").sort("timestamp")
    return df_out


def normalize_oiken(df: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    """
    Normalisation complete OIKEN.

    Etapes:
      1. Reconstruction grille UTC + verification.
      2. Si coherent: on garde la grille reconstruite.
      3. Sinon: fallback replace_time_zone classique.
      4. Validation des valeurs PV (>= 0).
      5. Detection outliers statistique sur load (sigma).
      6. Interpolation des nulls existants de forecast_load.
      7. Fix DST shift sur forecast_load (decalage 1h en CET).
    """
    log = {"strategy": None, "verification": None}

    # --- Etape 1: reconstruction ---
    df_recon = reconstruct_oiken_utc(df)
    verif = verify_local_consistency(df_recon)
    log["verification"] = verif

    if verif["is_consistent"]:
        log["strategy"] = "grid_reconstruction"
        out = df_recon.drop("timestamp_local")
    else:
        # Fallback: la grille continue n'a pas marche, OIKEN a un trou
        # ou un decalage. On utilise replace_time_zone qui gere cas par cas.
        log["strategy"] = "replace_time_zone_fallback"
        log["fallback_reason"] = (
            f"{verif['n_mismatches']} desaccords > tolerance "
            f"{DST_MISMATCH_TOLERANCE}"
        )
        out = fallback_oiken_utc(df)

    # --- Etape 2: PV non negatif ---
    pv_cols = ["pv_central_valais", "pv_sion", "pv_sierre",
               "pv_remote", "pv_total"]
    for c in pv_cols:
        if c in out.columns:
            # Nullify les valeurs hors bornes (negatif ou >> max)
            out = out.with_columns(
                pl.when(
                    (pl.col(c) < PV_MIN_KWH) | (pl.col(c) > PV_MAX_KWH)
                )
                .then(None)
                .otherwise(pl.col(c))
                .alias(c)
            )

    # --- Etape 3: outliers statistique sur load ---
    # |z-score| > LOAD_OUTLIER_SIGMA -> nullify
    load_mean = out["load"].mean()
    load_std = out["load"].std()
    if load_mean is not None and load_std is not None and load_std > 0:
        z_threshold = LOAD_OUTLIER_SIGMA
        n_before = out["load"].null_count()
        out = out.with_columns(
            pl.when(
                ((pl.col("load") - load_mean) / load_std).abs() > z_threshold
            )
            .then(None)
            .otherwise(pl.col("load"))
            .alias("load")
        )
        n_after = out["load"].null_count()
        log["load_outliers_nullified"] = n_after - n_before
    else:
        log["load_outliers_nullified"] = 0

    # --- Etape 4: forecast_load nulls (12 #N/A connus) ---
    log["forecast_load_nulls_before_interp"] = out["forecast_load"].null_count()
    out = out.with_columns(pl.col("forecast_load").interpolate())
    log["forecast_load_nulls_after_interp"] = out["forecast_load"].null_count()

    # --- Etape 5: DST shift fix sur forecast_load ---
    # OIKEN encode forecast_load avec un decalage de 1h en CET (hiver),
    # diagnostique via balayage empirique de la MAE par shift saisonnier.
    # A appliquer APRES interpolation pour ne pas shifter de nulls.
    out, dst_log = fix_forecast_load_dst_shift(out)
    log["dst_shift_fix"] = dst_log

    # Tri final + dedup eventuel (au cas ou fallback)
    out = out.unique(subset=["timestamp"], keep="first").sort("timestamp")

    log["n_rows_out"] = out.height
    log["ts_min"] = str(out["timestamp"].min())
    log["ts_max"] = str(out["timestamp"].max())

    return out, log


# ============================================================
# 2. Meteo -> UTC (simple)
# ============================================================

def to_utc_meteo(df: pl.DataFrame) -> pl.DataFrame:
    """
    Garantit que la colonne 'timestamp' a le dtype Datetime UTC.

    InfluxDB renvoie deja en UTC, mais selon la version du driver, le
    type Polars peut etre soit Datetime("ns", "UTC") soit Datetime("ns", None).
    On force le second cas a etre marque UTC SANS conversion (les valeurs
    sont deja UTC, on attache juste l'etiquette).
    """
    dtype = df["timestamp"].dtype

    if isinstance(dtype, pl.Datetime) and dtype.time_zone == "UTC":
        return df

    if isinstance(dtype, pl.Datetime) and dtype.time_zone is None:
        return df.with_columns(pl.col("timestamp").dt.replace_time_zone("UTC"))

    # Cas extreme: deja un autre fuseau, on convertit
    return df.with_columns(pl.col("timestamp").dt.convert_time_zone("UTC"))


# ============================================================
# 3. Outliers physiques (bornes config)
# ============================================================

def clip_physical_outliers(df: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    """
    Met a null les valeurs hors des bornes physiques definies dans config.

    Different de clip(): on PERD l'information mais on n'INVENTE pas
    une borne saturee qui biaiserait la moyenne. La nullification permet
    a l'imputation aval de reconstruire.

    Retour: (df_modifie, dict des comptages par colonne).
    """
    counts = {}
    for col, (lo, hi) in PHYSICAL_BOUNDS.items():
        if col not in df.columns:
            continue
        n_before = df[col].null_count()
        df = df.with_columns(
            pl.when((pl.col(col) < lo) | (pl.col(col) > hi))
            .then(None)
            .otherwise(pl.col(col))
            .alias(col)
        )
        n_after = df[col].null_count()
        counts[col] = n_after - n_before
    return df, counts


# ============================================================
# 4. Imputation avec limite de gap
# ============================================================

def interpolate_safe(df: pl.DataFrame, cols: list[str],
                     gap_limit: int = INTERPOLATION_GAP_LIMIT
                     ) -> pl.DataFrame:
    """
    Interpolation lineaire mais SEULEMENT pour les trous <= gap_limit.

    Au-dela, les nulls sont preserves (le modele les gerera ou on
    appliquera une strategie saisonniere ulterieurement).

    Implementation:
      1. Pour chaque colonne, identifier les "runs" de nulls consecutifs.
      2. Calculer la longueur de chaque run.
      3. Interpoler entierement, puis remettre a null les positions
         dont le run depasse gap_limit.

    [Choix de design] On utilise interpolate() puis masquage. Plus
    couteux que de faire l'interpolation conditionnelle, mais beaucoup
    plus simple et lisible. Sur 100k lignes, le surcout est negligeable.
    """
    for col in cols:
        if col not in df.columns:
            continue

        # Calcul de la longueur du run de null courant pour chaque ligne
        # Strategie: cumsum des transitions, puis count par groupe.
        is_null = df[col].is_null()

        # Identifiant de groupe: change a chaque transition null/non-null
        # Implementation: cumsum des changements
        df = df.with_columns([
            is_null.alias(f"_{col}_isnull"),
            (is_null != is_null.shift(1, fill_value=is_null.first()))
            .cum_sum()
            .alias(f"_{col}_grp"),
        ])

        # Pour chaque groupe, compter la taille (=longueur du run)
        run_len = df.group_by(f"_{col}_grp").agg(
            pl.len().alias(f"_{col}_runlen")
        )
        df = df.join(run_len, on=f"_{col}_grp", how="left")

        # Interpolation complete
        df = df.with_columns(pl.col(col).interpolate().alias(col))

        # Remettre a null les positions dont le run depasse la limite
        df = df.with_columns(
            pl.when(
                pl.col(f"_{col}_isnull") & (pl.col(f"_{col}_runlen") > gap_limit)
            )
            .then(None)
            .otherwise(pl.col(col))
            .alias(col)
        )

        # Cleanup colonnes temporaires
        df = df.drop([f"_{col}_isnull", f"_{col}_grp", f"_{col}_runlen"])

    return df


# ============================================================
# 5. Resampling 15 min
# ============================================================

def resample_real_to_15min(df: pl.DataFrame) -> pl.DataFrame:
    """
    Resample les mesures reelles (pas natif 10 min) vers 15 min.

    Pour chaque site:
      1. Cree une grille 15 min sur la plage du site.
      2. Joint les valeurs 10 min via join_asof (nearest, tolerance 10 min).
      3. Interpole les petits trous.

    [Limitation] Les colonnes cumulatives (precipitation, sunshine) sont
    traitees comme instantanees ici. L'erreur introduite est faible mais
    pas nulle (voir CUMULATIVE_COLS). Optimisation possible: agregation
    par fenetre 15min via group_by_dynamic, mais l'alignement sur les
    multiples de 15 min sur des donnees 10 min cree des fenetres a 1 ou
    2 points ce qui complique.
    """
    sites = df["site"].unique().to_list()
    site_frames = []

    for site in sites:
        df_site = df.filter(pl.col("site") == site).sort("timestamp")
        if df_site.height == 0:
            continue

        # Grille 15 min alignee sur les multiples de 15 min UTC
        t_min = df_site["timestamp"].min()
        t_max = df_site["timestamp"].max()

        # Aligner t_min sur le multiple de 15 min superieur
        # (eviter les decalages bizarres)
        t_min_aligned = _align_to_15min(t_min, ceil=True)
        t_max_aligned = _align_to_15min(t_max, ceil=False)

        grid = pl.DataFrame({
            "timestamp": pl.datetime_range(
                start=t_min_aligned,
                end=t_max_aligned,
                interval="15m",
                time_zone="UTC",
                eager=True,
            )
        })

        # join_asof: pour chaque timestamp grille, prend la valeur
        # 10 min la plus proche, tolerance = 10 min (= un pas natif)
        df_site_only_vals = df_site.drop("site")
        joined = grid.join_asof(
            df_site_only_vals,
            on="timestamp",
            strategy="nearest",
            tolerance="10m",
        )
        joined = joined.with_columns(pl.lit(site).alias("site"))

        # Interpolation des petits trous
        value_cols = [c for c in joined.columns if c not in ("timestamp", "site")]
        joined = interpolate_safe(joined, value_cols)

        site_frames.append(joined)

    if not site_frames:
        return df.head(0)

    return pl.concat(site_frames, how="diagonal_relaxed").sort(["site", "timestamp"])


def resample_pred_to_15min(df: pl.DataFrame) -> pl.DataFrame:
    """
    Resample les previsions COSMO-E (pas natif horaire ou 3h) vers 15 min.

    Methode: pour chaque site, creer la grille 15 min et interpoler
    lineairement entre les points natifs. join_asof n'est pas adapte
    car le pas natif est trop large (1h voire 3h), une valeur asof
    serait constante par paliers d'1h alors qu'on veut une transition
    progressive.
    """
    sites = df["site"].unique().to_list()
    site_frames = []

    for site in sites:
        df_site = df.filter(pl.col("site") == site).sort("timestamp")
        if df_site.height == 0:
            continue

        t_min = df_site["timestamp"].min()
        t_max = df_site["timestamp"].max()
        t_min_aligned = _align_to_15min(t_min, ceil=True)
        t_max_aligned = _align_to_15min(t_max, ceil=False)

        grid = pl.DataFrame({
            "timestamp": pl.datetime_range(
                start=t_min_aligned,
                end=t_max_aligned,
                interval="15m",
                time_zone="UTC",
                eager=True,
            )
        })

        df_site_only_vals = df_site.drop("site")

        # left join: les timestamps grille qui matchent un timestamp pred
        # gardent la valeur, les autres deviennent null
        joined = grid.join(df_site_only_vals, on="timestamp", how="left")

        # Interpolation lineaire entre les points pred (qui sont a 1h)
        # On accepte une limite plus large ici (24 = 6h) car les previsions
        # changent lentement.
        value_cols = [c for c in joined.columns if c != "timestamp"]
        joined = interpolate_safe(joined, value_cols, gap_limit=24)

        joined = joined.with_columns(pl.lit(site).alias("site"))
        site_frames.append(joined)

    if not site_frames:
        return df.head(0)

    return pl.concat(site_frames, how="diagonal_relaxed").sort(["site", "timestamp"])


def _align_to_15min(dt_value, ceil: bool):
    """
    Aligne un datetime sur le multiple de 15 min le plus proche.
    ceil=True: vers le haut. ceil=False: vers le bas.
    """
    # Polars datetime peut etre cast en Python datetime via to_pydatetime
    if not isinstance(dt_value, datetime):
        # Conversion via .replace
        return dt_value
    minute = dt_value.minute
    rest = minute % 15
    if rest == 0:
        return dt_value.replace(second=0, microsecond=0)
    if ceil:
        delta = 15 - rest
        return (dt_value + timedelta(minutes=delta)).replace(second=0, microsecond=0)
    else:
        return (dt_value - timedelta(minutes=rest)).replace(second=0, microsecond=0)


# ============================================================
# 6. Agregation des sites (mean, median, circular)
# ============================================================

def aggregate_meteo_sites(df: pl.DataFrame, prefix: str = "") -> pl.DataFrame:
    """
    Agrege les valeurs des differents sites en une seule serie.

    Methode par type:
      - INSTANT_COLS et CUMULATIVE_COLS: moyenne arithmetique
      - CIRCULAR_COLS (wind_direction): moyenne circulaire
        atan2(mean(sin(angle)), mean(cos(angle)))

    On utilise mean() qui ignore les nulls par defaut en Polars,
    donc un site avec valeur manquante a un instant t est simplement
    ecarte de la moyenne pour cet instant.

    Le parametre 'prefix' est ajoute aux noms des colonnes finales
    (utile pour distinguer real vs pred dans le merge final).
    """
    if "site" not in df.columns:
        return df

    # Detection des colonnes presentes
    cols_present = [c for c in df.columns if c not in ("timestamp", "site")]
    instant_p = [c for c in INSTANT_COLS if c in cols_present]
    cumul_p = [c for c in CUMULATIVE_COLS if c in cols_present]
    circular_p = [c for c in CIRCULAR_COLS if c in cols_present]
    other_p = [c for c in cols_present
               if c not in instant_p and c not in cumul_p and c not in circular_p]

    # Pour les colonnes circulaires, ajouter sin/cos en amont
    deg2rad = math.pi / 180.0
    rad2deg = 180.0 / math.pi
    extra_sin_cos = []
    for c in circular_p:
        df = df.with_columns([
            (pl.col(c) * deg2rad).sin().alias(f"_{c}_sin"),
            (pl.col(c) * deg2rad).cos().alias(f"_{c}_cos"),
        ])
        extra_sin_cos.append(c)

    # Construction des aggregations
    aggs = []
    aggs.extend([pl.col(c).mean().alias(c) for c in instant_p])
    aggs.extend([pl.col(c).mean().alias(c) for c in cumul_p])
    for c in circular_p:
        aggs.append(pl.col(f"_{c}_sin").mean().alias(f"_{c}_sin_avg"))
        aggs.append(pl.col(f"_{c}_cos").mean().alias(f"_{c}_cos_avg"))
    # Autres colonnes (e.g. pred_*): moyenne par defaut
    aggs.extend([pl.col(c).mean().alias(c) for c in other_p])

    df_agg = df.group_by("timestamp").agg(aggs).sort("timestamp")

    # Recomposer wind_direction depuis sin/cos
    for c in circular_p:
        df_agg = df_agg.with_columns(
            (
                pl.arctan2(
                    pl.col(f"_{c}_sin_avg"),
                    pl.col(f"_{c}_cos_avg"),
                ) * rad2deg
            ).mod(360.0).alias(c)
        ).drop([f"_{c}_sin_avg", f"_{c}_cos_avg"])

    # Application du prefix
    if prefix:
        rename_map = {c: f"{prefix}{c}" for c in cols_present}
        df_agg = df_agg.rename(rename_map)

    return df_agg


# ============================================================
# 7. Fusion finale
# ============================================================

def merge_all(oiken: pl.DataFrame,
              meteo_real: pl.DataFrame,
              meteo_pred: pl.DataFrame) -> pl.DataFrame:
    """
    Joint OIKEN (reference) + meteo_real + meteo_pred sur timestamp UTC.

    Le DataFrame OIKEN sert de grille de reference: on garde toutes ses
    lignes, et on ajoute les colonnes meteo via left join. Les timestamps
    OIKEN sans correspondance meteo auront des valeurs null (gerees en
    aval ou par le modele).
    """
    merged = oiken.join(meteo_real, on="timestamp", how="left")
    merged = merged.join(meteo_pred, on="timestamp", how="left")
    return merged


# ============================================================
# 8. Rapport qualite
# ============================================================

def quality_report(df: pl.DataFrame, oiken_log: dict,
                   real_outliers: dict, pred_outliers: dict) -> dict:
    """
    Genere un dict synthetique avec les metriques de la sortie finale.
    """
    n = df.height
    ts_min = df["timestamp"].min()
    ts_max = df["timestamp"].max()

    # Detection grossiere de trous dans la grille (pas de 15min strict)
    diffs = df["timestamp"].diff().drop_nulls()
    irreg = diffs.filter(diffs != timedelta(minutes=15)).len()

    # Taux de nulls par colonne
    null_pct = {}
    for c in df.columns:
        if c == "timestamp":
            continue
        nc = df[c].null_count()
        if nc > 0:
            null_pct[c] = round(100 * nc / n, 3)

    return {
        "n_rows": n,
        "n_expected_oiken": OIKEN_EXPECTED_ROWS,
        "ts_min_utc": str(ts_min),
        "ts_max_utc": str(ts_max),
        "n_irregular_intervals": irreg,
        "null_pct_per_col": null_pct,
        "oiken_normalization": oiken_log,
        "real_physical_outliers": real_outliers,
        "pred_physical_outliers": pred_outliers,
    }


# ============================================================
# 9. Main
# ============================================================

def main() -> int:
    print("=" * 60)
    print("PIPELINE NORMALISATION OIKEN ML")
    print("=" * 60)

    # Chargement des Parquet bruts produits par acquisition_pipeline
    print("\n[1/7] Chargement des Parquet bruts...")
    try:
        oiken = pl.read_parquet(DATA_RAW / "oiken_raw.parquet")
        meteo_real = pl.read_parquet(DATA_RAW / "meteo_real_raw.parquet")
        meteo_pred = pl.read_parquet(DATA_RAW / "meteo_pred_raw.parquet")
    except FileNotFoundError as e:
        print(f"  ERREUR: Parquet manquant. As-tu lance acquisition_pipeline?")
        print(f"  Detail: {e}")
        return 1

    print(f"  oiken_raw       : {oiken.shape}")
    print(f"  meteo_real_raw  : {meteo_real.shape}")
    print(f"  meteo_pred_raw  : {meteo_pred.shape}")

    # ---- OIKEN ----
    print("\n[2/7] Normalisation OIKEN (UTC + outliers + DST shift)...")
    oiken_norm, oiken_log = normalize_oiken(oiken)
    print(f"  Strategie utilisee : {oiken_log['strategy']}")
    verif = oiken_log["verification"]
    print(f"  Desaccords local   : {verif['n_mismatches']} "
          f"(tolerance {verif['tolerance']})")
    if verif["first_examples"]:
        print(f"  Exemples desaccords (DST forward attendus):")
        for ex in verif["first_examples"][:3]:
            print(f"    UTC {ex['utc']}: csv={ex['csv_local']} "
                  f"recalc={ex['recalc_local']}")
    print(f"  Outliers load nullifies : {oiken_log['load_outliers_nullified']}")
    print(f"  Forecast nulls avant/apres interp: "
          f"{oiken_log['forecast_load_nulls_before_interp']} / "
          f"{oiken_log['forecast_load_nulls_after_interp']}")
    dst_log = oiken_log.get("dst_shift_fix", {})
    print(f"  [DST fix] forecast_load shifte de +{dst_log.get('shift_pas', '?')} "
          f"pas (1h) sur {dst_log.get('n_cet_lines', 0):,} lignes CET, "
          f"inchange sur {dst_log.get('n_cest_lines', 0):,} lignes CEST")
    print(f"  [DST fix] nulls avant/apres shift: "
          f"{dst_log.get('n_null_before_shift', 0)} / "
          f"{dst_log.get('n_null_after_shift', 0)} "
          f"(+{dst_log.get('n_null_added_by_shift', 0)} crees)")
    print(f"  -> {oiken_norm.shape[0]:,} lignes UTC")

    # ---- METEO REAL ----
    print("\n[3/7] Normalisation meteo reelle (UTC + outliers)...")
    real_utc = to_utc_meteo(meteo_real)
    real_clean, real_outliers = clip_physical_outliers(real_utc)
    n_outliers_total = sum(real_outliers.values())
    print(f"  Outliers physiques nullifies : {n_outliers_total} total")
    for c, n in real_outliers.items():
        if n > 0:
            print(f"    {c}: {n}")

    print("\n[4/7] Resampling meteo reelle 10min -> 15min...")
    real_15 = resample_real_to_15min(real_clean)
    print(f"  -> {real_15.shape[0]:,} lignes (par site)")

    print("\n[5/7] Resampling meteo previsions horaire -> 15min...")
    pred_utc = to_utc_meteo(meteo_pred)
    # Prevision: pas de bornes physiques pour les statistiques de
    # dispersion (q10, q90, stde). On clip seulement les valeurs ctrl
    # qui sont des grandeurs physiques.
    pred_clean, pred_outliers = clip_physical_outliers(pred_utc)
    pred_15 = resample_pred_to_15min(pred_clean)
    print(f"  -> {pred_15.shape[0]:,} lignes (par site)")

    print("\n[6/7] Agregation des sites...")
    real_agg = aggregate_meteo_sites(real_15, prefix="meteo_")
    pred_agg = aggregate_meteo_sites(pred_15, prefix="")  # deja prefixe pred_
    print(f"  Real agrege : {real_agg.shape}")
    print(f"  Pred agrege : {pred_agg.shape}")

    print("\n[7/7] Fusion finale...")
    final = merge_all(oiken_norm, real_agg, pred_agg)
    print(f"  Dataset final : {final.shape}")

    # ---- Rapport ----
    report = quality_report(final, oiken_log, real_outliers, pred_outliers)

    print("\n  --- Resume final ---")
    print(f"  Lignes        : {report['n_rows']:,} "
          f"(attendu OIKEN {report['n_expected_oiken']:,})")
    print(f"  Plage UTC     : {report['ts_min_utc']} -> {report['ts_max_utc']}")
    print(f"  Intervalles irreguliers : {report['n_irregular_intervals']}")
    print(f"  Taux de nulls par colonne :")
    for c, p in sorted(report["null_pct_per_col"].items(),
                       key=lambda x: -x[1])[:10]:
        print(f"    {c:30s}: {p:>6.2f}%")
    if len(report["null_pct_per_col"]) > 10:
        print(f"    ... et {len(report['null_pct_per_col']) - 10} autres")

    # ---- Sauvegarde ----
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out_parquet = DATA_PROCESSED / "dataset_normalized.parquet"
    final.write_parquet(out_parquet)
    print(f"\n  -> {out_parquet}")

    DATA_REPORTS.mkdir(parents=True, exist_ok=True)
    out_report = DATA_REPORTS / "normalization_report.json"
    with open(out_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"  -> {out_report}")

    print("\nNORMALISATION TERMINEE")
    return 0


if __name__ == "__main__":
    sys.exit(main())