"""
Pipeline d'acquisition des données.

Ce module télécharge les données brutes depuis 2 sources :
1. Le fichier CSV fourni par OIKEN (courbe de charge + production PV)
2. La base InfluxDB de la HES-SO (données météo MeteoSuisse)

Une fois téléchargées, les données sont sauvegardées en Parquet
(format compressé et rapide à lire) pour ne pas re-télécharger à chaque fois.

Utilisation :
    python -m src.acquisition
"""

import warnings

import polars as pl
from influxdb_client import InfluxDBClient
from influxdb_client.client.warnings import MissingPivotFunction

# On importe tous les paramètres depuis config.py
from src.config import (
    INFLUXDB_URL, INFLUXDB_TOKEN, INFLUXDB_ORG, INFLUXDB_BUCKET,
    SITES, MEASUREMENTS_REAL, MEASUREMENTS_PRED, MEASUREMENT_SHORT_NAMES,
    DATA_RAW, OIKEN_START, OIKEN_END,
)

# Supprime un warning inutile de la librairie InfluxDB (lié à pivot())
warnings.simplefilter("ignore", MissingPivotFunction)


# ============================================================
# 1. Téléchargement des données météo depuis InfluxDB
# ============================================================
# Cette fonction unique gère les 2 cas :
#   - Mesures réelles (9 variables, pas de filtre "Prediction")
#   - Prévisions PRED (32 variables, filtre "Prediction" = "00")


def fetch_meteo(measurements: list[str],
                start: str = OIKEN_START,
                stop: str = OIKEN_END,
                prediction: str | None = None) -> pl.DataFrame:
    """Télécharge des données météo depuis InfluxDB.

    Pour chaque combinaison (site × measurement), on envoie une requête
    Flux au serveur, on récupère un petit DataFrame, et à la fin on
    fusionne tout en un seul grand DataFrame.

    Args:
        measurements: liste des noms de measurements à télécharger.
        start: date de début (ex: '2022-10-01').
        stop: date de fin (ex: '2025-09-30').
        prediction: si None → données réelles. Si '00' → prévisions contrôle.
    """
    # Connexion au serveur InfluxDB
    client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN,
                            org=INFLUXDB_ORG)
    query_api = client.query_api()

    all_frames = []                                # Liste pour stocker chaque petit DataFrame
    total = len(SITES) * len(measurements)         # Nombre total de requêtes
    count = 0

    for site in SITES:
        for measurement in measurements:
            count += 1

            # Nom court pour la colonne du DataFrame
            # Mesures réelles : on utilise le dictionnaire de noms courts (ex: "temperature_2m")
            # Prévisions PRED : on met en minuscule (ex: "pred_t_2m_ctrl")
            if measurement in MEASUREMENT_SHORT_NAMES:
                col_name = MEASUREMENT_SHORT_NAMES[measurement]
            else:
                col_name = measurement.lower()

            print(f"  [{count}/{total}] {site} / {col_name}...", end=" ")

            # --- Construction de la requête Flux ---
            # Flux est le langage de requête d'InfluxDB.
            # Les |> sont des "pipes" : chaque ligne filtre le résultat de la précédente.
            query = f'''
            from(bucket: "{INFLUXDB_BUCKET}")
              |> range(start: {start}T00:00:00Z, stop: {stop}T23:59:59Z)
              |> filter(fn: (r) => r._measurement == "{measurement}")
              |> filter(fn: (r) => r.Site == "{site}")
            '''
            # Pour les prévisions, on ajoute un filtre sur le membre d'ensemble
            # "00" = run de contrôle (prévision principale)
            if prediction is not None:
                query += f'  |> filter(fn: (r) => r.Prediction == "{prediction}")\n'

            # Pivot : réorganise les données en format tabulaire (une colonne par champ)
            query += '  |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")'

            # --- Exécution de la requête ---
            try:
                pdf = query_api.query_data_frame(query)  # Résultat en Pandas
            except Exception as e:
                print(f"ERREUR: {e}")
                continue  # On passe à la suivante si erreur

            if pdf.empty:
                print("(vide)")
                continue

            # Conversion Pandas → Polars, on ne garde que timestamp + valeur
            df = pl.from_pandas(pdf[["_time", "Value"]].rename(
                columns={"_time": "timestamp", "Value": col_name}
            ))
            # Ajout d'une colonne "site" pour savoir d'où vient chaque ligne
            df = df.with_columns(pl.lit(site).alias("site"))
            all_frames.append(df)
            print(f"OK ({len(df)} lignes)")

    client.close()

    if not all_frames:
        raise RuntimeError("Aucune donnée récupérée depuis InfluxDB.")

    # Fusion de tous les petits DataFrames en un seul
    result = _merge_frames(all_frames)
    print(f"\n→ Résultat : {result.shape[0]} lignes, {result.shape[1]} colonnes")
    return result


def _merge_frames(frames: list[pl.DataFrame]) -> pl.DataFrame:
    """Fusionne les DataFrames individuels en un seul.

    Chaque DataFrame a 3 colonnes : [timestamp, site, <une_mesure>].
    On les joint un par un sur (timestamp, site) pour obtenir :
    [timestamp, site, temperature_2m, pressure, humidity, ...]
    """
    merged = frames[0]
    for frame in frames[1:]:
        # Trouver le nom de la colonne de valeur (celle qui n'est pas timestamp/site)
        value_col = [c for c in frame.columns if c not in ("timestamp", "site")][0]

        # Éviter les doublons (si la même mesure apparaît 2 fois)
        if value_col in merged.columns:
            continue

        # Left join : on garde toutes les lignes de merged,
        # et on ajoute les valeurs quand (timestamp, site) correspondent
        merged = merged.join(frame, on=["timestamp", "site"], how="left")

    return merged.sort(["site", "timestamp"])


# ============================================================
# 2. Chargement du dataset OIKEN (fichier CSV)
# ============================================================

def load_oiken_csv(path: str) -> pl.DataFrame:
    """Charge et nettoie le fichier CSV fourni par OIKEN.

    Le CSV contient la courbe de charge normalisée et la production PV
    par zone. Plusieurs problèmes à gérer :
    - Le séparateur peut être ',' ou ';' selon la version du fichier
    - Les nombres utilisent parfois la virgule décimale (1,5 au lieu de 1.5)
    - Certaines valeurs sont '#N/A' (données manquantes)
    - Le format de date varie (points ou slashs comme séparateurs)
    """

    # --- Détection du séparateur ---
    # On lit la première ligne pour voir si c'est ',' ou ';'
    with open(path, "r") as f:
        first_line = f.readline()
    separator = ";" if ";" in first_line else ","

    # --- Lecture brute ---
    # infer_schema_length=0 : on lit TOUT en String (texte)
    # Pourquoi ? Parce que si Polars essaie de deviner le type (nombre),
    # il plantera sur les "#N/A" qui ne sont pas des nombres.
    df = pl.read_csv(path, infer_schema_length=0, separator=separator)

    # --- Renommage des colonnes ---
    # Les noms originaux sont très longs, on les raccourcit
    rename_map = {
        "timestamp": "timestamp",
        "standardised load [-]": "load",
        "standardised forecast load [-]": "forecast_load",
        "central valais solar production [kWh]": "pv_central_valais",
        "sion area solar production [kWh]": "pv_sion",
        "sierre area production [kWh]": "pv_sierre",
        "remote solar production [kWh]": "pv_remote",
    }
    df = df.rename(rename_map)

    # --- Nettoyage des colonnes numériques ---
    numeric_cols = ["load", "forecast_load", "pv_central_valais",
                    "pv_sion", "pv_sierre", "pv_remote"]

    for col in numeric_cols:
        df = df.with_columns(
            pl.when(pl.col(col) == "#N/A")    # Si la valeur est "#N/A"...
            .then(None)                        # ...on met null (donnée manquante)
            .otherwise(                        # Sinon...
                pl.col(col).str.replace(",", ".")  # ...remplacer virgule par point
            )
            .cast(pl.Float64)                  # Convertir le texte en nombre décimal
            .alias(col)
        )

    # --- Parsing du timestamp ---
    # Certains fichiers utilisent des points (01.01.2023) au lieu de slashs (01/01/2023)
    # On normalise tout en slashs avant de parser
    df = df.with_columns(
        pl.col("timestamp").str.replace_all(r"\.", "/").alias("timestamp")
    )
    df = df.with_columns(
        pl.col("timestamp").str.strptime(
            pl.Datetime, format="%d/%m/%Y %H:%M"  # Format : jour/mois/année heure:minute
        )
    )

    # --- Calcul de la production PV totale ---
    # Somme des 4 zones de production solaire
    df = df.with_columns(
        (pl.col("pv_central_valais") + pl.col("pv_sion")
         + pl.col("pv_sierre") + pl.col("pv_remote")).alias("pv_total")
    )

    # Tri par ordre chronologique
    df = df.sort("timestamp")

    # Affichage d'un résumé pour vérifier que tout va bien
    print(f"→ Dataset OIKEN : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    print(f"  Période : {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"  Valeurs nulles (forecast_load) : "
          f"{df['forecast_load'].null_count()}")

    return df


# ============================================================
# 3. Sauvegarde en Parquet
# ============================================================

def save_parquet(df: pl.DataFrame, name: str, directory=DATA_RAW) -> None:
    """Sauvegarde un DataFrame en format Parquet.

    Parquet est un format binaire compressé, beaucoup plus rapide
    à lire qu'un CSV et 10x plus petit. C'est le standard pour
    les pipelines de données en Python.
    """
    directory.mkdir(parents=True, exist_ok=True)  # Crée le dossier s'il n'existe pas
    path = directory / f"{name}.parquet"
    df.write_parquet(path)
    print(f"  Sauvegardé : {path} ({path.stat().st_size / 1024:.0f} KB)")


# ============================================================
# Point d'entrée (ce qui s'exécute quand on lance le script)
# ============================================================
# python -m src.acquisition → exécute tout ce bloc

if __name__ == "__main__":
    print("=" * 60)
    print("PIPELINE D'ACQUISITION — Energy Informatics 2")
    print("=" * 60)

    # Étape 1 : charger le CSV OIKEN et le convertir en Parquet
    print("\n[1/3] Chargement du dataset OIKEN...")
    oiken = load_oiken_csv(str(DATA_RAW / "oiken-data.csv"))
    save_parquet(oiken, "oiken_clean")

    # Étape 2 : télécharger les 9 mesures réelles (4 sites × 9 variables = 36 requêtes)
    print("\n[2/3] Extraction des mesures réelles MeteoSuisse...")
    print(f"  Sites : {SITES}")
    print(f"  Période : {OIKEN_START} → {OIKEN_END}")
    meteo_real = fetch_meteo(MEASUREMENTS_REAL)
    save_parquet(meteo_real, "meteo_real")

    # Étape 3 : télécharger les 32 prévisions (4 sites × 32 variables = 128 requêtes)
    # prediction="00" = run de contrôle (prévision principale de COSMO-E)
    print("\n[3/3] Extraction des prévisions MeteoSuisse...")
    meteo_pred = fetch_meteo(MEASUREMENTS_PRED, prediction="00")
    save_parquet(meteo_pred, "meteo_pred")

    print("\n" + "=" * 60)
    print("ACQUISITION TERMINÉE")
    print("=" * 60)