import warnings
import polars as pl
from influxdb_client import InfluxDBClient
from influxdb_client.client.warnings import MissingPivotFunction
from src.config import (
    INFLUXDB_URL, INFLUXDB_TOKEN, INFLUXDB_ORG, INFLUXDB_BUCKET,
    SITES, MEASUREMENTS_REAL, MEASUREMENTS_PRED, MEASUREMENT_SHORT_NAMES,
    DATA_RAW, OIKEN_START, OIKEN_END,
)

warnings.simplefilter("ignore", MissingPivotFunction)


def fetch_meteo(measurements, start=OIKEN_START, stop=OIKEN_END, prediction=None):
    client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
    query_api = client.query_api()
    all_frames = []

    for site in SITES:
        for measurement in measurements:
            col_name = MEASUREMENT_SHORT_NAMES.get(measurement, measurement.lower())

            query = f'''
            from(bucket: "{INFLUXDB_BUCKET}")
              |> range(start: {start}T00:00:00Z, stop: {stop}T23:59:59Z)
              |> filter(fn: (r) => r._measurement == "{measurement}")
              |> filter(fn: (r) => r.Site == "{site}")
            '''
            if prediction is not None:
                query += f'  |> filter(fn: (r) => r.Prediction == "{prediction}")\n'
            query += '  |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")'

            try:
                pdf = query_api.query_data_frame(query)
            except Exception:
                continue

            if pdf.empty:
                continue

            df = pl.from_pandas(pdf[["_time", "Value"]].rename(
                columns={"_time": "timestamp", "Value": col_name}
            ))
            df = df.with_columns(pl.lit(site).alias("site"))
            all_frames.append(df)

    client.close()

    if not all_frames:
        raise RuntimeError("Aucune donnée récupérée.")

    merged = all_frames[0]
    for frame in all_frames[1:]:
        value_col = [c for c in frame.columns if c not in ("timestamp", "site")][0]
        if value_col not in merged.columns:
            merged = merged.join(frame, on=["timestamp", "site"], how="left")

    return merged.sort(["site", "timestamp"])


def load_oiken_csv(path):
    with open(path, "r") as f:
        separator = ";" if ";" in f.readline() else ","

    df = pl.read_csv(path, infer_schema_length=0, separator=separator)

    df = df.rename({
        "timestamp": "timestamp",
        "standardised load [-]": "load",
        "standardised forecast load [-]": "forecast_load",
        "central valais solar production [kWh]": "pv_central_valais",
        "sion area solar production [kWh]": "pv_sion",
        "sierre area production [kWh]": "pv_sierre",
        "remote solar production [kWh]": "pv_remote",
    })

    for col in ["load", "forecast_load", "pv_central_valais", "pv_sion", "pv_sierre", "pv_remote"]:
        df = df.with_columns(
            pl.when(pl.col(col) == "#N/A").then(None)
            .otherwise(pl.col(col).str.replace(",", "."))
            .cast(pl.Float64).alias(col)
        )

    df = df.with_columns(pl.col("timestamp").str.replace_all(r"\.", "/"))
    df = df.with_columns(pl.col("timestamp").str.strptime(pl.Datetime, format="%d/%m/%Y %H:%M"))
    df = df.with_columns(
        (pl.col("pv_central_valais") + pl.col("pv_sion") + pl.col("pv_sierre") + pl.col("pv_remote")).alias("pv_total")
    )
    return df.sort("timestamp")


def save_parquet(df, name, directory=DATA_RAW):
    directory.mkdir(parents=True, exist_ok=True)
    df.write_parquet(directory / f"{name}.parquet")


if __name__ == "__main__":
    oiken = load_oiken_csv(str(DATA_RAW / "oiken-data.csv"))
    save_parquet(oiken, "oiken_clean")

    meteo_real = fetch_meteo(MEASUREMENTS_REAL)
    save_parquet(meteo_real, "meteo_real")

    meteo_pred = fetch_meteo(MEASUREMENTS_PRED, prediction="00")
    save_parquet(meteo_pred, "meteo_pred")

    print("Tout bon")
