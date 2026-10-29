# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "9d167aeb-04db-4690-939d-7e296550dbc9",
# META       "default_lakehouse_name": "POC_LH_VEHICLE_USAGE_BRONZE",
# META       "default_lakehouse_workspace_id": "f78fd47f-e385-4179-b31a-5e376322c392",
# META       "known_lakehouses": [
# META         {
# META           "id": "e7f40075-4dc5-4241-90b8-9b5a6af22ea4"
# META         },
# META         {
# META           "id": "9d167aeb-04db-4690-939d-7e296550dbc9"
# META         }
# META       ]
# META     }
# META   }
# META }

# MARKDOWN ********************

# # BRP2 - Bronze to Silver
# Read-only client CSVs -> reusable cleaned Delta observations. The ML notebook consumes a hash-verified export of the Delta readback.

# PARAMETERS CELL ********************

medallion_workspace_id = "f78fd47f-e385-4179-b31a-5e376322c392"
silver_lakehouse_id = "e7f40075-4dc5-4241-90b8-9b5a6af22ea4"
silver_run_id = "silver_final_20261016"
snapshot_date = "2026-08-28"


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

import hashlib
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd

def parse_flags(values: pd.Series) -> pd.Series:
    normalized = values.astype("string").str.strip().str.lower()
    mapping = {"true": True, "false": False, "1": True, "0": False}
    if not normalized.isin(mapping).all():
        raise ValueError(f"Unrecognized or missing boolean in {values.name}.")
    return normalized.map(mapping).astype(bool)

def prepare_features(source: Path, snapshot: str, sample_limit: int | None = 100000) -> tuple[pd.DataFrame, dict]:
    fleet_columns = [
        "SERIAL_NO", "PRODUCT_CODE", "MEASURE_DATE", "VEHICLE_USAGE",
        "USAGE_UNIT", "VALID_OBSERVATION_FLAG", "METAIS_DELETED",
        "REGISTRATION_DATE", "PLANT", "FLEET_REGION",
    ]
    fleet = pd.read_csv(
        source / "SANITIZED_RAW_FLEET_OBSERVATIONS.csv",
        usecols=fleet_columns, dtype="string",
    )
    reference = pd.read_csv(
        source / "SANITIZED_RAW_MODEL_FAMILY.csv",
        usecols=["MODEL_NO", "MODEL_YEAR", "PRODUCT_LINE_GROUP"], dtype="string",
    ).rename(columns={
        "MODEL_NO": "PRODUCT_CODE", "MODEL_YEAR": "model_year",
        "PRODUCT_LINE_GROUP": "product_line",
    })
    if reference["PRODUCT_CODE"].isna().any() or reference["PRODUCT_CODE"].duplicated().any():
        raise ValueError("Model reference key must be present and unique.")
    reference["model_year"] = pd.to_numeric(reference["model_year"], errors="coerce")
    if reference[["model_year", "product_line"]].isna().any().any():
        raise ValueError("Model reference attributes are incomplete.")
    raw_rows = len(fleet)
    if raw_rows > 1_000_000:
        raise ValueError("Input exceeds this small-team demo's one-million-row bound.")
    valid_flag = parse_flags(fleet["VALID_OBSERVATION_FLAG"])
    deleted = parse_flags(fleet["METAIS_DELETED"])
    for name in ["MEASURE_DATE", "REGISTRATION_DATE"]:
        fleet[name] = pd.to_datetime(fleet[name], errors="coerce", format="mixed")
    fleet["target_km"] = pd.to_numeric(fleet["VEHICLE_USAGE"], errors="coerce")
    valid = (
        valid_flag & ~deleted & fleet["USAGE_UNIT"].eq("KM")
        & fleet["SERIAL_NO"].notna() & fleet["PRODUCT_CODE"].notna()
        & fleet["target_km"].between(0, 300000)
        & (fleet["MEASURE_DATE"] >= fleet["REGISTRATION_DATE"])
        & (fleet["MEASURE_DATE"] <= pd.Timestamp(snapshot))
    ).fillna(False)
    clean = fleet.loc[valid].merge(
        reference, on="PRODUCT_CODE", how="left", validate="many_to_one", indicator=True,
    )
    unmatched = int(clean["_merge"].ne("both").sum())
    if unmatched:
        raise ValueError(f"{unmatched} valid observations have no model reference.")
    clean = clean.drop(columns="_merge")
    # One vehicle/date target, not repeated observation lines in the holdout.
    keys = ["SERIAL_NO", "MEASURE_DATE"]
    conflicting = clean.groupby(keys)["target_km"].transform("nunique").gt(1)
    conflicting_rows = int(conflicting.sum())
    conflicting_groups = len(clean.loc[conflicting, keys].drop_duplicates())
    clean = clean.loc[~conflicting].copy()
    duplicate_rows = int(clean.duplicated(keys).sum())
    clean = clean.drop_duplicates(keys).copy()
    clean["age_days"] = (clean["MEASURE_DATE"] - clean["REGISTRATION_DATE"]).dt.days
    clean["vehicle_fold"] = clean["SERIAL_NO"].map(
        lambda value: int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16) % 5
    )
    clean["sampling_key"] = (
        clean["SERIAL_NO"] + "|" + clean["MEASURE_DATE"].astype(str)
    ).map(lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest())
    eligible_rows = len(clean)
    ordered = clean.sort_values("sampling_key")
    data = (ordered if sample_limit is None else ordered.head(sample_limit)).copy()
    for name in ["product_line", "PLANT", "FLEET_REGION"]:
        data[name] = data[name].astype(object).where(data[name].notna(), np.nan)
        if data[name].isna().all():
            raise ValueError(f"Feature {name} is entirely missing.")
    profile = {
        "source_observation_rows": raw_rows,
        "excluded_invalid_rows": int((~valid).sum()),
        "duplicate_vehicle_date_rows_removed": duplicate_rows,
        "conflicting_vehicle_date_groups_quarantined": conflicting_groups,
        "conflicting_observation_rows_quarantined": conflicting_rows,
        "eligible_observation_rows": eligible_rows,
        "sampled_observation_rows": len(data),
        "sampled_vehicles": int(data["SERIAL_NO"].nunique()),
        "source_reference_rows": len(reference),
        "sample_measure_date_min": str(data["MEASURE_DATE"].min().date()),
        "sample_measure_date_max": str(data["MEASURE_DATE"].max().date()),
        "missing_plant_in_sample": int(data["PLANT"].isna().sum()),
        "product_lines": sorted(data["product_line"].dropna().unique().tolist()),
        "sampling": "SHA256(vehicle|measurement_date), first 100000 rows",
    }
    return data, profile

def split_holdout(data: pd.DataFrame, snapshot: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = pd.Timestamp(snapshot) - pd.Timedelta(days=365)
    train = data[(data["vehicle_fold"] != 0) & (data["MEASURE_DATE"] < cutoff)].copy()
    test = data[(data["vehicle_fold"] == 0) & (data["MEASURE_DATE"] >= cutoff)].copy()
    if len(train) < 100 or len(test) < 100:
        raise ValueError("Need at least 100 rows on both sides of the vehicle/time holdout.")
    if set(train["SERIAL_NO"]) & set(test["SERIAL_NO"]):
        raise AssertionError("Vehicle leakage between training and evaluation.")
    return train, test

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

"""Build reusable cleaned observations; publish only to the dedicated Silver lakehouse."""

import tempfile
import time

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import notebookutils


def prepare_silver() -> dict:
    started = time.monotonic()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", silver_run_id):
        raise ValueError("Invalid Silver publication identifier.")
    root = (f"abfss://{medallion_workspace_id}@onelake.dfs.fabric.microsoft.com/"
            f"{silver_lakehouse_id}")
    published = f"{root}/Files/POC_silver/{silver_run_id}"
    if notebookutils.fs.exists(published):
        raise FileExistsError("Silver execution already exists; choose another identifier.")
    source = Path("/lakehouse/default/Files/FAIL_RATES_PREDICTION_SANITIZED_DATASETS")
    full, _ = prepare_features(source, snapshot_date, sample_limit=None)
    sample, profile = prepare_features(source, snapshot_date)
    pd.testing.assert_frame_equal(full.head(100000), sample)
    train, test = split_holdout(sample, snapshot_date)
    full["sample_selected"] = full["sampling_key"].isin(sample["sampling_key"])
    full["evaluation_role"] = "not_selected"
    full.loc[full["sampling_key"].isin(train["sampling_key"]), "evaluation_role"] = "train"
    full.loc[full["sampling_key"].isin(test["sampling_key"]), "evaluation_role"] = "test"
    spark = SparkSession.builder.getOrCreate()
    qualified = "`POC_LH_VEHICLE_USAGE_SILVER`.`POC_silver`.`poc_observations`"
    spark.sql("CREATE SCHEMA IF NOT EXISTS `POC_LH_VEHICLE_USAGE_SILVER`.`POC_silver`")
    with tempfile.TemporaryDirectory() as temporary:
        local = Path(temporary) / "observations.parquet"
        full.to_parquet(local, index=False, coerce_timestamps="us")
        notebookutils.fs.mkdirs(published)
        notebookutils.fs.cp(local.as_uri(), published + "/candidate.parquet")
        frame = spark.read.parquet(published + "/candidate.parquet")
        frame.write.format("delta").mode("overwrite").saveAsTable(qualified)
        readback = spark.table(qualified)
        if readback.count() != len(full):
            raise AssertionError("Silver Delta readback differs.")
        if readback.groupBy("SERIAL_NO", "MEASURE_DATE").count().filter("count > 1").count():
            raise AssertionError("Silver vehicle/date key is not unique.")
        roles = {r["evaluation_role"]: r["count"] for r in readback.groupBy("evaluation_role").count().collect()}
        if roles.get("train") != len(train) or roles.get("test") != len(test):
            raise AssertionError("Silver holdout assignment differs.")
        leakage = (readback.filter("evaluation_role = 'train'").select("SERIAL_NO").distinct()
                   .join(readback.filter("evaluation_role = 'test'").select("SERIAL_NO").distinct(), "SERIAL_NO"))
        if leakage.count():
            raise AssertionError("Silver has vehicle leakage.")
        # Python consumes a portable export of the Delta readback, never the CSVs.
        restored = readback.toPandas().sort_values("sampling_key").reset_index(drop=True)
        pd.testing.assert_frame_equal(full.reset_index(drop=True), restored, check_dtype=False)
        restored.to_parquet(local, index=False, coerce_timestamps="us")
        digest = hashlib.sha256(local.read_bytes()).hexdigest()
        notebookutils.fs.cp(local.as_uri(), published + "/observations.parquet")
    result = {
        "status": "published", "silver_run_id": silver_run_id, "snapshot_date": snapshot_date,
        "silver_lakehouse_id": silver_lakehouse_id, "eligible_rows": len(full),
        "sample_profile": profile, "roles": roles, "export_sha256": digest,
        "delta_table": qualified, "client_sources_modified": False,
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }
    notebookutils.fs.put(published + "/published.json", json.dumps(result, indent=2), True)
    print(json.dumps(result, indent=2))
    return result


silver_publication = prepare_silver()


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
