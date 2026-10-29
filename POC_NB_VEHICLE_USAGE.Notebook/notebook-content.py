# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "jupyter",
# META     "jupyter_kernel_name": "python3.12"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "fbb63bb5-9be9-413e-98f8-5483d1552137",
# META       "default_lakehouse_name": "POC_LH_VEHICLE_USAGE_GOLD",
# META       "default_lakehouse_workspace_id": "f78fd47f-e385-4179-b31a-5e376322c392",
# META       "known_lakehouses": [
# META         {
# META           "id": "fbb63bb5-9be9-413e-98f8-5483d1552137"
# META         },
# META         {
# META           "id": "e7f40075-4dc5-4241-90b8-9b5a6af22ea4"
# META         }
# META       ]
# META     }
# META   }
# META }

# CELL ********************

# MAGIC %%configure
# MAGIC {"numCores": 2}


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# # BRP2 - Vehicle mileage estimation
# One notebook, one experiment, one versioned ML model. Client CSVs stay read-only.
# Estimate cumulative mileage at a known age, not future failure probability.


# PARAMETERS CELL ********************

demo_run_id = ""
snapshot_date = "2026-08-28"
source_directory = "FAIL_RATES_PREDICTION_SANITIZED_DATASETS"
report_source_run_id = "medallion_final_20261016"
medallion_workspace_id = "f78fd47f-e385-4179-b31a-5e376322c392"
silver_lakehouse_id = "e7f40075-4dc5-4241-90b8-9b5a6af22ea4"
gold_lakehouse_id = "fbb63bb5-9be9-413e-98f8-5483d1552137"
silver_run_id = "silver_final_20261016"
mode = "train"


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ## 1. Consume reusable Silver and evaluate
# Read the hash-verified Parquet export of the Silver Delta readback. Never reconstruct features from Bronze CSVs in this notebook. Split by vehicle and time. The code logs five aggregate metrics to the experiment, never a metric per vehicle.


# CELL ********************

"""One focused Fabric demo: train, track, register and reload a mileage model."""

import hashlib
import importlib.metadata
import json
import re
import tempfile
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


EXPERIMENT = "POC_EXP_VEHICLE_USAGE"
MODEL = "POC_ML_VEHICLE_USAGE"
FEATURES = ["age_days", "model_year", "product_line", "PLANT", "FLEET_REGION"]


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


def build_model() -> Pipeline:
    return Pipeline([
        ("features", ColumnTransformer([
            ("numeric", SimpleImputer(strategy="median"), ["age_days", "model_year"]),
            ("categorical", Pipeline([
                ("missing", SimpleImputer(strategy="most_frequent")),
                ("encoding", OneHotEncoder(handle_unknown="ignore")),
            ]), ["product_line", "PLANT", "FLEET_REGION"]),
        ])),
        ("regressor", RandomForestRegressor(
            n_estimators=80, max_depth=10, min_samples_leaf=30, n_jobs=2, random_state=42,
        )),
    ])


def run_demo(demo_run_id: str, snapshot: str, source_directory: str, mode: str) -> dict:
    started = time.monotonic()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", demo_run_id):
        raise ValueError("Use a unique alphanumeric demo_run_id.")
    if mode not in {"profile", "train"}:
        raise ValueError("mode must be profile or train.")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", snapshot):
        raise ValueError("snapshot must be YYYY-MM-DD.")
    pd.Timestamp(snapshot)
    base = Path("/lakehouse/default/Files")
    if source_directory != "FAIL_RATES_PREDICTION_SANITIZED_DATASETS":
        raise ValueError("Only the inventoried client source directory is allowed.")
    output = base / "POC_vehicle_usage" / demo_run_id
    if output.exists():
        raise FileExistsError("Run output already exists; use a new demo_run_id.")
    data, profile = load_silver_features(snapshot)
    train, test = split_holdout(data, snapshot)
    print(json.dumps({"profile": profile, "train_rows": len(train), "test_rows": len(test)}, indent=2))
    if mode == "profile":
        output.mkdir(parents=True)
        result = {"status": "profiled", "demo_run_id": demo_run_id, **profile}
        (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    model = build_model()
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=f"POC_usage_{demo_run_id}") as run:
        mlflow.log_params({
            "demo_run_id": demo_run_id, "snapshot_date": snapshot,
            "temporal_cutoff": str((pd.Timestamp(snapshot) - pd.Timedelta(days=365)).date()),
            "holdout_vehicle_fold": 0, "n_estimators": 80, "max_depth": 10,
            "min_samples_leaf": 30, "sampling_row_limit": 100000,
            "target": "observed cumulative mileage at measurement date",
            "silver_run_id": silver_run_id, "silver_lakehouse_id": silver_lakehouse_id,
        })
        model.fit(train[FEATURES], train["target_km"])
        estimates = model.predict(test[FEATURES])
        baseline = np.full(len(test), train["target_km"].median())
        metrics = {
            "mae_km": float(mean_absolute_error(test["target_km"], estimates)),
            "baseline_mae_km": float(mean_absolute_error(test["target_km"], baseline)),
            "rmse_km": float(np.sqrt(mean_squared_error(test["target_km"], estimates))),
            "train_rows": float(len(train)), "test_rows": float(len(test)),
        }
        mlflow.log_metrics(metrics)
        if not np.isfinite(list(metrics.values())).all():
            raise ValueError("Nonfinite evaluation metrics.")
        if metrics["mae_km"] >= metrics["baseline_mae_km"]:
            raise ValueError("Model did not beat the median baseline; do not register as demo evidence.")
        with tempfile.TemporaryDirectory() as directory:
            profile_path = Path(directory) / "data_quality.json"
            profile_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
            mlflow.log_artifact(str(profile_path))
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(test["target_km"], estimates, s=8, alpha=0.3)
        top = max(float(test["target_km"].max()), float(estimates.max()))
        ax.plot([0, top], [0, top], "--", color="gray")
        ax.set(xlabel="Observed mileage (km)", ylabel="Predicted mileage (km)",
               title="Vehicle-separated, later-period holdout")
        fig.tight_layout()
        mlflow.log_figure(fig, "observed_vs_predicted.png")
        plt.show()
        plt.close(fig)
        sample = train[FEATURES].head(5)
        info = mlflow.sklearn.log_model(
            model, artifact_path="model",
            signature=infer_signature(sample, model.predict(sample)),
            input_example=sample,
        )
        version = mlflow.register_model(info.model_uri, MODEL)
        registered_uri = f"models:/{MODEL}/{version.version}"
        loaded_model = mlflow.sklearn.load_model(registered_uri)
        reloaded_predictions = loaded_model.predict(test[FEATURES])
        np.testing.assert_allclose(estimates, reloaded_predictions, rtol=1e-12, atol=1e-9)
        client = MlflowClient()
        metadata = client.get_model_version(MODEL, str(version.version))
        if metadata.run_id != run.info.run_id:
            raise AssertionError("Registered model lineage does not match this training run.")
        latest = data.sort_values("MEASURE_DATE").drop_duplicates("SERIAL_NO", keep="last")
        predictions = latest[["SERIAL_NO", "MEASURE_DATE"] + FEATURES].copy()
        predictions["predicted_km"] = loaded_model.predict(latest[FEATURES])
        if not np.isfinite(predictions["predicted_km"]).all():
            raise ValueError("Registered model produced invalid predictions.")
        predictions["demo_run_id"] = demo_run_id
        predictions["model_version"] = str(version.version)
        predictions["mlflow_run_id"] = run.info.run_id
        output.mkdir(parents=True)
        predictions.to_parquet(
            output / "predictions.parquet", index=False,
            coerce_timestamps="us", allow_truncated_timestamps=False,
        )
        saved_predictions = pd.read_parquet(output / "predictions.parquet")
        if len(saved_predictions) != len(predictions):
            raise AssertionError("Persisted prediction row count differs.")
        np.testing.assert_allclose(
            saved_predictions["predicted_km"], predictions["predicted_km"], rtol=1e-12, atol=1e-9,
        )
        result = {
            "status": "passed", "demo_run_id": demo_run_id, "snapshot_date": snapshot,
            "experiment_name": EXPERIMENT, "experiment_id": run.info.experiment_id,
            "mlflow_run_id": run.info.run_id, "model_name": MODEL,
            "model_version": str(version.version), "registered_model_uri": registered_uri,
            "registered_model_lineage_verified": True,
            "registered_model_reload_predictions_match": True,
            "persisted_prediction_readback_verified": True,
            "metrics": metrics,
            "mae_improvement_fraction": 1 - metrics["mae_km"] / metrics["baseline_mae_km"],
            "prediction_rows": len(predictions), "profile": profile,
            "silver_run_id": silver_run_id, "silver_lakehouse_id": silver_lakehouse_id,
            "gold_lakehouse_id": gold_lakehouse_id,
            "elapsed_seconds": round(time.monotonic() - started, 2),
            "versions": {
                package: importlib.metadata.version(package)
                for package in ["scikit-learn", "pandas", "numpy", "pyarrow"]
            },
            "limitations": [
                "Exploratory mileage estimation, not a failure-rate model or future trajectory.",
                "Latest predictions include training vehicles; evaluation uses only the held-out vehicles.",
                "Fixed client snapshot; no upstream refresh or source modifications.",
            ],
        }
        result["versions"]["mlflow"] = mlflow.__version__
    if client.get_run(run.info.run_id).info.status != "FINISHED":
        raise AssertionError("Training run did not finish.")
    (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

"""Read the validated Silver Delta export inside Fabric's Python runtime."""

from io import BytesIO

import notebookutils
import requests


def load_silver_features(snapshot: str) -> tuple[pd.DataFrame, dict]:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", silver_run_id):
        raise ValueError("Invalid Silver publication identifier.")
    root = (f"https://onelake.dfs.fabric.microsoft.com/{medallion_workspace_id}/"
            f"{silver_lakehouse_id}/Files/POC_silver/{silver_run_id}")
    token = notebookutils.credentials.getToken("storage")
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(root + "/published.json", headers=headers, timeout=120, allow_redirects=False)
    response.raise_for_status()
    evidence = response.json()
    if evidence["status"] != "published" or evidence["snapshot_date"] != snapshot:
        raise ValueError("Silver snapshot is not published or does not match.")
    response = requests.get(root + "/observations.parquet", headers=headers, timeout=120, allow_redirects=False)
    response.raise_for_status()
    if hashlib.sha256(response.content).hexdigest() != evidence["export_sha256"]:
        raise AssertionError("Silver export differs from its validated Delta readback.")
    full = pd.read_parquet(BytesIO(response.content))
    if len(full) != evidence["eligible_rows"]:
        raise AssertionError("Silver export row count mismatch.")
    data = full.sort_values("sampling_key").head(100000).copy()
    for name in ["product_line", "PLANT", "FLEET_REGION"]:
        data[name] = data[name].astype(object).where(data[name].notna(), np.nan)
    if len(data) != evidence["sample_profile"]["sampled_observation_rows"]:
        raise AssertionError("Silver deterministic sample mismatch.")
    return data, evidence["sample_profile"]


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

"""Prepare reporting candidates from an existing validated model, without refitting."""


def prepare_reporting(report_run_id: str, source_run_id: str) -> dict:
    for value in [report_run_id, source_run_id]:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", value):
            raise ValueError("Invalid reporting/source run identifier.")
    base = Path("/lakehouse/default/Files")
    source = base / "POC_vehicle_usage" / source_run_id
    summary = json.loads((source / "summary.json").read_text("utf-8"))
    if summary["status"] != "passed" or summary["demo_run_id"] != source_run_id:
        raise ValueError("Only a validated source execution can be published.")
    output = base / "POC_vehicle_usage" / report_run_id
    if output.exists():
        raise FileExistsError("Reporting candidate exists; use another run identifier.")
    if (summary["silver_run_id"] != silver_run_id
            or summary["silver_lakehouse_id"] != silver_lakehouse_id
            or summary["gold_lakehouse_id"] != gold_lakehouse_id):
        raise ValueError("Training and reporting must consume the same Silver publication and Gold target.")
    data, profile = load_silver_features(summary["snapshot_date"])
    if profile != summary["profile"]:
        raise ValueError("Source profile changed; do not mix current sources with an older model result.")
    train, test = split_holdout(data, summary["snapshot_date"])
    model = mlflow.sklearn.load_model(summary["registered_model_uri"])
    metadata = MlflowClient().get_model_version(MODEL, summary["model_version"])
    if metadata.run_id != summary["mlflow_run_id"]:
        raise AssertionError("Model lineage changed.")
    predicted_test = model.predict(test[FEATURES])
    baseline = float(train["target_km"].median())
    if not np.isclose(
        mean_absolute_error(test["target_km"], predicted_test),
        summary["metrics"]["mae_km"], rtol=1e-10,
    ):
        raise AssertionError("Recomputed holdout differs from the validated model result.")
    predictions = pd.read_parquet(source / "predictions.parquet")
    if len(predictions) != summary["prediction_rows"]:
        raise AssertionError("Source prediction count changed.")
    np.testing.assert_allclose(
        model.predict(predictions[FEATURES]), predictions["predicted_km"], rtol=1e-12, atol=1e-9,
    )
    evaluation = test[["MEASURE_DATE", "product_line", "model_year", "FLEET_REGION"]].copy()
    evaluation["observed_km"] = test["target_km"]
    evaluation["predicted_km"] = predicted_test
    evaluation["absolute_error"] = np.abs(test["target_km"] - predicted_test)
    evaluation["squared_error"] = (test["target_km"] - predicted_test) ** 2
    evaluation["baseline_absolute_error"] = np.abs(test["target_km"] - baseline)
    evaluation = evaluation.reset_index(drop=True)
    dimensions = ["product_line", "model_year", "FLEET_REGION"]
    segment = pd.concat([predictions[dimensions], evaluation[dimensions]]).drop_duplicates()
    segment["FLEET_REGION"] = segment["FLEET_REGION"].fillna("Unspecified")
    segment = segment.drop_duplicates().sort_values(dimensions).reset_index(drop=True)
    segment["segment_key"] = np.arange(1, len(segment) + 1, dtype=np.int64)
    tables = {}
    for name, frame in [("predictions", predictions), ("evaluation", evaluation)]:
        frame = frame.copy()
        frame["FLEET_REGION"] = frame["FLEET_REGION"].fillna("Unspecified")
        frame = frame.merge(segment, on=dimensions, how="left", validate="many_to_one")
        if frame["segment_key"].isna().any():
            raise AssertionError("Segment relationship has orphan rows.")
        frame["measurement_date"] = pd.to_datetime(frame["MEASURE_DATE"]).dt.normalize()
        frame["publication_key"] = 1
        if name == "predictions":
            frame["mileage_band_order"] = pd.cut(
                frame["predicted_km"], [-1, 1000, 5000, 10000, 20000, 50000, np.inf],
                labels=False,
            ).astype("int64")
            labels = ["0–1 000 km", "1 000–5 000 km", "5 000–10 000 km",
                      "10 000–20 000 km", "20 000–50 000 km", "50 000 km and above"]
            frame["mileage_band"] = frame["mileage_band_order"].map(dict(enumerate(labels)))
            columns = ["segment_key", "measurement_date", "publication_key",
                       "predicted_km", "mileage_band_order", "mileage_band"]
        else:
            frame["error_band_order"] = pd.cut(
                frame["absolute_error"], [-1, 1000, 3000, 5000, 10000, 20000, np.inf],
                labels=False,
            ).astype("int64")
            labels = ["0–1 000 km", "1 000–3 000 km", "3 000–5 000 km",
                      "5 000–10 000 km", "10 000–20 000 km", "20 000 km and above"]
            frame["error_band"] = frame["error_band_order"].map(dict(enumerate(labels)))
            columns = ["segment_key", "measurement_date", "publication_key", "observed_km",
                       "predicted_km", "absolute_error", "squared_error",
                       "baseline_absolute_error", "error_band_order", "error_band"]
        tables["POC_" + name] = frame[columns]
    tables["POC_segments"] = segment.rename(columns={"FLEET_REGION": "region"})
    dates = pd.date_range(
        min(frame["measurement_date"].min() for name, frame in tables.items()
            if name in {"POC_predictions", "POC_evaluation"}),
        max(frame["measurement_date"].max() for name, frame in tables.items()
            if name in {"POC_predictions", "POC_evaluation"}), freq="D",
    )
    tables["POC_calendar"] = pd.DataFrame({"date": dates, "year": dates.year, "month": dates.month})
    tables["POC_publication"] = pd.DataFrame([{
        "publication_key": 1, "source_run_id": source_run_id,
        "report_run_id": report_run_id, "model_version": summary["model_version"],
        "mlflow_run_id": summary["mlflow_run_id"], "snapshot_date": summary["snapshot_date"],
    }])
    quality = [
        ("Source observations", profile["source_observation_rows"]),
        ("Invalid rows excluded", profile["excluded_invalid_rows"]),
        ("Conflicting groups excluded", profile["conflicting_vehicle_date_groups_quarantined"]),
        ("Conflicting rows excluded", profile["conflicting_observation_rows_quarantined"]),
        ("Duplicate rows removed", profile["duplicate_vehicle_date_rows_removed"]),
        ("Eligible observations", profile["eligible_observation_rows"]),
        ("Deterministic sample", profile["sampled_observation_rows"]),
        ("Training rows", len(train)), ("Test rows", len(test)),
    ]
    tables["POC_quality"] = pd.DataFrame({
        "check_order": range(1, len(quality) + 1),
        "check_name": [x[0] for x in quality],
        "row_count": [x[1] for x in quality],
        "scope": "Global population", "publication_key": 1,
    })
    output.mkdir(parents=True)
    for name, frame in tables.items():
        frame.to_parquet(
            output / f"{name}.parquet", index=False,
            coerce_timestamps="us", allow_truncated_timestamps=False,
        )
        if len(pd.read_parquet(output / f"{name}.parquet")) != len(frame):
            raise AssertionError(f"Reporting readback mismatch: {name}")
    expected_by_segment = []
    for _, row in segment.iterrows():
        key = row["segment_key"]
        selected = tables["POC_evaluation"].query("segment_key == @key")
        expected_by_segment.append({
            "product_line": row["product_line"], "model_year": int(row["model_year"]),
            "region": row["FLEET_REGION"], "test_rows": len(selected),
            "prediction_rows": int(tables["POC_predictions"]["segment_key"].eq(key).sum()),
            "mae": float(selected["absolute_error"].mean()) if len(selected) else None,
            "baseline_mae": float(selected["baseline_absolute_error"].mean()) if len(selected) else None,
        })
    result = {
        "status": "reporting_prepared", "demo_run_id": report_run_id,
        "source_run_id": source_run_id, "model_version": summary["model_version"],
        "silver_run_id": silver_run_id, "silver_lakehouse_id": silver_lakehouse_id,
        "gold_lakehouse_id": gold_lakehouse_id,
        "table_rows": {name: len(frame) for name, frame in tables.items()},
        "expected_global": summary["metrics"],
        "expected_by_segment": expected_by_segment,
        "model_retrained": False, "client_sources_modified": False,
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "expected_by_segment"}, indent=2))
    return result


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ## 2. Train, register and reload
# Run the random forest; compare to the training median. Log parameters, metrics, chart and model artifacts. Register a Fabric ML model version, reload it and verify identical holdout predictions.


# CELL ********************

from datetime import datetime, timezone
demo_run_id = demo_run_id or datetime.now(timezone.utc).strftime('demo_%Y%m%dT%H%M%S%fZ')
try:
    demo_result = (prepare_reporting(demo_run_id, report_source_run_id)
                   if mode == 'reporting' else
                   run_demo(demo_run_id, snapshot_date, source_directory, mode))
except (ValueError, OSError, KeyError, TypeError, RuntimeError, AssertionError,
        mlflow.exceptions.MlflowException) as error:
    error_directory = Path('/lakehouse/default/Files/POC_vehicle_usage_errors')
    error_directory.mkdir(parents=True, exist_ok=True)
    (error_directory / 'last_error.json').write_text(
        json.dumps({'demo_run_id': demo_run_id, 'error_type': type(error).__name__, 'message': str(error)}), encoding='utf-8')
    raise


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ## 3. Read the result
# MAE is in kilometres; lower is better. The baseline is a reference, not an accuracy target. Predictions are stored as Parquet under Files/POC_vehicle_usage, not as a managed SQL table.


# CELL ********************

print(json.dumps(demo_result, indent=2))


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }
