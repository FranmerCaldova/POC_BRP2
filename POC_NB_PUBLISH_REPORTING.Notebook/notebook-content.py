# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "fbb63bb5-9be9-413e-98f8-5483d1552137",
# META       "default_lakehouse_name": "POC_LH_VEHICLE_USAGE_GOLD",
# META       "default_lakehouse_workspace_id": "f78fd47f-e385-4179-b31a-5e376322c392"
# META     }
# META   }
# META }

# PARAMETERS CELL ********************

report_run_id = "report_final_20261016"
medallion_workspace_id = "f78fd47f-e385-4179-b31a-5e376322c392"
gold_lakehouse_id = "fbb63bb5-9be9-413e-98f8-5483d1552137"


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# # BRP2 - reporting adapter
# Only publishes prepared files into POC_reporting Delta tables. No training and no source CSV changes.

# CELL ********************

"""Materialize prepared reporting Parquet as schema-qualified Delta tables."""

import json
import re
import time

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def publish_reporting(report_run_id: str) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", report_run_id):
        raise ValueError("Invalid reporting execution identifier.")
    started = time.monotonic()
    spark = SparkSession.builder.getOrCreate()
    import notebookutils
    root = f"abfss://{medallion_workspace_id}@onelake.dfs.fabric.microsoft.com/{gold_lakehouse_id}"
    candidate = f"{root}/Files/POC_vehicle_usage/{report_run_id}"
    evidence = json.loads(notebookutils.fs.head(candidate + "/summary.json", 1000000))
    if evidence["status"] != "reporting_prepared" or evidence["demo_run_id"] != report_run_id:
        raise ValueError("Reporting candidate is not validated.")
    if evidence["gold_lakehouse_id"] != gold_lakehouse_id:
        raise ValueError("Candidate belongs to another Gold lakehouse.")
    names = ["POC_predictions", "POC_evaluation", "POC_segments",
             "POC_calendar", "POC_publication", "POC_quality"]
    frames = {name: spark.read.parquet(f"{candidate}/{name}.parquet") for name in names}
    # Keep older validated candidates reusable while publishing English labels.
    for table, column in [("POC_predictions", "mileage_band"), ("POC_evaluation", "error_band")]:
        frames[table] = frames[table].withColumn(column, F.regexp_replace(column, "et plus", "and above"))
    check_names = [
        "Source observations", "Invalid rows excluded", "Conflicting groups excluded",
        "Conflicting rows excluded", "Duplicate rows removed", "Eligible observations",
        "Deterministic sample", "Training rows", "Test rows",
    ]
    labels = F.array(*[F.lit(name) for name in check_names])
    frames["POC_quality"] = frames["POC_quality"].withColumn(
        "check_name", F.element_at(labels, F.col("check_order").cast("int")),
    ).withColumn("scope", F.lit("Global population"))
    frames["POC_segments"] = frames["POC_segments"].withColumn(
        "region", F.when(F.col("region") == "Non renseignée", F.lit("Unspecified")).otherwise(F.col("region")),
    )
    for name, frame in frames.items():
        if frame.count() != evidence["table_rows"][name]:
            raise AssertionError(f"Candidate row mismatch for {name}.")
    for name in ["POC_predictions", "POC_evaluation"]:
        if frames[name].join(frames["POC_segments"], "segment_key", "left_anti").count():
            raise AssertionError("Orphan segment rows.")
        if frames[name].join(
            frames["POC_calendar"],
            frames[name].measurement_date == frames["POC_calendar"].date,
            "left_anti",
        ).count():
            raise AssertionError("Orphan date rows.")
    spark.sql("CREATE SCHEMA IF NOT EXISTS `POC_LH_VEHICLE_USAGE_GOLD`.`POC_reporting`")
    schemas = {}
    for name, frame in frames.items():
        qualified = f"`POC_LH_VEHICLE_USAGE_GOLD`.`POC_reporting`.`{name}`"
        # Writes are restricted to the POC reporting namespace, never client Files.
        frame.write.format("delta").mode("overwrite").saveAsTable(qualified)
        persisted = spark.table(qualified)
        if persisted.count() != evidence["table_rows"][name]:
            raise AssertionError(f"Published row mismatch for {name}.")
        schemas[name] = json.loads(persisted.schema.json())
    result = {
        "status": "published", "report_run_id": report_run_id,
        "source_run_id": evidence["source_run_id"], "model_version": evidence["model_version"],
        "schema": "POC_reporting", "table_rows": evidence["table_rows"],
        "table_schemas": schemas, "elapsed_seconds": round(time.monotonic() - started, 2),
        "report_language": "en-US",
        "gold_lakehouse_id": gold_lakehouse_id, "silver_run_id": evidence["silver_run_id"],
        "silver_lakehouse_id": evidence["silver_lakehouse_id"],
    }
    notebookutils.fs.put(candidate + "/published.json", json.dumps(result, indent=2), True)
    print(json.dumps(result, indent=2))
    return result


reporting_publication = publish_reporting(report_run_id)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
