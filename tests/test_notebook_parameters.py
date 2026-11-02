import ast
import hashlib
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = "11111111-1111-4111-8111-111111111111"
BRONZE = "22222222-2222-4222-8222-222222222222"
SILVER = "33333333-3333-4333-8333-333333333333"
GOLD = "44444444-4444-4444-8444-444444444444"
NAMES = ["POC_NB_PREPARE_SILVER", "POC_NB_VEHICLE_USAGE", "POC_NB_PUBLISH_REPORTING"]


def load_definitions(name):
    source = (ROOT / f"{name}.Notebook" / "notebook-content.py").read_text(encoding="utf-8-sig")
    tree = ast.parse(source)
    parameters = source.split("# PARAMETERS CELL ********************", 1)[1].split(
        "# METADATA ********************", 1
    )[0]
    namespace = {
        "re": re, "json": json, "Path": Path, "pd": pd, "np": np,
        "hashlib": hashlib, "Pipeline": object,
    }
    exec(compile(parameters, name, "exec"), namespace)
    definitions = ast.Module(
        body=[
            node for node in tree.body
            if isinstance(node, ast.FunctionDef)
            or (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in {"FEATURES", "MODEL", "EXPERIMENT"})
        ],
        type_ignores=[],
    )
    exec(compile(definitions, name, "exec"), namespace)
    return namespace


class NotebookParametersTest(unittest.TestCase):
    def setUp(self):
        self.utils = MagicMock()
        self.utils.runtime.context = {
            "currentWorkspaceId": WORKSPACE,
            "defaultLakehouseWorkspaceId": WORKSPACE,
            "defaultLakehouseId": GOLD,
            "defaultLakehouseName": "Target Gold",
        }
        self.utils.lakehouse.get.return_value = {"id": SILVER}
        self.modules = patch.dict(sys.modules, {"notebookutils": self.utils})
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def configured(self, name):
        namespace = load_definitions(name)
        namespace.update(
            medallion_workspace_id=WORKSPACE,
            bronze_lakehouse_id=BRONZE,
            silver_lakehouse_id=SILVER,
            gold_lakehouse_id=GOLD,
            silver_lakehouse_name="Target Silver",
            gold_lakehouse_name="Target Gold",
            source_directory="approved/input",
            silver_directory="custom/silver",
            results_directory="custom/results",
            errors_directory="custom/errors",
            silver_schema="Clean Data",
            reporting_schema="Report Data",
            experiment_name="Target_Experiment",
            model_name="Target_Model",
            silver_run_id="silver_new",
            notebookutils=self.utils,
        )
        if name == NAMES[1]:
            namespace["EXPERIMENT"] = namespace["experiment_name"]
            namespace["MODEL"] = namespace["model_name"]
        self.utils.runtime.context["defaultLakehouseId"] = BRONZE if name == NAMES[0] else GOLD
        return namespace

    def test_notebook_syntax_and_parameter_markers(self):
        for name in NAMES:
            with self.subTest(name=name):
                source = (ROOT / f"{name}.Notebook" / "notebook-content.py").read_text(encoding="utf-8-sig")
                compile(source, name, "exec")
                self.assertEqual(source.count("# PARAMETERS CELL ********************"), 1)

    def test_target_configuration_is_accepted(self):
        for name in NAMES:
            with self.subTest(name=name):
                self.configured(name)["validate_deployment"]()

    def test_wrong_workspace_or_attachment_is_rejected(self):
        for name in NAMES:
            for key in ["currentWorkspaceId", "defaultLakehouseWorkspaceId", "defaultLakehouseId"]:
                with self.subTest(name=name, key=key):
                    namespace = self.configured(name)
                    original = self.utils.runtime.context[key]
                    self.utils.runtime.context[key] = "unexpected"
                    with self.assertRaises(ValueError):
                        namespace["validate_deployment"]()
                    self.utils.runtime.context[key] = original

    def test_invalid_guid_is_rejected(self):
        for name in NAMES:
            namespace = self.configured(name)
            namespace["medallion_workspace_id"] = "workspace-name-not-an-id"
            with self.subTest(name=name), self.assertRaises(ValueError):
                namespace["validate_deployment"]()

    def test_unsafe_paths_are_rejected(self):
        for name in NAMES:
            key = "source_directory" if name == NAMES[0] else "results_directory"
            for value in ["../other", "/absolute", "safe/../other", "safe\\other", "https://host", ""]:
                namespace = self.configured(name)
                namespace[key] = value
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    namespace["validate_deployment"]()

    def test_silver_name_must_match_id(self):
        namespace = self.configured(NAMES[0])
        self.utils.lakehouse.get.return_value = {"id": GOLD}
        with self.assertRaisesRegex(ValueError, "name and ID"):
            namespace["validate_deployment"]()

    def test_invalid_sql_names_are_rejected(self):
        for name, key in [(NAMES[0], "silver_schema"), (NAMES[2], "reporting_schema")]:
            for value in ["", "bad`identifier", "line\nbreak"]:
                namespace = self.configured(name)
                namespace[key] = value
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    namespace["validate_deployment"]()

    def test_overlapping_ml_directories_are_rejected(self):
        namespace = self.configured(NAMES[1])
        namespace["errors_directory"] = "custom/results/errors"
        with self.assertRaisesRegex(ValueError, "overlap"):
            namespace["validate_deployment"]()

    def test_silver_publishes_to_configured_directory_and_catalog(self):
        namespace = self.configured(NAMES[0])
        namespace.update(time=__import__("time"), tempfile=tempfile)
        full = pd.DataFrame({"sampling_key": ["a", "b"], "SERIAL_NO": ["v1", "v2"]})
        namespace["prepare_features"] = MagicMock(side_effect=lambda *a, **k: (full.copy(), {}))
        namespace["split_holdout"] = lambda data, date: (data.iloc[:1], data.iloc[1:])
        spark = MagicMock()
        namespace["SparkSession"] = SimpleNamespace(builder=SimpleNamespace(getOrCreate=lambda: spark))
        readback = spark.table.return_value
        readback.count.return_value = 2
        readback.groupBy.return_value.count.return_value.filter.return_value.count.return_value = 0
        readback.groupBy.return_value.count.return_value.collect.return_value = [
            {"evaluation_role": "train", "count": 1}, {"evaluation_role": "test", "count": 1},
        ]
        readback.filter.return_value.select.return_value.distinct.return_value.join.return_value.count.return_value = 0
        restored = full.assign(sample_selected=True, evaluation_role=["train", "test"])
        readback.toPandas.return_value = restored
        self.utils.fs.exists.return_value = False

        def fake_parquet(frame, path, **kwargs):
            Path(path).write_bytes(b"mock-parquet")

        with patch.object(pd.DataFrame, "to_parquet", fake_parquet):
            result = namespace["prepare_silver"]()
        namespace["prepare_features"].assert_any_call(
            Path("/lakehouse/default/Files") / "approved/input", namespace["snapshot_date"],
            sample_limit=None,
        )
        qualified = "`Target Silver`.`Clean Data`.`poc_observations`"
        spark.read.parquet.return_value.write.format.return_value.mode.return_value.saveAsTable.assert_called_once_with(qualified)
        target = f"abfss://{WORKSPACE}@onelake.dfs.fabric.microsoft.com/{SILVER}/Files/custom/silver/silver_new"
        self.assertEqual(self.utils.fs.put.call_args.args[0], target + "/published.json")
        self.assertEqual(result["delta_table"], qualified)
        self.assertEqual(result["source_directory"], "approved/input")

    def test_ml_reads_configured_silver_path_and_checks_lineage(self):
        namespace = self.configured(NAMES[1])
        content = b"parquet-test"
        evidence = {
            "status": "published", "snapshot_date": namespace["snapshot_date"],
            "silver_lakehouse_id": SILVER, "silver_run_id": "silver_new",
            "workspace_id": WORKSPACE, "eligible_rows": 1,
            "export_sha256": hashlib.sha256(content).hexdigest(),
            "sample_profile": {"sampled_observation_rows": 1},
        }
        response = MagicMock()
        response.json.return_value = evidence
        response.content = content
        requests = MagicMock()
        requests.get.return_value = response
        namespace.update(requests=requests, BytesIO=__import__("io").BytesIO)
        data = pd.DataFrame({
            "sampling_key": ["a"], "product_line": ["ATV"],
            "PLANT": ["Plant"], "FLEET_REGION": ["Region"],
        })
        with patch.object(pd, "read_parquet", return_value=data):
            namespace["load_silver_features"](namespace["snapshot_date"])
        root = f"https://onelake.dfs.fabric.microsoft.com/{WORKSPACE}/{SILVER}/Files/custom/silver/silver_new"
        self.assertEqual(requests.get.call_args_list[0].args[0], root + "/published.json")
        self.assertEqual(requests.get.call_args_list[1].args[0], root + "/observations.parquet")
        evidence["workspace_id"] = GOLD
        with self.assertRaisesRegex(ValueError, "workspace"):
            namespace["load_silver_features"](namespace["snapshot_date"])

    def test_ml_profile_writes_configured_results_path(self):
        namespace = self.configured(NAMES[1])
        namespace.update(time=__import__("time"))
        namespace["load_silver_features"] = MagicMock(return_value=(object(), {"count": 200}))
        namespace["split_holdout"] = MagicMock(return_value=([1] * 100, [2] * 100))
        with tempfile.TemporaryDirectory() as temporary:
            namespace["Path"] = lambda value: Path(temporary)
            result = namespace["run_demo"]("new_run", namespace["snapshot_date"], "approved/input", "profile")
            output = Path(temporary) / "custom" / "results" / "new_run" / "summary.json"
            self.assertEqual(json.loads(output.read_text())["status"], "profiled")
            self.assertEqual(result["status"], "profiled")
            with self.assertRaises(FileExistsError):
                namespace["run_demo"]("new_run", namespace["snapshot_date"], "approved/input", "profile")

    def test_reporting_rejects_wrong_model_before_loading_it(self):
        namespace = self.configured(NAMES[1])
        summary = {
            "status": "passed", "demo_run_id": "trained",
            "silver_run_id": "silver_new", "silver_lakehouse_id": SILVER,
            "gold_lakehouse_id": GOLD, "model_name": "Other_Model",
            "experiment_name": "Target_Experiment",
        }
        with tempfile.TemporaryDirectory() as temporary:
            namespace["Path"] = lambda value: Path(temporary)
            source = Path(temporary) / "custom" / "results" / "trained"
            source.mkdir(parents=True)
            (source / "summary.json").write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, "same model"):
                namespace["prepare_reporting"]("report_new", "trained")

    def test_reporting_uses_recorded_version_and_configured_paths(self):
        namespace = self.configured(NAMES[1])
        row = {
            "age_days": 100, "model_year": 2025, "product_line": "ATV",
            "PLANT": "Plant", "FLEET_REGION": "Region", "target_km": 1200.0,
            "MEASURE_DATE": pd.Timestamp("2026-08-28"),
        }
        data = pd.DataFrame([row])
        profile = dict.fromkeys([
            "source_observation_rows", "excluded_invalid_rows",
            "conflicting_vehicle_date_groups_quarantined",
            "conflicting_observation_rows_quarantined", "duplicate_vehicle_date_rows_removed",
            "eligible_observation_rows", "sampled_observation_rows",
        ], 1)
        model = MagicMock()
        model.predict.side_effect = lambda frame: np.full(len(frame), 1000.0)
        mlflow = MagicMock()
        mlflow.sklearn.load_model.return_value = model
        namespace.update(
            mlflow=mlflow,
            MlflowClient=lambda: SimpleNamespace(get_model_version=lambda *a: SimpleNamespace(run_id="ml_run")),
            mean_absolute_error=lambda actual, predicted: float(np.abs(actual - predicted).mean()),
            load_silver_features=lambda date: (data, profile),
            split_holdout=lambda frame, date: (data, data),
        )
        summary = {
            "status": "passed", "demo_run_id": "trained", "silver_run_id": "silver_new",
            "silver_lakehouse_id": SILVER, "gold_lakehouse_id": GOLD,
            "model_name": "Target_Model", "experiment_name": "Target_Experiment",
            "registered_model_uri": "models:/Target_Model/7", "model_version": "7",
            "snapshot_date": namespace["snapshot_date"], "mlflow_run_id": "ml_run",
            "profile": profile, "metrics": {"mae_km": 200.0}, "prediction_rows": 1,
        }
        predictions = data.assign(predicted_km=1000.0)
        tables = {}

        def fake_write(frame, path, **kwargs):
            tables[str(path)] = frame.copy()

        def fake_read(path, **kwargs):
            return predictions.copy() if Path(path).name == "predictions.parquet" else tables[str(path)].copy()

        with tempfile.TemporaryDirectory() as temporary:
            namespace["Path"] = lambda value: Path(temporary)
            source = Path(temporary) / "custom" / "results" / "trained"
            source.mkdir(parents=True)
            for provenance in [{}, {
                "workspace_id": WORKSPACE, "silver_directory": "custom/silver",
                "results_directory": "custom/results",
            }]:
                summary.update(provenance)
                (source / "summary.json").write_text(json.dumps(summary))
                run_id = "report_new" if provenance else "report_legacy"
                with patch.object(pd.DataFrame, "to_parquet", fake_write), patch.object(pd, "read_parquet", fake_read):
                    result = namespace["prepare_reporting"](run_id, "trained")
                output = Path(temporary) / "custom" / "results" / run_id
                self.assertEqual(json.loads((output / "summary.json").read_text())["model_version"], "7")
                self.assertEqual(len(result["table_rows"]), 6)
                self.assertEqual(result["results_directory"], "custom/results")
                self.assertEqual(result["workspace_id"], WORKSPACE)
                self.assertFalse(result["model_retrained"])
            mlflow.sklearn.load_model.assert_called_with("models:/Target_Model/7")
            for key, value in [
                ("registered_model_uri", "models:/Target_Model/8"),
                ("snapshot_date", "2020-01-01"), ("workspace_id", GOLD),
                ("silver_directory", "wrong"), ("results_directory", "wrong"),
            ]:
                invalid = dict(summary, **{key: value})
                (source / "summary.json").write_text(json.dumps(invalid))
                with self.subTest(key=key), self.assertRaises(ValueError):
                    namespace["prepare_reporting"]("rejected", "trained")

    def test_publication_reads_target_and_writes_renamed_catalog(self):
        namespace = self.configured(NAMES[2])
        namespace.update(time=__import__("time"), F=MagicMock())
        spark = MagicMock()
        namespace["SparkSession"] = SimpleNamespace(builder=SimpleNamespace(getOrCreate=lambda: spark))
        names = ["POC_predictions", "POC_evaluation", "POC_segments", "POC_calendar", "POC_publication", "POC_quality"]
        evidence = {
            "status": "reporting_prepared", "demo_run_id": "report_new",
            "gold_lakehouse_id": GOLD, "workspace_id": WORKSPACE,
            "results_directory": "custom/results", "source_run_id": "trained",
            "model_version": "1", "silver_run_id": "silver_new",
            "silver_lakehouse_id": SILVER, "table_rows": {name: 10 for name in names},
        }
        self.utils.fs.head.return_value = json.dumps(evidence)
        frame = spark.read.parquet.return_value
        frame.withColumn.return_value = frame
        frame.count.return_value = 10
        frame.join.return_value.count.return_value = 0
        spark.table.return_value.count.return_value = 10
        spark.table.return_value.schema.json.return_value = "{}"
        result = namespace["publish_reporting"]("report_new")
        candidate = f"abfss://{WORKSPACE}@onelake.dfs.fabric.microsoft.com/{GOLD}/Files/custom/results/report_new"
        self.assertEqual(self.utils.fs.head.call_args.args[0], candidate + "/summary.json")
        spark.sql.assert_called_once_with("CREATE SCHEMA IF NOT EXISTS `Target Gold`.`Report Data`")
        written = [call.args[0] for call in frame.write.format.return_value.mode.return_value.saveAsTable.call_args_list]
        self.assertEqual(written, [f"`Target Gold`.`Report Data`.`{name}`" for name in names])
        self.assertEqual(result["schema"], "Report Data")
        evidence["workspace_id"] = GOLD
        self.utils.fs.head.return_value = json.dumps(evidence)
        with self.assertRaisesRegex(ValueError, "another workspace"):
            namespace["publish_reporting"]("report_new")


if __name__ == "__main__":
    unittest.main()
