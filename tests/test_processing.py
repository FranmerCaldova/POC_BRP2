import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from test_deployment import CAPACITY, LAKES, ROOT, TENANT, WORKSPACE, FakeNotebookFabric, deploy


SPEC = importlib.util.spec_from_file_location("process", ROOT / "deployment" / "process.py")
process = importlib.util.module_from_spec(SPEC)
with patch.dict(sys.modules, {"deploy": deploy}):
    SPEC.loader.exec_module(process)


class FakeStorage:
    workspace_id = WORKSPACE

    def __init__(self):
        self.files = {name: {"etag": "etag1", "size": 100} for name in process.FILES}
        self.headers = {name: sorted(required) or ["unused_column"] for name, required in process.FILES.items()}
        self.existing = []
        self.output_exists = False

    def metadata(self, lakehouse_id, path):
        name = path.rsplit("/", 1)[-1]
        if name.endswith(".csv"):
            return self.files.get(name)
        return {"etag": "output", "size": 0} if self.output_exists else None

    def header(self, lakehouse_id, path):
        return self.headers[path.rsplit("/", 1)[-1]]

    def list_paths(self, lakehouse_id, path):
        return [{"name": path + "/" + name} for name in self.existing]

    def json(self, lakehouse_id, path):
        rows = {name: 10 for name in process.GOLD_TABLES}
        if lakehouse_id == LAKES["silver"]:
            return {"status": "published", "workspace_id": WORKSPACE,
                    "silver_run_id": "silver_target01", "silver_lakehouse_id": LAKES["silver"],
                    "snapshot_date": "2026-08-28"}
        base = {"workspace_id": WORKSPACE, "gold_lakehouse_id": LAKES["gold"], "model_version": "1",
                "silver_run_id": "silver_target01", "silver_lakehouse_id": LAKES["silver"],
                "snapshot_date": "2026-08-28"}
        if "/train_target01/" in path:
            return {**base, "status": "passed", "demo_run_id": "train_target01",
                    "model_name": "POC_ML_VEHICLE_USAGE", "experiment_name": "POC_EXP_VEHICLE_USAGE",
                    "registered_model_lineage_verified": True, "registered_model_reload_predictions_match": True,
                    "persisted_prediction_readback_verified": True}
        if path.endswith("published.json"):
            return {**base, "status": "published", "report_run_id": "report_target01",
                    "source_run_id": "train_target01", "table_rows": rows, "schema": "POC_reporting"}
        return {**base, "status": "reporting_prepared", "demo_run_id": "report_target01",
                "source_run_id": "train_target01", "table_rows": rows}


class ProcessingTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.client = FakeNotebookFabric()
        self.config = {
            "tenant_id": TENANT, "capacity_id": CAPACITY, "workspace_name": "BRP_PROMPT_TEST",
            "workspace_id": WORKSPACE, "lakehouses": {role: f"BRP_{role.upper()}" for role in LAKES},
            "notebook_parameters": {"run_prefix": "target01", "snapshot_date": "2026-08-28"},
        }
        plan = deploy.build_notebook_plan(self.client, self.config)
        deploy.apply_plan(self.client, self.config, plan, plan["approval_id"], self.directory / "import.json", "notebooks")
        self.storage = FakeStorage()

    def test_plan_validates_sources_and_orders_parameterized_steps(self):
        plan = process.build_plan(self.client, self.storage, self.config)
        self.assertEqual(len(plan["sources"]), 4)
        self.assertEqual([s["step"] for s in plan["steps"]], ["silver", "train", "reporting", "publish"])
        self.assertEqual(plan["steps"][2]["parameters"]["mode"], "reporting")
        self.assertEqual(plan["steps"][2]["parameters"]["demo_run_id"], "report_target01")
        self.assertEqual(plan["steps"][2]["parameters"]["report_source_run_id"], "train_target01")
        self.assertEqual(len(self.client.posts), 3)

    def test_missing_empty_or_wrong_header_files_rejected(self):
        name = "SANITIZED_RAW_MODEL_FAMILY.csv"
        for value in [None, {"etag": "x", "size": 0}]:
            self.storage.files[name] = value
            with self.assertRaisesRegex(deploy.DeploymentError, "Missing or empty"):
                process.build_plan(self.client, self.storage, self.config)
        self.storage.files[name] = {"etag": "x", "size": 100}
        self.storage.headers[name] = ["MODEL_NO"]
        with self.assertRaisesRegex(deploy.DeploymentError, "required columns"):
            process.build_plan(self.client, self.storage, self.config)

    def test_fresh_runtime_prefix_reuses_unchanged_notebook_definitions(self):
        definitions = json.dumps(self.client.definitions, sort_keys=True)
        self.config["notebook_parameters"]["run_prefix"] = "next02"
        plan = process.build_plan(self.client, self.storage, self.config)
        self.assertEqual([s["parameters"].get("demo_run_id") for s in plan["steps"][1:3]],
                         ["train_next02", "report_next02"])
        self.assertEqual(plan["steps"][0]["parameters"]["silver_run_id"], "silver_next02")
        self.assertTrue(all(n["stored_definition_sha256"] for n in plan["notebooks"]))
        self.assertEqual(definitions, json.dumps(self.client.definitions, sort_keys=True))
        with self.assertRaisesRegex(deploy.DeploymentError, "differs"):
            deploy.build_notebook_plan(self.client, self.config)

    def test_run_folders_and_unapproved_table_overwrites_rejected(self):
        self.storage.output_exists = True
        with self.assertRaisesRegex(deploy.DeploymentError, "fresh run_prefix"):
            process.build_plan(self.client, self.storage, self.config)
        self.storage.output_exists = False
        self.storage.existing = ["poc_predictions"]
        with self.assertRaisesRegex(deploy.DeploymentError, "overwrite approval"):
            process.build_plan(self.client, self.storage, self.config)
        self.config["processing"] = {"allow_table_overwrite": True}
        plan = process.build_plan(self.client, self.storage, self.config)
        self.assertTrue(plan["existing_tables_to_overwrite"])

    def test_changed_source_etag_invalidates_approval_before_any_job(self):
        plan = process.build_plan(self.client, self.storage, self.config)
        self.storage.files["SANITIZED_RAW_MODEL_FAMILY.csv"]["etag"] = "new-etag"
        with patch.object(process, "start_notebook") as start:
            with self.assertRaisesRegex(deploy.DeploymentError, "changed"):
                process.run(self.client, self.storage, self.config, plan, plan["approval_id"], self.directory / "run.json")
            start.assert_not_called()

    def test_sequence_requires_completed_jobs_and_verified_evidence(self):
        plan = process.build_plan(self.client, self.storage, self.config)
        sequence = []
        with patch.object(process, "start_notebook", side_effect=lambda c, w, step: sequence.append(step["step"]) or "/job"):
            with patch.object(process, "wait_job", return_value={"status": "Completed"}):
                result = process.run(self.client, self.storage, self.config, plan, plan["approval_id"], self.directory / "run.json")
        self.assertEqual(sequence, ["silver", "train", "reporting", "publish"])
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(result["evidence"]["publish"]["model_version"], "1")

    def test_failure_stops_downstream_and_preserves_job_handle(self):
        plan = process.build_plan(self.client, self.storage, self.config)
        with patch.object(process, "start_notebook", return_value="/accepted/job") as start:
            with patch.object(process, "wait_job", side_effect=deploy.DeploymentError("Failed")):
                with self.assertRaises(deploy.DeploymentError):
                    process.run(self.client, self.storage, self.config, plan, plan["approval_id"], self.directory / "run.json")
        self.assertEqual(start.call_count, 1)
        journal = json.loads((self.directory / "run.json").read_text())
        self.assertEqual(journal["status"], "failed")
        self.assertEqual(journal["jobs"]["silver"]["path"], "/accepted/job")

    def test_evidence_wrong_target_or_model_rejected(self):
        plan = process.build_plan(self.client, self.storage, self.config)
        previous = {"train": {"demo_run_id": "train_target01", "model_version": "1"}}
        with patch.object(self.storage, "json", return_value={
            "status": "reporting_prepared", "workspace_id": TENANT,
        }):
            with self.assertRaisesRegex(deploy.DeploymentError, "mismatch"):
                process.evidence(self.storage, plan, plan["steps"][2], previous)

    def test_start_payload_and_public_job_location(self):
        step = process.build_plan(self.client, self.storage, self.config)["steps"][1]
        job_id = "88888888-8888-4888-8888-888888888888"
        with patch.object(self.client, "request", create=True, return_value=(
            202, {"Location": "https://regional.example/jobs/instances/" + job_id}, None,
        )) as request:
            path = process.start_notebook(self.client, WORKSPACE, step)
        self.assertEqual(path, f"/workspaces/{WORKSPACE}/items/{step['id']}/jobs/instances/{job_id}")
        payload = request.call_args.args[2]
        self.assertEqual(payload["executionData"]["compute"], "Jupyter")
        self.assertEqual(payload["executionData"]["computeConfiguration"]["numCores"], 2)
        self.assertTrue(all(p["type"] == "Text" for p in payload["parameters"]))

    def test_job_poll_requires_terminal_completion_and_handles_timeout(self):
        with patch.object(self.client, "get", side_effect=[{"status": "InProgress"}, {"status": "Completed"}]):
            with patch.object(process.time, "sleep"):
                self.assertEqual(process.wait_job(self.client, "/job")["status"], "Completed")
        for status in ["Failed", "Cancelled", "Deduped", "unknown"]:
            with patch.object(self.client, "get", return_value={"status": status}):
                with self.assertRaises(deploy.DeploymentError):
                    process.wait_job(self.client, "/job")
        with patch.object(process.time, "monotonic", side_effect=[0, 4000]):
            with self.assertRaisesRegex(deploy.DeploymentError, "may still be running"):
                process.wait_job(self.client, "/job")


if __name__ == "__main__":
    unittest.main()
