import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from test_deployment import CAPACITY, LAKES, ROOT, TENANT, WORKSPACE, FakeNotebookFabric, deploy


SPEC = importlib.util.spec_from_file_location("reporting", ROOT / "deployment" / "reporting.py")
reporting = importlib.util.module_from_spec(SPEC)
with patch.dict(sys.modules, {"deploy": deploy}):
    SPEC.loader.exec_module(reporting)


class FakeReportingFabric(FakeNotebookFabric):
    def create(self, path, body):
        if path not in [f"/workspaces/{WORKSPACE}/semanticModels", f"/workspaces/{WORKSPACE}/reports"]:
            raise AssertionError(path)
        self.posts.append((path, body))
        item_id = f"88888888-8888-4888-8888-{len(self.posts):012d}"
        kind = "SemanticModel" if path.endswith("semanticModels") else "Report"
        self.items.append({"id": item_id, "displayName": body["displayName"], "type": kind})
        self.definitions[item_id] = body["definition"]
        return {"id": item_id}

    def post_operation(self, path):
        item_id = path.split("/")[-2]
        return {"definition": self.definitions[item_id]}


class ReportingTest(unittest.TestCase):
    def setUp(self):
        self.client = FakeReportingFabric()
        self.config = {
            "tenant_id": TENANT, "capacity_id": CAPACITY, "workspace_name": "BRP_PROMPT_TEST",
            "workspace_id": WORKSPACE, "lakehouses": {role: f"BRP_{role.upper()}" for role in LAKES},
            "notebook_parameters": {"run_prefix": "target01", "reporting_schema": "CustomReporting"},
        }
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def test_rendered_model_uses_target_and_preserves_six_tables(self):
        resources = deploy.build_plan(self.client, self.config)
        parts = reporting.render_model(resources)
        source = parts["definition/expressions.tmdl"].decode()
        self.assertIn(WORKSPACE + "/" + LAKES["gold"], source)
        self.assertNotIn(deploy.PROTECTED_WORKSPACE, source)
        self.assertNotIn(".platform", parts)
        self.assertEqual(len(reporting.model_inventory(parts)), 6)
        for path, data in parts.items():
            if path.startswith("definition/tables/"):
                self.assertIn(b"schemaName: CustomReporting", data)
        original = (ROOT / f"{reporting.MODEL}.SemanticModel" / "definition" / "expressions.tmdl").read_text()
        self.assertIn(deploy.PROTECTED_WORKSPACE, original)

    def test_report_binding_is_connection_to_exact_target_model(self):
        parts = reporting.render_report(WORKSPACE)
        properties = json.loads(parts["definition.pbir"])
        self.assertEqual(properties["datasetReference"], {
            "byConnection": {"connectionString": f"semanticmodelid={WORKSPACE}"},
        })
        self.assertEqual(len(reporting.validate_bindings(parts, reporting.render_model(deploy.build_plan(self.client, self.config)))), 6)

    def test_missing_model_member_is_rejected(self):
        parts = reporting.render_model(deploy.build_plan(self.client, self.config))
        parts["definition/tables/Evaluation.tmdl"] = parts["definition/tables/Evaluation.tmdl"].replace(
            b"measure 'Model MAE'", b"measure 'Other MAE'",
        )
        with self.assertRaisesRegex(deploy.DeploymentError, "bindings differ"):
            reporting.validate_bindings(reporting.render_report(WORKSPACE), parts)

    def test_only_exact_service_generated_connection_expansion_is_normalized(self):
        expected = reporting.render_report(WORKSPACE)
        actual = dict(expected)
        properties = json.loads(actual["definition.pbir"])
        connection = properties["datasetReference"]["byConnection"]
        connection["connectionString"] = (
            "Data Source=powerbi://api.powerbi.com/v1.0/myorg/BRP_PROMPT_TEST;"
            f"initial catalog={reporting.MODEL};integrated security=ClaimsToken;semanticmodelid={WORKSPACE}"
        )
        actual["definition.pbir"] = json.dumps(properties).encode()
        self.assertEqual(reporting.signature(actual, "BRP_PROMPT_TEST"), reporting.signature(expected))
        self.assertNotEqual(reporting.signature(actual, "OTHER_WORKSPACE"), reporting.signature(expected))
        connection["connectionString"] += ";unknown=value"
        actual["definition.pbir"] = json.dumps(properties).encode()
        with self.assertRaises(deploy.DeploymentError):
            reporting.signature(actual, "BRP_PROMPT_TEST")

    def test_create_and_rerun_reuse_model_and_report_without_overwrite(self):
        plan = reporting.build_plan(self.client, self.config)
        self.assertEqual(self.client.posts, [])
        journal = reporting.apply(self.client, self.config, plan, plan["approval_id"], self.directory / "report.json")
        self.assertEqual(journal["status"], "succeeded")
        self.assertTrue(journal["model"]["verified"])
        self.assertTrue(journal["report"]["verified"])
        self.assertEqual(len(self.client.posts), 2)
        report_parts = reporting.decode_parts(self.client.definitions[journal["report"]["id"]])
        connection = json.loads(report_parts["definition.pbir"])["datasetReference"]["byConnection"]["connectionString"]
        self.assertEqual(connection, "semanticmodelid=" + journal["model"]["id"])
        rerun = reporting.build_plan(self.client, self.config)
        self.assertEqual(rerun["model"]["action"], "reuse")
        self.assertEqual(rerun["report"]["action"], "reuse")

    def test_definition_drift_and_name_collision_are_blocked(self):
        self.client.items.append({"id": WORKSPACE, "displayName": reporting.MODEL, "type": "Notebook"})
        with self.assertRaisesRegex(deploy.DeploymentError, "collision"):
            reporting.build_plan(self.client, self.config)
        self.client = FakeReportingFabric()
        plan = reporting.build_plan(self.client, self.config)
        journal = reporting.apply(self.client, self.config, plan, plan["approval_id"], self.directory / "report.json")
        parts = reporting.decode_parts(self.client.definitions[journal["model"]["id"]])
        parts["definition/model.tmdl"] += b"\nannotation ManualChange = true\n"
        self.client.definitions[journal["model"]["id"]] = reporting.encode_parts(parts, "TMDL")
        with self.assertRaisesRegex(deploy.DeploymentError, "differs"):
            reporting.build_plan(self.client, self.config)

    def test_approval_changes_and_existing_journal_prevent_creation(self):
        plan = reporting.build_plan(self.client, self.config)
        with self.assertRaises(deploy.DeploymentError):
            reporting.apply(self.client, self.config, plan, "wrong", self.directory / "report.json")
        journal = self.directory / "report.json"
        journal.write_text("existing")
        with self.assertRaises(deploy.DeploymentError):
            reporting.apply(self.client, self.config, plan, plan["approval_id"], journal)
        self.assertEqual(self.client.posts, [])

    def validation_fixture(self):
        counts = {name: index + 1 for index, name in enumerate(reporting.TABLES.values())}
        train = {"model_version": "2", "demo_run_id": "train_next", "metrics": {
            "mae_km": 10.0, "baseline_mae_km": 20.0, "rmse_km": 15.0,
        }, "mae_improvement_fraction": 0.5}
        publish = {"report_run_id": "report_next", "table_rows": counts}
        evidence = {"train": train, "publish": publish}
        row = {**counts, **train["metrics"], "improvement": 0.5, "model_version": "2",
               "ml_run": "train_next", "reporting_run": "report_next"}
        result = {"results": [{"tables": [{"rows": [{"[" + k + "]": v for k, v in row.items()}]}]}]}
        return evidence, result

    def test_live_validation_compares_counts_metrics_and_publication_lineage(self):
        evidence, result = self.validation_fixture()
        self.assertEqual(reporting.compare_data(result, evidence)["model_version"], "2")
        for key, value in [("[model_version]", "old"), ("[POC_predictions]", 999),
                           ("[mae_km]", None), ("[rmse_km]", float("nan")), ("[improvement]", True)]:
            with self.subTest(key=key):
                _, altered = self.validation_fixture()
                altered["results"][0]["tables"][0]["rows"][0][key] = value
                with self.assertRaises(deploy.DeploymentError):
                    reporting.compare_data(altered, evidence)
        with self.assertRaises(deploy.DeploymentError):
            reporting.compare_data({"results": [{"error": {"code": "QueryFailed"}}]}, evidence)

    def test_refresh_records_handle_then_validates_report_and_live_data(self):
        evidence, query = self.validation_fixture()
        processing = {"status": "succeeded", "workspace_id": WORKSPACE, "evidence": evidence}
        deployed = {"status": "succeeded", "workspace_id": WORKSPACE,
                    "model": {"id": LAKES["gold"]}, "report": {"id": LAKES["silver"]}}
        plan = {role: {**deployed[role], "action": "reuse"} for role in ("model", "report")}
        pbi = Mock()
        pbi.request.side_effect = [
            (202, {"Location": "https://regional.example/refreshes/" + CAPACITY}, {}),
            (200, {}, {"status": "Completed"}),
            (200, {}, {"datasetId": LAKES["gold"], "webUrl": "https://example.test/report"}),
            (200, {}, query),
        ]
        with patch.object(reporting, "build_plan", return_value=plan):
            result = reporting.validate_live(self.client, pbi, self.config, processing, deployed,
                                             self.directory / "live.json", refresh=True)
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(result["refresh_id"], CAPACITY)
        self.assertEqual(result["visual_validation"], "not_performed")
        self.assertIn(CAPACITY, pbi.request.call_args_list[1].args[1])
        self.assertEqual(pbi.request.call_args_list[0].args[2]["retryCount"], 0)

    def test_failed_refresh_stops_without_querying_or_retrying(self):
        evidence, _ = self.validation_fixture()
        processing = {"status": "succeeded", "workspace_id": WORKSPACE, "evidence": evidence}
        deployed = {"status": "succeeded", "workspace_id": WORKSPACE,
                    "model": {"id": LAKES["gold"]}, "report": {"id": LAKES["silver"]}}
        plan = {role: {**deployed[role], "action": "reuse"} for role in ("model", "report")}
        pbi = Mock()
        pbi.request.side_effect = [
            (202, {"Location": "https://regional.example/refreshes/" + CAPACITY}, {}),
            (200, {}, {"status": "Failed", "messages": [{"code": "CredentialsError"}]}),
        ]
        path = self.directory / "failed-live.json"
        with patch.object(reporting, "build_plan", return_value=plan):
            with self.assertRaisesRegex(deploy.DeploymentError, "Power BI Service"):
                reporting.validate_live(self.client, pbi, self.config, processing, deployed, path, refresh=True)
        self.assertEqual(pbi.request.call_count, 2)
        self.assertEqual(json.loads(path.read_text())["refresh_id"], CAPACITY)
        self.assertEqual(json.loads(path.read_text())["status"], "failed")


if __name__ == "__main__":
    unittest.main()
