import importlib.util
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("deploy", ROOT / "deployment" / "deploy.py")
deploy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(deploy)

TENANT = "11111111-1111-4111-8111-111111111111"
CAPACITY = "22222222-2222-4222-8222-222222222222"
WORKSPACE = "33333333-3333-4333-8333-333333333333"
LAKES = {
    "bronze": "44444444-4444-4444-8444-444444444444",
    "gold": "55555555-5555-4555-8555-555555555555",
    "silver": "66666666-6666-4666-8666-666666666666",
}


class FakeFabric:
    tenant_id = TENANT

    def __init__(self):
        self.capacity = {"id": CAPACITY, "displayName": "Test", "sku": "F16", "state": "Active"}
        self.workspaces = []
        self.items = []
        self.posts = []

    def list(self, path):
        if path == "/capacities":
            return [self.capacity]
        if path == "/workspaces":
            return self.workspaces
        if path == f"/workspaces/{WORKSPACE}/items":
            return self.items
        raise AssertionError(path)

    def get(self, path):
        if path == f"/workspaces/{WORKSPACE}":
            return {"id": WORKSPACE, "capacityId": CAPACITY}
        lake_id = path.rsplit("/", 1)[-1]
        for role, value in LAKES.items():
            if value == lake_id:
                return {"id": value, "displayName": f"BRP_{role.upper()}",
                        "properties": {"defaultSchema": "dbo"}}
        raise AssertionError(path)

    def create(self, path, body):
        self.posts.append((path, body))
        if path == "/workspaces":
            self.workspaces = [{"id": WORKSPACE, "displayName": body["displayName"]}]
            return {"id": WORKSPACE}
        role = body["displayName"].split("_")[1].lower()
        item = {"id": LAKES[role], "displayName": body["displayName"], "type": "Lakehouse"}
        self.items.append(item)
        return item


class DeploymentTest(unittest.TestCase):
    def setUp(self):
        self.config = {
            "tenant_id": TENANT, "capacity_id": CAPACITY, "workspace_name": "BRP_PROMPT_TEST",
            "lakehouses": {role: f"BRP_{role.upper()}" for role in LAKES},
        }
        self.client = FakeFabric()
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def validate(self, config):
        target = self.directory / "target.json"
        target.write_text(json.dumps(config))
        return deploy.load_config(target)

    def test_config(self):
        self.assertEqual(self.validate(self.config), self.config)
        for changes in [
            {"workspace_name": "BRP2"}, {"workspace_id": deploy.PROTECTED_WORKSPACE},
            {"tenant_id": "invalid"}, {"allow_overwrite": True},
            {"lakehouses": {"bronze": "same", "silver": "Same", "gold": "other"}},
            {"lakehouses": {"bronze": "bad/path", "silver": "silver", "gold": "gold"}},
        ]:
            with self.subTest(changes=changes), self.assertRaises(deploy.DeploymentError):
                self.validate(dict(self.config, **changes))

    def test_plan_is_read_only_and_deterministic(self):
        first = deploy.build_plan(self.client, self.config)
        self.assertEqual(first, deploy.build_plan(self.client, self.config))
        self.assertEqual(first["workspace"]["action"], "create")
        self.assertTrue(all(v["action"] == "create" for v in first["lakehouses"].values()))
        self.assertEqual(self.client.posts, [])

    def test_wrong_tenant_or_capacity(self):
        for changes in [{"tenant_id": WORKSPACE}, {"capacity_id": WORKSPACE}]:
            with self.subTest(changes=changes), self.assertRaises(deploy.DeploymentError):
                deploy.build_plan(self.client, dict(self.config, **changes))
        for changes in [{"state": "Inactive"}, {"sku": "PP3"}]:
            self.client = FakeFabric()
            self.client.capacity.update(changes)
            with self.subTest(changes=changes), self.assertRaises(deploy.DeploymentError):
                deploy.build_plan(self.client, self.config)

    def test_workspace_collision_requires_explicit_id(self):
        self.client.workspaces = [{"id": WORKSPACE, "displayName": self.config["workspace_name"]}]
        with self.assertRaisesRegex(deploy.DeploymentError, "already exists"):
            deploy.build_plan(self.client, self.config)
        self.config["workspace_id"] = WORKSPACE
        self.assertEqual(deploy.build_plan(self.client, self.config)["workspace"]["action"], "reuse")
        self.client.workspaces *= 2
        with self.assertRaisesRegex(deploy.DeploymentError, "Ambiguous"):
            deploy.build_plan(self.client, self.config)

    def test_protected_id_is_blocked_even_with_another_name(self):
        self.client.workspaces = [{"id": deploy.PROTECTED_WORKSPACE,
                                   "displayName": self.config["workspace_name"]}]
        with self.assertRaisesRegex(deploy.DeploymentError, "protected"):
            deploy.build_plan(self.client, self.config)

    def test_wrong_workspace_capacity_or_non_lakehouse_rejected(self):
        self.client.workspaces = [{"id": WORKSPACE, "displayName": self.config["workspace_name"]}]
        self.config["workspace_id"] = WORKSPACE
        with patch.object(self.client, "get", return_value={"capacityId": WORKSPACE}):
            with self.assertRaisesRegex(deploy.DeploymentError, "capacity"):
                deploy.build_plan(self.client, self.config)
        self.client.items = [{"id": LAKES["bronze"], "displayName": "BRP_BRONZE", "type": "Notebook"}]
        with self.assertRaisesRegex(deploy.DeploymentError, "non-lakehouse"):
            deploy.build_plan(self.client, self.config)

    def test_schema_support_required_for_reuse(self):
        self.client.workspaces = [{"id": WORKSPACE, "displayName": self.config["workspace_name"]}]
        self.config["workspace_id"] = WORKSPACE
        self.client.items = [{"id": LAKES["bronze"], "displayName": "BRP_BRONZE", "type": "Lakehouse"}]
        original = self.client.get
        with patch.object(self.client, "get", side_effect=lambda path: (
            {} if "/lakehouses/" in path else original(path)
        )):
            with self.assertRaisesRegex(deploy.DeploymentError, "schema"):
                deploy.build_plan(self.client, self.config)

    def test_apply_requires_matching_approval_and_unchanged_state(self):
        plan = deploy.build_plan(self.client, self.config)
        for approval in ["invalid", plan["approval_id"]]:
            if approval == plan["approval_id"]:
                self.client.capacity["displayName"] = "Changed"
            with self.subTest(approval=approval), self.assertRaises(deploy.DeploymentError):
                deploy.apply_plan(self.client, self.config, plan, approval, self.directory / "journal.json")
        self.assertEqual(self.client.posts, [])

    def test_apply_verifies_resources_and_rerun_is_noop(self):
        plan = deploy.build_plan(self.client, self.config)
        journal_path = self.directory / "journal.json"
        journal = deploy.apply_plan(self.client, self.config, plan, plan["approval_id"], journal_path)
        self.assertEqual(journal["status"], "succeeded")
        self.assertEqual(journal["workspace_id"], WORKSPACE)
        self.assertEqual(set(journal["lakehouses"]), {"bronze", "silver", "gold"})
        self.assertEqual(len(self.client.posts), 4)
        self.assertEqual(self.client.posts[0][1]["capacityId"], CAPACITY)
        self.assertTrue(all(body["creationPayload"]["enableSchemas"] for _, body in self.client.posts[1:]))
        self.assertEqual(json.loads(journal_path.read_text()), journal)
        self.config["workspace_id"] = WORKSPACE
        rerun = deploy.build_plan(self.client, self.config)
        self.assertTrue(all(v["action"] == "reuse" for v in rerun["lakehouses"].values()))
        deploy.apply_plan(self.client, self.config, rerun, rerun["approval_id"], self.directory / "rerun.json")
        self.assertEqual(len(self.client.posts), 4)

    def test_failure_retains_created_ids_and_does_not_retry_posts(self):
        plan = deploy.build_plan(self.client, self.config)
        original = self.client.create

        def fail(path, body):
            if body.get("displayName") == "BRP_GOLD":
                raise deploy.DeploymentError("Simulated failure")
            return original(path, body)

        journal_path = self.directory / "journal.json"
        with patch.object(self.client, "create", side_effect=fail):
            with self.assertRaisesRegex(deploy.DeploymentError, "Simulated"):
                deploy.apply_plan(self.client, self.config, plan, plan["approval_id"], journal_path)
        journal = json.loads(journal_path.read_text())
        self.assertEqual(journal["status"], "failed")
        self.assertEqual(journal["workspace_id"], WORKSPACE)
        self.assertIn("bronze", journal["lakehouses"])
        self.assertEqual(journal["pending"], "create_gold")
        self.assertEqual(len(self.client.posts), 2)

    def test_existing_journal_not_overwritten(self):
        plan = deploy.build_plan(self.client, self.config)
        journal = self.directory / "journal.json"
        journal.write_text("existing evidence")
        with self.assertRaisesRegex(deploy.DeploymentError, "Journal already exists"):
            deploy.apply_plan(self.client, self.config, plan, plan["approval_id"], journal)
        self.assertEqual(journal.read_text(), "existing evidence")
        self.assertEqual(self.client.posts, [])


class TransportTest(unittest.TestCase):
    def test_external_or_redirect_urls_rejected(self):
        for url in [
            "https://evil.example/v1/workspaces", "http://api.fabric.microsoft.com/v1/workspaces",
            "https://api.fabric.microsoft.com.evil.example/v1/workspaces",
            "https://user@api.fabric.microsoft.com/v1/workspaces",
            "https://api.fabric.microsoft.com/v2/workspaces",
        ]:
            with self.subTest(url=url), self.assertRaises(deploy.DeploymentError):
                deploy.safe_url(url)
        self.assertEqual(deploy.safe_url("/workspaces"), deploy.API + "/workspaces")
        with self.assertRaises(deploy.DeploymentError):
            deploy.NoRedirect().redirect_request(None, None, 302, "", {}, "https://evil.example")

    def test_pagination_visits_every_page(self):
        client = deploy.FabricClient(TENANT)
        continuation = deploy.API + "/workspaces?continuationToken=next"
        with patch.object(client, "get", side_effect=[
            {"value": [1], "continuationUri": continuation}, {"value": [2]},
        ]) as get:
            self.assertEqual(client.list("/workspaces"), [1, 2])
            self.assertEqual(get.call_count, 2)
        with patch.object(client, "get", return_value={"value": [], "continuationUri": deploy.API + "/workspaces"}):
            with self.assertRaisesRegex(deploy.DeploymentError, "Repeated"):
                client.list("/workspaces")
        with patch.object(client, "get", return_value={"value": [], "continuationToken": "more"}):
            with self.assertRaisesRegex(deploy.DeploymentError, "incomplete"):
                client.list("/workspaces")

    def test_async_creation_waits_for_success_and_result(self):
        client = deploy.FabricClient(TENANT)
        with patch.object(client, "request", side_effect=[
            (202, {"Location": deploy.API + "/operations/op"}, {}),
            (200, {}, {"status": "Running"}),
            (200, {}, {"status": "Succeeded"}),
            (200, {}, {"id": WORKSPACE}),
        ]) as request, patch.object(deploy.time, "sleep"):
            self.assertEqual(client.create("/workspaces", {}), {"id": WORKSPACE})
            self.assertEqual(request.call_args.args, ("GET", deploy.API + "/operations/op/result"))

    def test_async_failed_missing_location_and_timeout_are_errors(self):
        for responses in [
            [(202, {}, {})],
            [(202, {"Location": deploy.API + "/operations/op"}, {}), (200, {}, {"status": "Failed"})],
        ]:
            client = deploy.FabricClient(TENANT)
            with patch.object(client, "request", side_effect=responses), patch.object(deploy.time, "sleep"):
                with self.assertRaises(deploy.DeploymentError):
                    client.create("/workspaces", {})
        with patch.object(client, "request", return_value=(202, {"Location": deploy.API + "/operations/op"}, {})):
            with patch.object(deploy.time, "monotonic", side_effect=[0, 1000]):
                with self.assertRaisesRegex(deploy.DeploymentError, "timed out"):
                    client.create("/workspaces", {})

    def test_http_failure_does_not_expose_token_or_server_body(self):
        client = deploy.FabricClient(TENANT)
        error = urllib.error.HTTPError(deploy.API + "/workspaces", 403, "sensitive body", {}, None)
        with patch.object(deploy, "azure_json", return_value={"accessToken": "test-secret"}):
            with patch.object(client.opener, "open", side_effect=error):
                with self.assertRaises(deploy.DeploymentError) as raised:
                    client.get("/workspaces")
        self.assertIn("HTTP 403", str(raised.exception))
        self.assertNotIn("test-secret", str(raised.exception))
        self.assertNotIn("sensitive body", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
