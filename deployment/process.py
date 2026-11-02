"""Validate CSVs and execute the BRP Silver -> ML -> reporting -> Gold chain."""

import argparse
import csv
import io
import json
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import deploy


FILES = {
    "SANITIZED_RAW_CLAIMS.csv": set(),
    "SANITIZED_RAW_VEHICLE_DATA.csv": set(),
    "SANITIZED_RAW_MODEL_FAMILY.csv": {"MODEL_NO", "MODEL_YEAR", "PRODUCT_LINE_GROUP"},
    "SANITIZED_RAW_FLEET_OBSERVATIONS.csv": {
        "SERIAL_NO", "PRODUCT_CODE", "MEASURE_DATE", "VEHICLE_USAGE", "USAGE_UNIT",
        "VALID_OBSERVATION_FLAG", "METAIS_DELETED", "REGISTRATION_DATE", "PLANT", "FLEET_REGION",
    },
}
GOLD_TABLES = ["POC_predictions", "POC_evaluation", "POC_segments", "POC_calendar", "POC_publication", "POC_quality"]


class OneLake:
    def __init__(self, tenant_id, workspace_id):
        self.tenant_id = tenant_id
        self.workspace_id = deploy.guid(workspace_id)
        self.opener = urllib.request.build_opener(deploy.NoRedirect())

    def request(self, method, lakehouse_id, path, *, query=None, headers=None, first_line=False, missing_ok=False):
        deploy.guid(lakehouse_id)
        if path.startswith("/") or any(part in {"", ".", ".."} for part in path.split("/")):
            raise deploy.DeploymentError("Invalid OneLake path.")
        token = deploy.azure_json(
            "account", "get-access-token", "--tenant", self.tenant_id, "--resource", "https://storage.azure.com/",
        )["accessToken"]
        url = f"https://onelake.dfs.fabric.microsoft.com/{self.workspace_id}"
        if query is None:
            url += "/" + lakehouse_id + "/" + urllib.parse.quote(path, safe="/")
        else:
            url += "?" + urllib.parse.urlencode({**query, "directory": lakehouse_id + "/" + path})
        request = urllib.request.Request(url, method=method, headers={
            "Authorization": f"Bearer {token}", "x-ms-version": "2023-11-03", **(headers or {}),
        })
        try:
            with self.opener.open(request, timeout=120) as response:
                body = response.readline(65537) if first_line else response.read(4 * 1024 * 1024 + 1)
                if len(body) > (65536 if first_line else 4 * 1024 * 1024):
                    raise deploy.DeploymentError("OneLake metadata/header exceeds supported size.")
                return dict(response.headers), body
        except urllib.error.HTTPError as error:
            if missing_ok and error.code == 404:
                return None
            raise deploy.DeploymentError(f"OneLake {method} {path}: HTTP {error.code}.") from None

    def metadata(self, lakehouse_id, path):
        response = self.request("HEAD", lakehouse_id, path, missing_ok=True)
        if response is None:
            return None
        headers, _ = response
        # HTTPMessage is case-insensitive; preserve that behavior after conversion.
        headers = {key.lower(): value for key, value in headers.items()}
        return {"etag": headers["etag"], "size": int(headers["content-length"])}

    def header(self, lakehouse_id, path):
        _, body = self.request("GET", lakehouse_id, path, headers={"Range": "bytes=0-65535"}, first_line=True)
        columns = next(csv.reader(io.StringIO(body.decode("utf-8-sig")), strict=True))
        if not columns or len(set(columns)) != len(columns) or any(not c.strip() for c in columns):
            raise deploy.DeploymentError(f"Missing or duplicate CSV headers in {path}.")
        return columns

    def json(self, lakehouse_id, path):
        _, body = self.request("GET", lakehouse_id, path)
        return json.loads(body)

    def list_paths(self, lakehouse_id, path):
        query = {"resource": "filesystem", "recursive": "false"}
        paths = []
        while True:
            response = self.request("GET", lakehouse_id, path, query=query, missing_ok=True)
            if response is None:
                return []
            headers, body = response
            paths.extend(json.loads(body)["paths"])
            continuation = next((v for k, v in headers.items() if k.lower() == "x-ms-continuation"), None)
            if not continuation:
                return paths
            if continuation == query.get("continuation"):
                raise deploy.DeploymentError("Repeated OneLake continuation token.")
            query["continuation"] = continuation


def build_plan(client, storage, config):
    plan = deploy.build_notebook_plan(client, config, runtime_parameters=True)
    if any(n["action"] != "reuse" for n in plan["notebooks"]):
        raise deploy.DeploymentError("Import and verify all notebooks before processing.")
    if storage.workspace_id != plan["workspace"]["id"]:
        raise deploy.DeploymentError("Storage workspace does not match the target.")
    notebooks = {n["name"]: n for n in plan["notebooks"]}
    silver = notebooks["POC_NB_PREPARE_SILVER"]["parameters"]
    ml = notebooks["POC_NB_VEHICLE_USAGE"]["parameters"]
    publisher = notebooks["POC_NB_PUBLISH_REPORTING"]["parameters"]
    lakes = plan["lakehouses"]
    sources = {}
    for filename, required in FILES.items():
        path = f"Files/{silver['source_directory']}/{filename}"
        metadata = storage.metadata(lakes["bronze"]["id"], path)
        if not metadata or metadata["size"] <= 0:
            raise deploy.DeploymentError(f"Missing or empty source CSV: {filename}.")
        columns = storage.header(lakes["bronze"]["id"], path)
        if not required <= set(columns):
            raise deploy.DeploymentError(f"Missing required columns in {filename}: {sorted(required - set(columns))}.")
        sources[filename] = {**metadata, "columns": columns, "path": path}
    for role, path in [
        ("silver", f"Files/{silver['silver_directory']}/{silver['silver_run_id']}"),
        ("gold", f"Files/{ml['results_directory']}/{ml['demo_run_id']}"),
        ("gold", f"Files/{ml['results_directory']}/{publisher['report_run_id']}"),
    ]:
        if storage.metadata(lakes[role]["id"], path):
            raise deploy.DeploymentError("Run output already exists. Choose a fresh run_prefix; no automatic replay.")
    existing = []
    for role, schema, tables in [
        ("silver", silver["silver_schema"], ["poc_observations"]),
        ("gold", publisher["reporting_schema"], GOLD_TABLES),
    ]:
        paths = storage.list_paths(lakes[role]["id"], f"Tables/{schema}")
        names = {p["name"].rsplit("/", 1)[-1].casefold() for p in paths}
        existing.extend(f"{role}.{schema}.{table}" for table in tables if table.casefold() in names)
    if existing and not config.get("processing", {}).get("allow_table_overwrite", False):
        raise deploy.DeploymentError(f"Published tables already exist; explicit overwrite approval required: {existing}.")
    reporting = {**ml, "mode": "reporting", "demo_run_id": publisher["report_run_id"]}
    steps = [
        ("silver", "POC_NB_PREPARE_SILVER", silver, "Spark", "bronze"),
        ("train", "POC_NB_VEHICLE_USAGE", ml, "Jupyter", "gold"),
        ("reporting", "POC_NB_VEHICLE_USAGE", reporting, "Jupyter", "gold"),
        ("publish", "POC_NB_PUBLISH_REPORTING", publisher, "Spark", "gold"),
    ]
    plan.pop("approval_id")
    plan.update(
        scope="processing", sources=sources, existing_tables_to_overwrite=existing,
        steps=[{"step": step, "notebook": name, "id": notebooks[name]["id"],
                "parameters": params, "compute": compute, "default_lakehouse_id": lakes[role]["id"]}
               for step, name, params, compute, role in steps],
        limitations=[
            "Runs Spark/Python compute, trains and registers a model, writes Silver and Gold.",
            "No Bronze table creation, data copy or changes to BRP2.",
            "Gold publication is sequential, not atomic. A failed run can leave partial outputs.",
            "CSV header validation checks fields used by this POC, not every row or unused business field.",
        ],
    )
    return {**plan, "approval_id": deploy.digest(plan)}


def start_notebook(client, workspace_id, step):
    body = {
        "parameters": [{"name": key, "value": value, "type": "Text"} for key, value in step["parameters"].items()],
        "executionData": {"compute": step["compute"], "computeConfiguration": {
            "defaultLakehouse": {"referenceType": "ById", "itemId": step["default_lakehouse_id"],
                                 "workspaceId": workspace_id},
        }},
    }
    if step["compute"] == "Jupyter":
        body["executionData"]["computeConfiguration"]["numCores"] = 2
    status, headers, _ = client.request(
        "POST", f"/workspaces/{workspace_id}/notebooks/{step['id']}/jobs/execute/instances?beta=false", body,
    )
    if status != 202 or not headers.get("Location"):
        raise deploy.DeploymentError("Notebook run was not accepted with a job Location; inspect before retrying.")
    job_id = deploy.guid(urllib.parse.urlsplit(headers["Location"]).path.rstrip("/").rsplit("/", 1)[-1])
    return f"/workspaces/{workspace_id}/items/{step['id']}/jobs/instances/{job_id}"


def wait_job(client, path, timeout=3600):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = client.get(path)
        status = result.get("status")
        if status == "Completed":
            return result
        if status in {"Failed", "Cancelled", "Canceled", "Deduped"}:
            raise deploy.DeploymentError(f"Notebook job {status}: {json.dumps(result.get('failureReason', {}))}")
        if status not in {"NotStarted", "InProgress"}:
            raise deploy.DeploymentError(f"Unknown notebook job status: {status}.")
        time.sleep(15)
    raise deploy.DeploymentError(f"Notebook job timed out and may still be running: {path}. Do not retry blindly.")


def evidence(storage, plan, step, previous):
    workspace_id = plan["workspace"]["id"]
    lakes = plan["lakehouses"]
    params = step["parameters"]
    name = step["step"]
    if name == "silver":
        result = storage.json(lakes["silver"]["id"], f"Files/{params['silver_directory']}/{params['silver_run_id']}/published.json")
        expected = {"status": "published", "workspace_id": workspace_id,
                    "silver_run_id": params["silver_run_id"], "silver_lakehouse_id": lakes["silver"]["id"]}
    else:
        run = params["report_run_id"] if name == "publish" else params["demo_run_id"]
        filename = "published.json" if name == "publish" else "summary.json"
        result = storage.json(lakes["gold"]["id"], f"Files/{params['results_directory']}/{run}/{filename}")
        expected = {
            "status": {"train": "passed", "reporting": "reporting_prepared", "publish": "published"}[name],
            "workspace_id": workspace_id, "gold_lakehouse_id": lakes["gold"]["id"],
        }
        if name != "publish":
            expected["demo_run_id"] = run
        else:
            expected["report_run_id"] = run
        if name in {"reporting", "publish"}:
            expected["source_run_id"] = previous["train"]["demo_run_id"]
            expected["model_version"] = previous["train"]["model_version"]
    expected.update(
        silver_run_id=plan["steps"][0]["parameters"]["silver_run_id"],
        silver_lakehouse_id=lakes["silver"]["id"],
    )
    if name != "publish":
        expected["snapshot_date"] = params["snapshot_date"]
    else:
        expected["schema"] = params["reporting_schema"]
    if name == "train":
        expected.update(
            model_name=params["model_name"], experiment_name=params["experiment_name"],
            registered_model_lineage_verified=True, registered_model_reload_predictions_match=True,
            persisted_prediction_readback_verified=True,
        )
    for key, value in expected.items():
        if result.get(key) != value:
            raise deploy.DeploymentError(f"{name} evidence mismatch: {key}.")
    if name in {"reporting", "publish"} and set(result.get("table_rows", {})) != set(GOLD_TABLES):
        raise deploy.DeploymentError(f"{name} evidence must contain all six Gold tables.")
    if name == "train" and (not isinstance(result.get("model_version"), str) or not result["model_version"].isdigit()):
        raise deploy.DeploymentError("Training evidence requires a registered model version.")
    if name == "publish" and result["table_rows"] != previous["reporting"]["table_rows"]:
        raise deploy.DeploymentError("Published Gold row counts differ from reporting preparation.")
    return result


def run(client, storage, config, approved, approval_id, journal_path):
    if approved.get("approval_id") != approval_id or build_plan(client, storage, config) != approved:
        raise deploy.DeploymentError("Processing plan, inputs or target changed. Generate a fresh plan.")
    if journal_path.exists():
        raise deploy.DeploymentError("Processing journal already exists; choose a fresh path.")
    workspace_id = approved["workspace"]["id"]
    journal = {"status": "in_progress", "workspace_id": workspace_id, "approval_id": approval_id,
               "started_at": datetime.now(timezone.utc).isoformat(), "jobs": {}, "evidence": {}}
    deploy.write_json(journal_path, journal)
    try:
        for step in approved["steps"]:
            name = step["step"]
            journal["pending"] = name
            deploy.write_json(journal_path, journal)
            path = start_notebook(client, workspace_id, step)
            journal["jobs"][name] = {"path": path, "status": "accepted"}
            deploy.write_json(journal_path, journal)
            print(f"{name}: accepted {path}", flush=True)
            result = wait_job(client, path)
            journal["jobs"][name]["status"] = result["status"]
            journal["evidence"][name] = evidence(storage, approved, step, journal["evidence"])
            deploy.write_json(journal_path, journal)
            print(f"{name}: completed and evidence verified", flush=True)
        journal.pop("pending", None)
        journal["status"] = "succeeded"
        deploy.write_json(journal_path, journal)
    except (deploy.DeploymentError, OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as error:
        journal["status"] = "failed"
        journal["error_type"] = type(error).__name__
        deploy.write_json(journal_path, journal)
        raise
    return journal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["plan", "run"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--approve")
    parser.add_argument("--journal", type=Path)
    args = parser.parse_args()
    try:
        config = deploy.load_config(args.config)
        paths = [args.config.resolve(), args.plan.resolve()]
        if args.journal:
            paths.append(args.journal.resolve())
        if len(set(paths)) != len(paths):
            raise deploy.DeploymentError("Use distinct configuration, plan and journal paths.")
        account = deploy.azure_json("account", "show")
        if account["tenantId"] != config["tenant_id"]:
            raise deploy.DeploymentError("Azure CLI tenant does not match configuration.")
        client = deploy.FabricClient(config["tenant_id"])
        storage = OneLake(config["tenant_id"], config["workspace_id"])
        if args.command == "plan":
            if args.approve or args.journal:
                raise deploy.DeploymentError("Approval/journal are run-only.")
            result = build_plan(client, storage, config)
            deploy.write_json(args.plan, result)
            print(json.dumps(result, indent=2))
        else:
            if not args.approve or not args.journal:
                raise deploy.DeploymentError("run requires --approve and --journal.")
            result = run(client, storage, config, json.loads(args.plan.read_text()), args.approve, args.journal)
            print(json.dumps({"status": result["status"], "workspace_id": result["workspace_id"]}))
    except (deploy.DeploymentError, OSError, ValueError, KeyError, TypeError, csv.Error, subprocess.TimeoutExpired) as error:
        print(f"Processing failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
