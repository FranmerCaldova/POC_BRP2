"""Plan or deploy Fabric workspace/lakehouses and notebook definitions. No execution."""

import argparse
import ast
import base64
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path
from uuid import UUID


API = "https://api.fabric.microsoft.com/v1"
PROTECTED_WORKSPACE = "f78fd47f-e385-4179-b31a-5e376322c392"
ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = {
    "POC_NB_PREPARE_SILVER": ("bronze", "silver"),
    "POC_NB_VEHICLE_USAGE": ("gold", "silver"),
    "POC_NB_PUBLISH_REPORTING": ("gold",),
}
PARAMETER_MARKER = "# PARAMETERS CELL ********************"
METADATA_MARKER = "# METADATA ********************"


class DeploymentError(Exception):
    pass


def azure_json(*arguments):
    executable = shutil.which("az")
    if not executable:
        raise DeploymentError("Azure CLI is missing. Install it and run az login first.")
    result = subprocess.run(
        [executable, *arguments, "--output", "json", "--only-show-errors"],
        capture_output=True, text=True, timeout=120, check=False,
    )
    if result.returncode:
        raise DeploymentError(
            "Azure CLI authentication failed. Run az login for the target tenant; "
            "use Azure CLI directly for detailed diagnostics."
        )
    return json.loads(result.stdout)


def safe_url(path):
    url = API + path if path.startswith("/") else path
    parsed = urllib.parse.urlsplit(url)
    if (parsed.scheme != "https" or parsed.netloc != "api.fabric.microsoft.com"
            or not parsed.path.startswith("/v1/") or parsed.fragment):
        raise DeploymentError("Refusing to send credentials outside the Fabric v1 API.")
    return url


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise DeploymentError("Unexpected API redirect; request was not forwarded.")


class FabricClient:
    def __init__(self, tenant_id):
        self.tenant_id = tenant_id
        self.opener = urllib.request.build_opener(NoRedirect())

    def request(self, method, path, body=None):
        url = safe_url(path)
        token = azure_json(
            "account", "get-access-token", "--tenant", self.tenant_id,
            "--resource", "https://api.fabric.microsoft.com",
        )["accessToken"]
        request = urllib.request.Request(
            url, method=method,
            data=None if body is None else json.dumps(body).encode("utf-8"),
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        )
        try:
            with self.opener.open(request, timeout=120) as response:
                data = response.read()
                return response.status, response.headers, json.loads(data) if data else {}
        except urllib.error.HTTPError as error:
            # Do not log request headers, tokens or arbitrary server response bodies.
            raise DeploymentError(
                f"Fabric {method} {urllib.parse.urlsplit(url).path}: HTTP {error.code}; "
                f"request ID: {error.headers.get('request-id', 'unavailable')}. "
                "Check tenant, permissions, capacity, and service availability."
            ) from None

    def get(self, path):
        return self.request("GET", path)[2]

    def list(self, path):
        result, visited = [], set()
        while path:
            url = safe_url(path)
            if url in visited:
                raise DeploymentError("Repeated Fabric pagination URL.")
            visited.add(url)
            page = self.get(path)
            result.extend(page["value"])
            path = page.get("continuationUri")
            if page.get("continuationToken") and not path:
                raise DeploymentError("Pagination token without continuationUri; refusing incomplete discovery.")
        return result

    def create(self, path, body):
        return self.post_operation(path, body, success_codes=(201,))

    def definition(self, workspace_id, notebook_id):
        result = self.post_operation(
            f"/workspaces/{workspace_id}/notebooks/{notebook_id}/getDefinition?format=fabricGitSource"
        )
        return result["definition"]

    def post_operation(self, path, body=None, success_codes=(200, 201)):
        status, headers, data = self.request("POST", path, body)
        if status in success_codes:
            return data
        if status != 202 or not (headers.get("Location") or headers.get("x-ms-operation-id")):
            raise DeploymentError("Unexpected operation response; inspect target before retrying.")
        operation_id = headers.get("x-ms-operation-id")
        # Fabric can return a regional Location; poll its ID via the public API instead.
        operation = (f"{API}/operations/{guid(operation_id)}" if operation_id
                     else safe_url(headers["Location"]))
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            time.sleep(min(max(int(headers.get("Retry-After", "5")), 1), 60))
            _, headers, state = self.request("GET", operation)
            if state.get("status") == "Succeeded":
                return self.get(operation.rstrip("/") + "/result")
            if state.get("status") in {"Failed", "Canceled", "Cancelled"}:
                raise DeploymentError("Fabric operation failed; inspect operation and target before retrying.")
        raise DeploymentError("Operation timed out; operation may still finish. Inspect target before replanning.")


def guid(value):
    try:
        if not isinstance(value, str) or str(UUID(value)) != value:
            raise ValueError()
    except (ValueError, AttributeError):
        raise DeploymentError("IDs must be canonical lowercase UUID strings.") from None
    return value


def load_config(path):
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {"tenant_id", "capacity_id", "workspace_name", "lakehouses"}
    if not required <= set(config) or set(config) - required - {"workspace_id", "notebook_parameters"}:
        raise DeploymentError("Configuration requires tenant_id, capacity_id, workspace_name, lakehouses; optional workspace_id and notebook_parameters.")
    for key in ["tenant_id", "capacity_id"]:
        guid(config[key])
    name = config["workspace_name"]
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,99}", name):
        raise DeploymentError("Use a workspace name of 1-100 letters, digits, underscores or hyphens.")
    if name.casefold() in {"brp", "brp2", "my_workspace", "admin_monitoring"}:
        raise DeploymentError("Protected workspace name.")
    if "workspace_id" in config:
        guid(config["workspace_id"])
        if config["workspace_id"] == PROTECTED_WORKSPACE:
            raise DeploymentError("The original BRP2 workspace is protected.")
    lakes = config["lakehouses"]
    if not isinstance(lakes, dict) or set(lakes) != {"bronze", "silver", "gold"}:
        raise DeploymentError("Specify exactly bronze, silver and gold lakehouse names.")
    if any(not isinstance(n, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,99}", n) for n in lakes.values()):
        raise DeploymentError("Lakehouse names must use letters, digits and underscores, starting with a letter.")
    if len({n.casefold() for n in lakes.values()}) != 3:
        raise DeploymentError("Use three distinct lakehouse names.")
    if "notebook_parameters" in config:
        validate_notebook_parameters(config["notebook_parameters"])
    return config


def validate_notebook_parameters(parameters):
    optional = {
        "snapshot_date", "source_directory", "silver_directory", "results_directory",
        "errors_directory", "silver_schema", "reporting_schema", "experiment_name", "model_name",
    }
    if (not isinstance(parameters, dict) or "run_prefix" not in parameters
            or set(parameters) - optional - {"run_prefix"}):
        raise DeploymentError("notebook_parameters requires run_prefix and only documented parameter keys.")
    if not isinstance(parameters["run_prefix"], str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,60}", parameters["run_prefix"]):
        raise DeploymentError("run_prefix must contain 1-60 letters, digits, underscores or hyphens.")
    for key, value in parameters.items():
        if not isinstance(value, str):
            raise DeploymentError(f"Notebook parameter must be a string: {key}.")
        if key.endswith("_directory") and not re.fullmatch(r"[A-Za-z0-9_-]+(?:/[A-Za-z0-9_-]+)*", value):
            raise DeploymentError(f"Unsafe notebook directory: {key}.")
        if key.endswith("_schema") and not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,99}", value):
            raise DeploymentError(f"Invalid notebook schema: {key}.")
        if key in {"experiment_name", "model_name"} and not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", value):
            raise DeploymentError(f"Invalid ML name: {key}.")
        if key == "snapshot_date":
            try:
                if date.fromisoformat(value).isoformat() != value:
                    raise ValueError()
            except ValueError:
                raise DeploymentError("snapshot_date must be a valid YYYY-MM-DD date.") from None
    results = parameters.get("results_directory", "POC_vehicle_usage")
    errors = parameters.get("errors_directory", "POC_vehicle_usage_errors")
    if results == errors or results.startswith(errors + "/") or errors.startswith(results + "/"):
        raise DeploymentError("Results and error directories must not overlap.")


def unique_named(items, name):
    matches = [item for item in items if item["displayName"].casefold() == name.casefold()]
    if len(matches) > 1:
        raise DeploymentError(f"Ambiguous name: {name}.")
    return matches[0] if matches else None


def build_plan(client, config):
    if client.tenant_id != config["tenant_id"]:
        raise DeploymentError("Authenticated tenant differs from configuration.")
    capacity = next((c for c in client.list("/capacities") if c["id"] == config["capacity_id"]), None)
    if not capacity or capacity.get("state") != "Active":
        raise DeploymentError("Configured capacity is not visible and active.")
    if not str(capacity.get("sku", "")).startswith("F"):
        raise DeploymentError("This bootstrap requires an F or Fabric trial capacity, not PPU.")
    workspace = unique_named(client.list("/workspaces"), config["workspace_name"])
    if workspace and workspace["id"] == PROTECTED_WORKSPACE:
        raise DeploymentError("The original BRP2 workspace is protected.")
    if workspace:
        if config.get("workspace_id") != workspace["id"]:
            raise DeploymentError("Workspace already exists. Review it and explicitly configure workspace_id to reuse it.")
        actual = client.get(f"/workspaces/{workspace['id']}")
        if actual.get("capacityId") != config["capacity_id"]:
            raise DeploymentError("Existing workspace is not on the configured capacity; automatic reassignment is disabled.")
    elif config.get("workspace_id"):
        raise DeploymentError("Configured workspace ID/name is not visible; no workspace will be created.")
    items = client.list(f"/workspaces/{workspace['id']}/items") if workspace else []
    lakes = {}
    for role, name in sorted(config["lakehouses"].items()):
        item = unique_named([i for i in items if i.get("type") != "SQLEndpoint"], name)
        if not item and any(i["displayName"].casefold() == name.casefold() for i in items):
            raise DeploymentError(f"A SQL endpoint exists without a matching lakehouse: {name}.")
        if item:
            if item.get("type") != "Lakehouse":
                raise DeploymentError(f"Name is occupied by a non-lakehouse item: {name}.")
            details = client.get(f"/workspaces/{workspace['id']}/lakehouses/{item['id']}")
            if not details.get("properties", {}).get("defaultSchema"):
                raise DeploymentError(f"Cannot verify schema support for {name}; refusing reuse.")
        lakes[role] = {"name": name, "id": item["id"] if item else None, "action": "reuse" if item else "create"}
    plan = {
        "version": 1, "scope": "workspace-and-lakehouse-bootstrap-only",
        "config": config,
        "capacity": {"id": capacity["id"], "name": capacity["displayName"], "sku": capacity["sku"]},
        "workspace": {"name": config["workspace_name"], "id": workspace["id"] if workspace else None,
                      "action": "reuse" if workspace else "create"},
        "lakehouses": lakes,
        "limitations": [
            "Read-only discovery does not prove permission to create or assign resources.",
            "No data copy, notebook import/execution, ML, Power BI or Data Agent deployment.",
            "No overwrites, deletes, capacity provisioning/resume or reassignment.",
        ],
    }
    return {**plan, "approval_id": digest(plan)}


def notebook_metadata(source):
    match = re.search(r"(?m)^(?:# META [^\n]*(?:\n|$))+", source)
    if not match:
        raise DeploymentError("Notebook metadata block is missing.")
    metadata = json.loads("\n".join(line[len("# META "):] for line in match[0].splitlines()))
    return match, metadata


def notebook_parameters(source):
    if source.count(PARAMETER_MARKER) != 1:
        raise DeploymentError("Expected exactly one notebook parameter cell.")
    start = source.index(PARAMETER_MARKER) + len(PARAMETER_MARKER)
    end = source.index(METADATA_MARKER, start)
    parameters = {}
    for node in ast.parse(source[start:end]).body:
        if (not isinstance(node, ast.Assign) or len(node.targets) != 1
                or not isinstance(node.targets[0], ast.Name)):
            raise DeploymentError("Only literal parameter assignments are supported.")
        key = node.targets[0].id
        if key in parameters:
            raise DeploymentError(f"Duplicate notebook parameter: {key}.")
        parameters[key] = ast.literal_eval(node.value)
    return start, end, parameters


def render_notebook(name, config, resources):
    source = (ROOT / f"{name}.Notebook" / "notebook-content.py").read_text(encoding="utf-8-sig")
    source_sha = hashlib.sha256(source.encode()).hexdigest()
    options = config["notebook_parameters"]
    validate_notebook_parameters(options)
    prefix = options["run_prefix"]
    workspace_id = resources["workspace"]["id"]
    lakes = resources["lakehouses"]
    attachments = NOTEBOOKS[name]
    default = lakes[attachments[0]]
    metadata_match, metadata = notebook_metadata(source)
    metadata["dependencies"]["lakehouse"] = {
        "default_lakehouse": default["id"], "default_lakehouse_name": default["name"],
        "default_lakehouse_workspace_id": workspace_id,
        "known_lakehouses": [{"id": lakes[role]["id"]} for role in attachments],
    }
    metadata_text = "".join("# META " + line + "\n" for line in json.dumps(metadata, indent=2).splitlines())
    source = source[:metadata_match.start()] + metadata_text + source[metadata_match.end():]
    start, end, parameters = notebook_parameters(source)
    replacements = {
        **{key: value for key, value in options.items() if key != "run_prefix"},
        "medallion_workspace_id": workspace_id,
        "bronze_lakehouse_id": lakes["bronze"]["id"],
        "silver_lakehouse_id": lakes["silver"]["id"],
        "gold_lakehouse_id": lakes["gold"]["id"],
        "silver_lakehouse_name": lakes["silver"]["name"],
        "gold_lakehouse_name": lakes["gold"]["name"],
        "silver_run_id": f"silver_{prefix}", "demo_run_id": f"train_{prefix}",
        "report_source_run_id": f"train_{prefix}", "report_run_id": f"report_{prefix}",
    }
    parameters.update({key: value for key, value in replacements.items() if key in parameters})
    cell = "\n\n" + "\n".join(f"{key} = {value!r}" for key, value in parameters.items()) + "\n\n"
    source = source[:start] + cell + source[end:]
    compile(source, name, "exec")
    if PROTECTED_WORKSPACE in source:
        raise DeploymentError(f"Source workspace reference remains in {name}.")
    return source, {
        "name": name, "source_sha256": source_sha,
        "definition_sha256": digest(notebook_signature(source)),
        "default_lakehouse": default["name"], "parameters": parameters,
    }


def notebook_signature(source):
    """Compare executable cells, magic, markdown, kernel and lakehouse bindings."""
    _, metadata = notebook_metadata(source)
    lake = dict(metadata["dependencies"]["lakehouse"])
    workspace_id = lake["default_lakehouse_workspace_id"]
    lake["known_lakehouses"] = sorted([
        {"id": item["id"], "workspaceId": item.get("workspaceId", workspace_id)}
        for item in lake.get("known_lakehouses", [])
    ], key=lambda item: (item["id"], item["workspaceId"]))
    pieces = re.split(r"(?m)^# (CELL|PARAMETERS CELL|MARKDOWN) \*{20}[ \t]*$", source)
    cells = []
    for kind, body in zip(pieces[1::2], pieces[2::2]):
        code = body.split(METADATA_MARKER, 1)[0]
        if kind == "MARKDOWN":
            content = [line.rstrip() for line in code.splitlines() if line.strip()]
        else:
            content = {
                "ast": ast.dump(ast.parse(code), include_attributes=False),
                "magic": [line.rstrip() for line in code.splitlines() if line.startswith("# MAGIC ")],
            }
        cells.append({"kind": kind, "content": content})
    if not cells:
        raise DeploymentError("Notebook has no cells.")
    return {
        "kernel": metadata["kernel_info"], "lakehouse": lake, "cells": cells,
        "ast": ast.dump(ast.parse(source), include_attributes=False),
    }


def definition_source(definition):
    parts = [part for part in definition["parts"] if part["path"] == "notebook-content.py"]
    if len(parts) != 1 or parts[0]["payloadType"] != "InlineBase64":
        raise DeploymentError("Expected one inline Python FabricGitSource notebook part.")
    return base64.b64decode(parts[0]["payload"], validate=True).decode("utf-8-sig").replace("\r\n", "\n")


def build_notebook_plan(client, config):
    resources = build_plan(client, config)
    if resources["workspace"]["action"] != "reuse" or any(
        lake["action"] != "reuse" for lake in resources["lakehouses"].values()
    ):
        raise DeploymentError("Bootstrap the workspace and all lakehouses before importing notebooks.")
    if "notebook_parameters" not in config:
        raise DeploymentError("Notebook import requires notebook_parameters with a fresh run_prefix.")
    workspace_id = resources["workspace"]["id"]
    items = client.list(f"/workspaces/{workspace_id}/items")
    notebooks = []
    for name in NOTEBOOKS:
        _, planned = render_notebook(name, config, resources)
        item = unique_named(items, name)
        if item:
            if item.get("type") != "Notebook":
                raise DeploymentError(f"Notebook name is occupied by a different item type: {name}.")
            actual = definition_source(client.definition(workspace_id, item["id"]))
            if digest(notebook_signature(actual)) != planned["definition_sha256"]:
                raise DeploymentError(f"Existing notebook differs: {name}. Automatic overwrite is disabled.")
        notebooks.append({**planned, "id": item["id"] if item else None,
                          "action": "reuse" if item else "create"})
    resources.pop("approval_id")
    resources.update(
        scope="notebook-import-only", notebooks=notebooks,
        limitations=[
            "Imports definitions only; no notebook execution, data copy or ML training.",
            "No semantic model, report or Data Agent deployment.",
            "Existing different notebooks are rejected, never overwritten.",
            "Definition discovery uses the read-only POST getDefinition API.",
        ],
    )
    return {**resources, "approval_id": digest(resources)}


def build_stage_plan(client, config, stage):
    if stage == "notebooks":
        return build_notebook_plan(client, config)
    if stage != "bootstrap":
        raise DeploymentError("Unknown deployment stage.")
    return build_plan(client, config)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def apply_plan(client, config, approved, approval_id, journal_path, stage="bootstrap"):
    if approved.get("approval_id") != approval_id or build_stage_plan(client, config, stage) != approved:
        raise DeploymentError("Approval or target state changed. Generate and approve a fresh plan.")
    if journal_path.exists():
        raise DeploymentError("Journal already exists. Use a fresh journal path; previous evidence will not be overwritten.")
    journal = {"started_at": datetime.now(timezone.utc).isoformat(),
               "approval_id": approval_id, "stage": stage, "status": "in_progress",
               "workspace_id": None, "lakehouses": {}}
    sources = {}
    for notebook in approved.get("notebooks", []):
        source, details = render_notebook(notebook["name"], config, approved)
        if any(details[key] != notebook[key] for key in details):
            raise DeploymentError("Notebook source changed after planning. Generate a fresh plan.")
        sources[notebook["name"]] = source
    write_json(journal_path, journal)
    try:
        workspace = approved["workspace"]
        if workspace["action"] == "create":
            journal["pending"] = "create_workspace"
            write_json(journal_path, journal)
            created = client.create("/workspaces", {
                "displayName": workspace["name"], "capacityId": config["capacity_id"],
                "description": "BRP isolated prompt deployment test",
            })
            workspace_id = guid(created["id"])
        else:
            workspace_id = workspace["id"]
        journal["workspace_id"] = workspace_id
        write_json(journal_path, journal)
        if workspace_id == PROTECTED_WORKSPACE:
            raise DeploymentError("Refusing writes in the protected BRP2 workspace.")
        if client.get(f"/workspaces/{workspace_id}").get("capacityId") != config["capacity_id"]:
            raise DeploymentError("Workspace capacity assignment not confirmed; inspect before retrying.")
        for role, lake in approved["lakehouses"].items():
            journal["pending"] = f"{lake['action']}_{role}"
            write_json(journal_path, journal)
            if lake["action"] == "create":
                created = client.create(f"/workspaces/{workspace_id}/lakehouses", {
                    "displayName": lake["name"], "creationPayload": {"enableSchemas": True},
                })
                lake_id = guid(created["id"])
            else:
                lake_id = lake["id"]
            journal["lakehouses"][role] = {"id": lake_id, "name": lake["name"]}
            write_json(journal_path, journal)
            actual = client.get(f"/workspaces/{workspace_id}/lakehouses/{lake_id}")
            if (actual.get("displayName") != lake["name"]
                    or not actual.get("properties", {}).get("defaultSchema")):
                raise DeploymentError(f"Lakehouse readback/schema verification failed: {lake['name']}.")
        if stage == "notebooks":
            journal["notebooks"] = {}
        for notebook in approved.get("notebooks", []):
            name = notebook["name"]
            journal["pending"] = f"{notebook['action']}_{name}"
            write_json(journal_path, journal)
            if notebook["action"] == "create":
                created = client.create(f"/workspaces/{workspace_id}/notebooks", {
                    "displayName": name,
                    "description": "BRP parameterized notebook; imported without execution.",
                    "definition": {"format": "fabricGitSource", "parts": [{
                        "path": "notebook-content.py", "payloadType": "InlineBase64",
                        "payload": base64.b64encode(sources[name].encode()).decode("ascii"),
                    }]},
                })
                notebook_id = guid(created["id"])
            else:
                notebook_id = notebook["id"]
            journal["notebooks"][name] = {
                "id": notebook_id, "definition_sha256": notebook["definition_sha256"], "verified": False,
            }
            write_json(journal_path, journal)
            actual = definition_source(client.definition(workspace_id, notebook_id))
            if digest(notebook_signature(actual)) != notebook["definition_sha256"]:
                raise DeploymentError(f"Notebook definition readback differs: {name}.")
            journal["notebooks"][name]["verified"] = True
            write_json(journal_path, journal)
        journal.pop("pending", None)
        journal["status"] = "succeeded"
        write_json(journal_path, journal)
    except (DeploymentError, OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as error:
        journal["status"] = "failed"
        journal["error_type"] = type(error).__name__
        write_json(journal_path, journal)
        raise
    return journal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["plan", "apply"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=["bootstrap", "notebooks"], default="bootstrap")
    parser.add_argument("--plan", type=Path, required=True, help="Local plan JSON output/input.")
    parser.add_argument("--approve", help="Exact approval_id from the reviewed plan; apply only.")
    parser.add_argument("--journal", type=Path, help="New local journal path; required for apply.")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        if args.config.resolve() == args.plan.resolve():
            raise DeploymentError("Configuration and plan must have different paths.")
        account = azure_json("account", "show")
        if account["tenantId"] != config["tenant_id"]:
            raise DeploymentError("Azure CLI is logged into a different tenant. Run az login --tenant TARGET.")
        client = FabricClient(account["tenantId"])
        if args.command == "plan":
            if args.approve or args.journal:
                raise DeploymentError("Approval and journal options are apply-only.")
            plan = build_stage_plan(client, config, args.stage)
            write_json(args.plan, plan)
            print(json.dumps(plan, indent=2))
        else:
            if not args.approve or not args.journal:
                raise DeploymentError("Apply requires --approve and --journal.")
            if len({args.config.resolve(), args.plan.resolve(), args.journal.resolve()}) != 3:
                raise DeploymentError("Configuration, plan and journal must have different paths.")
            result = apply_plan(client, config, json.loads(args.plan.read_text()), args.approve, args.journal, args.stage)
            print(json.dumps(result, indent=2))
    except (DeploymentError, OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as error:
        print(f"Deployment failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
