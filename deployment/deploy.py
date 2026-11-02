"""Plan or bootstrap an isolated Fabric workspace. No notebook execution."""

import argparse
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
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID


API = "https://api.fabric.microsoft.com/v1"
PROTECTED_WORKSPACE = "f78fd47f-e385-4179-b31a-5e376322c392"


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
        status, headers, data = self.request("POST", path, body)
        if status == 201:
            return data
        if status != 202 or not headers.get("Location"):
            raise DeploymentError("Unexpected creation response; inspect target before retrying.")
        operation = safe_url(headers["Location"])
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            time.sleep(min(max(int(headers.get("Retry-After", "5")), 1), 60))
            _, headers, state = self.request("GET", operation)
            if state.get("status") == "Succeeded":
                return self.get(operation.rstrip("/") + "/result")
            if state.get("status") in {"Failed", "Canceled", "Cancelled"}:
                raise DeploymentError("Fabric creation operation failed; inspect operation and target before retrying.")
        raise DeploymentError("Creation timed out; operation may still finish. Inspect target before replanning.")


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
    if set(config) not in (required, required | {"workspace_id"}):
        raise DeploymentError("Configuration keys must be tenant_id, capacity_id, workspace_name, lakehouses, optional workspace_id.")
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
    return config


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
        item = unique_named(items, name)
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


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def apply_plan(client, config, approved, approval_id, journal_path):
    if approved.get("approval_id") != approval_id or build_plan(client, config) != approved:
        raise DeploymentError("Approval or target state changed. Generate and approve a fresh plan.")
    if journal_path.exists():
        raise DeploymentError("Journal already exists. Use a fresh journal path; previous evidence will not be overwritten.")
    journal = {"started_at": datetime.now(timezone.utc).isoformat(),
               "approval_id": approval_id, "status": "in_progress", "workspace_id": None, "lakehouses": {}}
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
            plan = build_plan(client, config)
            write_json(args.plan, plan)
            print(json.dumps(plan, indent=2))
        else:
            if not args.approve or not args.journal:
                raise DeploymentError("Apply requires --approve and --journal.")
            if len({args.config.resolve(), args.plan.resolve(), args.journal.resolve()}) != 3:
                raise DeploymentError("Configuration, plan and journal must have different paths.")
            result = apply_plan(client, config, json.loads(args.plan.read_text()), args.approve, args.journal)
            print(json.dumps(result, indent=2))
    except (DeploymentError, OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as error:
        print(f"Deployment failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
