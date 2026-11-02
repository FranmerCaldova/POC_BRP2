"""Deploy the checked-in Direct Lake model and PBIR report to the target."""

import argparse
import base64
import json
import math
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import deploy


MODEL = "POC_SM_VEHICLE_USAGE"
REPORT = "POC_RPT_VEHICLE_USAGE"
TABLES = {
    "Predictions": "POC_predictions", "Evaluation": "POC_evaluation", "Segment": "POC_segments",
    "Calendar": "POC_calendar", "Publication": "POC_publication", "Quality": "POC_quality",
}


class PowerBIClient:
    def __init__(self, tenant_id):
        self.tenant_id = tenant_id
        self.opener = urllib.request.build_opener(deploy.NoRedirect())

    def request(self, method, path, body=None):
        if not path.startswith("/groups/") or urllib.parse.urlsplit(path).netloc:
            raise deploy.DeploymentError("Invalid Power BI API path.")
        token = deploy.azure_json(
            "account", "get-access-token", "--tenant", self.tenant_id,
            "--resource", "https://analysis.windows.net/powerbi/api",
        )["accessToken"]
        request = urllib.request.Request(
            "https://api.powerbi.com/v1.0/myorg" + path, method=method,
            data=None if body is None else json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        )
        try:
            with self.opener.open(request, timeout=120) as response:
                body = response.read(4 * 1024 * 1024 + 1)
                if len(body) > 4 * 1024 * 1024:
                    raise deploy.DeploymentError("Power BI validation response exceeds supported size.")
                return response.status, response.headers, json.loads(body) if body else {}
        except urllib.error.HTTPError as error:
            raise deploy.DeploymentError(
                f"Power BI {method} {path}: HTTP {error.code}. "
                "Check Service connection/credentials and permissions; do not retry credential errors."
            ) from None


def validation_query():
    fields = [f'"{gold}", COUNTROWS(\'{table}\')' for table, gold in TABLES.items()]
    fields += [
        '"mae_km", [Model MAE]', '"baseline_mae_km", [Baseline MAE]',
        '"rmse_km", [RMSE]', '"improvement", [MAE Improvement]',
    ]
    # Match the existing measures' exclusion of the relationship-generated blank member.
    for label, column in [("model_version", "Model Version"), ("ml_run", "ML Run"), ("reporting_run", "Reporting Run")]:
        fields.append(
            f'"{label}", CALCULATE(SELECTEDVALUE(\'Publication\'[{column}]), '
            'KEEPFILTERS(NOT ISBLANK(\'Publication\'[Publication Key])))'
        )
    return "EVALUATE ROW(" + ", ".join(fields) + ")"


def compare_data(result, evidence):
    results = result.get("results", [])
    if "error" in result or len(results) != 1 or "error" in results[0]:
        raise deploy.DeploymentError("Power BI query failed or returned unexpected results.")
    tables = results[0].get("tables", [])
    if len(tables) != 1 or "error" in tables[0] or len(tables[0].get("rows", [])) != 1:
        raise deploy.DeploymentError("Power BI validation requires exactly one result row.")
    row = {key.removeprefix("[").removesuffix("]"): value for key, value in tables[0]["rows"][0].items()}
    train, publication = evidence["train"], evidence["publish"]
    expected = {
        **publication["table_rows"], "model_version": train["model_version"],
        "ml_run": train["demo_run_id"], "reporting_run": publication["report_run_id"],
    }
    if set(publication["table_rows"]) != set(TABLES.values()):
        raise deploy.DeploymentError("Processing evidence does not contain all six Gold tables.")
    for key, value in expected.items():
        if row.get(key) != value:
            raise deploy.DeploymentError(f"Live Power BI data mismatch: {key}.")
    metrics = {key: train["metrics"][key] for key in ("mae_km", "baseline_mae_km", "rmse_km")}
    metrics["improvement"] = train["mae_improvement_fraction"]
    for key, value in metrics.items():
        actual = row.get(key)
        if (type(actual) not in (float, int) or not math.isfinite(actual)
                or not math.isclose(actual, value, rel_tol=1e-8, abs_tol=1e-7)):
            raise deploy.DeploymentError(f"Live Power BI metric mismatch: {key}.")
    return row


def validate_live(client, powerbi, config, processing, reporting, journal_path, *, refresh=False, timeout=900):
    workspace_id = config["workspace_id"]
    if any(j.get("status") != "succeeded" or j.get("workspace_id") != workspace_id for j in (processing, reporting)):
        raise deploy.DeploymentError("Successful processing/reporting journals for this workspace are required.")
    plan = build_plan(client, config)
    if any(plan[role]["action"] != "reuse" or plan[role]["id"] != reporting[role]["id"] for role in ("model", "report")):
        raise deploy.DeploymentError("Live reporting IDs differ from the verified deployment journal.")
    if journal_path.exists():
        raise deploy.DeploymentError("Validation journal already exists; choose a fresh path.")
    model_id, report_id = reporting["model"]["id"], reporting["report"]["id"]
    base = f"/groups/{workspace_id}/datasets/{model_id}"
    journal = {"status": "in_progress", "workspace_id": workspace_id, "model_id": model_id,
               "report_id": report_id, "visual_validation": "not_performed"}
    deploy.write_json(journal_path, journal)
    try:
        if refresh:
            journal["pending"] = "refresh_submission"
            deploy.write_json(journal_path, journal)
            status, headers, _ = powerbi.request("POST", base + "/refreshes", {
                "type": "full", "commitMode": "transactional", "retryCount": 0,
            })
            if status != 202:
                raise deploy.DeploymentError("Refresh not accepted; inspect before retrying.")
            location = headers.get("Location", "")
            refresh_id = deploy.guid(urllib.parse.urlsplit(location).path.rstrip("/").rsplit("/", 1)[-1])
            journal["refresh_id"] = refresh_id
            journal["pending"] = "refresh_completion"
            deploy.write_json(journal_path, journal)
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                _, _, result = powerbi.request("GET", base + "/refreshes/" + refresh_id)
                journal["refresh"] = result
                deploy.write_json(journal_path, journal)
                state = result.get("status")
                if state == "Completed":
                    break
                if state not in {"Unknown", "NotStarted", "InProgress"}:
                    raise deploy.DeploymentError(
                        f"Refresh {state}; inspect validation journal. For credential errors, "
                        "configure the connection in Power BI Service; do not retry."
                    )
                time.sleep(10)
            else:
                raise deploy.DeploymentError("Refresh timed out and may still run; do not retry blindly.")
        journal["pending"] = "live_data_validation"
        _, _, report = powerbi.request("GET", f"/groups/{workspace_id}/reports/{report_id}")
        if report.get("datasetId") != model_id:
            raise deploy.DeploymentError("Live report is not bound to the target semantic model.")
        _, _, result = powerbi.request("POST", base + "/executeQueries", {
            "queries": [{"query": validation_query()}], "serializerSettings": {"includeNulls": True},
        })
        journal["query_result"] = result
        deploy.write_json(journal_path, journal)
        journal["verified_data"] = compare_data(result, processing["evidence"])
        journal.update(status="succeeded", report_url=report.get("webUrl"))
        journal.pop("pending", None)
        deploy.write_json(journal_path, journal)
    except (deploy.DeploymentError, OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as error:
        journal.update(status="failed", error_type=type(error).__name__)
        deploy.write_json(journal_path, journal)
        raise
    return journal


def decode_parts(definition):
    parts = {}
    for part in definition["parts"]:
        if part["path"] == ".platform":
            continue
        if part["payloadType"] != "InlineBase64" or part["path"] in parts:
            raise deploy.DeploymentError("Invalid or duplicate definition part.")
        parts[part["path"]] = base64.b64decode(part["payload"], validate=True)
    return parts


def encode_parts(parts, format_name):
    return {"format": format_name, "parts": [
        {"path": path, "payloadType": "InlineBase64", "payload": base64.b64encode(value).decode("ascii")}
        for path, value in sorted(parts.items())
    ]}


def local_parts(name, item_type):
    folder = deploy.ROOT / f"{name}.{item_type}"
    return {path.relative_to(folder).as_posix(): path.read_bytes()
            for path in folder.rglob("*") if path.is_file() and not any(p.startswith(".") for p in path.relative_to(folder).parts)}


def render_model(resources):
    parts = local_parts(MODEL, "SemanticModel")
    key = "definition/expressions.tmdl"
    text = parts[key].decode("utf-8-sig")
    target = (f"https://onelake.dfs.fabric.microsoft.com/{resources['workspace']['id']}/"
              f"{resources['lakehouses']['gold']['id']}")
    text, count = re.subn(
        r'https://onelake\.dfs\.fabric\.microsoft\.com/[a-f0-9-]{36}/[a-f0-9-]{36}',
        target, text,
    )
    if count != 1:
        raise deploy.DeploymentError("Expected one shared Direct Lake connection URL.")
    parts[key] = text.encode("utf-8")
    schema = resources["config"].get("notebook_parameters", {}).get("reporting_schema", "POC_reporting")
    for path in parts:
        if path.startswith("definition/tables/") and path.endswith(".tmdl"):
            text = parts[path].decode("utf-8-sig").replace("\r\n", "\n")
            text, count = re.subn(r"(?m)^(\s*schemaName: )POC_reporting[ \t]*$", lambda m: m[1] + schema, text)
            if count != 1:
                raise deploy.DeploymentError(f"Expected one reporting schema binding in {path}.")
            parts[path] = text.encode("utf-8")
        if deploy.PROTECTED_WORKSPACE.encode() in parts[path]:
            raise deploy.DeploymentError("Source workspace reference remains in model definition.")
    return parts


def render_report(model_id):
    parts = local_parts(REPORT, "Report")
    properties = json.loads(parts["definition.pbir"])
    properties["datasetReference"] = {"byConnection": {"connectionString": f"semanticmodelid={model_id}"}}
    parts["definition.pbir"] = json.dumps(properties, indent=2).encode()
    return parts


def model_inventory(parts):
    inventory = {}
    for path, content in parts.items():
        if not path.startswith("definition/tables/") or not path.endswith(".tmdl"):
            continue
        text = content.decode("utf-8-sig")
        table_match = re.search(r"(?m)^table (?:'([^']+)'|([^\r\n]+))", text)
        if not table_match:
            raise deploy.DeploymentError(f"Missing table declaration in {path}.")
        name = next(value for value in table_match.groups() if value is not None)
        members = re.findall(r"(?m)^\t(?:column|measure) (?:'([^']+)'|([^\s=]+))", text)
        inventory[name] = {next(value for value in pair if value) for pair in members}
    return inventory


def validate_bindings(report, model):
    inventory = model_inventory(model)
    invalid = set()

    def visit(value):
        if isinstance(value, dict):
            entity = value.get("Entity")
            if isinstance(entity, str) and entity not in inventory:
                invalid.add(entity)
            for kind in ("Column", "Measure"):
                field = value.get(kind)
                if isinstance(field, dict):
                    table = field.get("Expression", {}).get("SourceRef", {}).get("Entity")
                    member = field.get("Property")
                    if table and (table not in inventory or member not in inventory[table]):
                        invalid.add(f"{table}.{member}")
            for key, child in value.items():
                if key in {"queryRef", "nativeQueryRef"} and isinstance(child, str) and "." in child:
                    table, member = child.split(".", 1)
                    if table in inventory and member not in inventory[table]:
                        invalid.add(child)
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    for path, content in report.items():
        if path.endswith(".json") and path.startswith("definition/"):
            visit(json.loads(content))
    if invalid:
        raise deploy.DeploymentError(f"Report/model bindings differ: {sorted(invalid)}.")
    return {name: sorted(values) for name, values in sorted(inventory.items())}


def signature(parts, workspace_name=None):
    normalized = {}
    for path, value in parts.items():
        if path.endswith((".json", ".pbir", ".pbism")):
            normalized[path] = json.loads(value)
            if path == "definition.pbir" and workspace_name:
                connection = normalized[path].get("datasetReference", {}).get("byConnection", {})
                prefix = (f"Data Source=powerbi://api.powerbi.com/v1.0/myorg/{workspace_name};"
                          f"initial catalog={MODEL};integrated security=ClaimsToken;")
                text = connection.get("connectionString", "")
                if text.startswith(prefix):
                    model_id = text[len(prefix):].removeprefix("semanticmodelid=")
                    deploy.guid(model_id)
                    connection["connectionString"] = "semanticmodelid=" + model_id
        elif path.endswith(".tmdl"):
            lines = value.decode("utf-8-sig").replace("\r\n", "\n").splitlines()
            normalized[path] = [
                line.strip() for line in lines if line.strip() and not line.lstrip().startswith(("///", "lineageTag:"))
            ]
        else:
            normalized[path] = base64.b64encode(value).decode()
    return deploy.digest(normalized)


def get_parts(client, workspace_id, item_type, item_id, format_name):
    client.skill = "powerbi-report-management" if item_type == "reports" else "semantic-model-authoring"
    result = client.post_operation(
        f"/workspaces/{workspace_id}/{item_type}/{item_id}/getDefinition?format={format_name}"
    )
    return decode_parts(result["definition"])


def build_plan(client, config):
    resources = deploy.build_plan(client, config)
    if resources["workspace"]["action"] != "reuse" or any(l["action"] != "reuse" for l in resources["lakehouses"].values()):
        raise deploy.DeploymentError("Bootstrap all resources before reporting deployment.")
    workspace_id = resources["workspace"]["id"]
    items = client.list(f"/workspaces/{workspace_id}/items")
    model = deploy.unique_named(items, MODEL)
    report = deploy.unique_named(items, REPORT)
    expected_model = render_model(resources)
    expected_report = render_report(model["id"] if model else "$TARGET_MODEL")
    bindings = validate_bindings(expected_report, expected_model)
    for item, name, kind, endpoint, format_name, expected in [
        (model, MODEL, "SemanticModel", "semanticModels", "TMDL", expected_model),
        (report, REPORT, "Report", "reports", "PBIR", expected_report),
    ]:
        if item:
            if item.get("type") != kind:
                raise deploy.DeploymentError(f"Reporting item type/name collision: {name}.")
            if signature(get_parts(client, workspace_id, endpoint, item["id"], format_name),
                         config["workspace_name"]) != signature(expected):
                raise deploy.DeploymentError(f"Existing reporting definition differs: {name}; overwrite disabled.")
    resources.pop("approval_id")
    resources.update(
        scope="reporting-definitions",
        model={"name": MODEL, "id": model["id"] if model else None, "action": "reuse" if model else "create",
               "definition_sha256": signature(expected_model)},
        report={"name": REPORT, "id": report["id"] if report else None, "action": "reuse" if report else "create",
                "definition_sha256": signature(expected_report)},
        bindings=bindings,
        limitations=["Deploys definitions, not source data or notebook jobs.",
                     "Requires refresh/query validation after Gold publication; rendering requires a browser.",
                     "Does not deploy Data Agent or overwrite differing existing definitions."],
    )
    return {**resources, "approval_id": deploy.digest(resources)}


def apply(client, config, approved, approval_id, journal_path):
    if approved.get("approval_id") != approval_id or build_plan(client, config) != approved:
        raise deploy.DeploymentError("Reporting plan or target changed; replan before applying.")
    if journal_path.exists():
        raise deploy.DeploymentError("Journal already exists; choose a fresh path.")
    workspace_id = approved["workspace"]["id"]
    journal = {"status": "in_progress", "workspace_id": workspace_id, "approval_id": approval_id}
    deploy.write_json(journal_path, journal)
    try:
        for role, endpoint, format_name in [("model", "semanticModels", "TMDL"), ("report", "reports", "PBIR")]:
            item = approved[role]
            parts = render_model(approved) if role == "model" else render_report(journal["model"]["id"])
            journal["pending"] = role
            deploy.write_json(journal_path, journal)
            if item["action"] == "create":
                client.skill = "powerbi-report-management" if role == "report" else "semantic-model-authoring"
                created = client.create(f"/workspaces/{workspace_id}/{endpoint}", {
                    "displayName": item["name"], "definition": encode_parts(parts, format_name),
                })
                item_id = deploy.guid(created["id"])
            else:
                item_id = item["id"]
            journal[role] = {"id": item_id, "verified": False}
            deploy.write_json(journal_path, journal)
            actual = get_parts(client, workspace_id, endpoint, item_id, format_name)
            if signature(actual, config["workspace_name"]) != signature(parts):
                # Persist only definition metadata locally for diagnosis, never credentials/data rows.
                deploy.write_json(journal_path.with_suffix(f".{role}-readback.json"), encode_parts(actual, format_name))
                raise deploy.DeploymentError(f"{role} definition readback differs; inspect before retrying.")
            if role == "model":
                model_parts = actual
            else:
                validate_bindings(actual, model_parts)
            journal[role]["verified"] = True
            deploy.write_json(journal_path, journal)
        journal.pop("pending")
        journal["status"] = "succeeded"
        deploy.write_json(journal_path, journal)
    except (deploy.DeploymentError, OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as error:
        journal.update(status="failed", error_type=type(error).__name__)
        deploy.write_json(journal_path, journal)
        raise
    return journal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["plan", "apply", "validate"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--approve")
    parser.add_argument("--journal", type=Path)
    parser.add_argument("--processing-journal", type=Path)
    parser.add_argument("--reporting-journal", type=Path)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    try:
        config = deploy.load_config(args.config)
        paths = [path.resolve() for path in (
            args.config, args.plan, args.journal, args.processing_journal, args.reporting_journal,
        ) if path]
        if len(set(paths)) != len(paths):
            raise deploy.DeploymentError("Use distinct configuration, plan and journal paths.")
        if deploy.azure_json("account", "show")["tenantId"] != config["tenant_id"]:
            raise deploy.DeploymentError("Azure CLI tenant mismatch.")
        client = deploy.FabricClient(config["tenant_id"], skill="semantic-model-authoring")
        if args.command == "validate":
            if (args.plan or args.approve or not args.journal
                    or not args.processing_journal or not args.reporting_journal):
                raise deploy.DeploymentError("validate requires separate processing, reporting and output journals.")
            result = validate_live(
                client, PowerBIClient(config["tenant_id"]), config,
                json.loads(args.processing_journal.read_text()), json.loads(args.reporting_journal.read_text()),
                args.journal, refresh=args.refresh,
            )
            print(json.dumps(result, indent=2))
        elif not args.plan or args.processing_journal or args.reporting_journal or args.refresh:
            raise deploy.DeploymentError("plan/apply require --plan; processing/reporting journals and refresh are validate-only.")
        elif args.command == "plan":
            if args.journal or args.approve:
                raise deploy.DeploymentError("Approval/journal are apply-only.")
            result = build_plan(client, config)
            deploy.write_json(args.plan, result)
            print(json.dumps(result, indent=2))
        else:
            if not args.journal or not args.approve:
                raise deploy.DeploymentError("apply requires --approve and --journal.")
            result = apply(client, config, json.loads(args.plan.read_text()), args.approve, args.journal)
            print(json.dumps(result, indent=2))
    except (deploy.DeploymentError, OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as error:
        print(f"Reporting deployment failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
