# POC_BRP2

Fabric proof of concept for cumulative vehicle mileage estimation, not failure prediction.

## Two prompt workflows

Open this repository in VS Code with Copilot agent mode:

- `/brp-deploy <target workspace and active capacity>`:
  [deployment prompt](.github/prompts/brp-deploy.prompt.md). Creates the isolated
  workspace, three lakehouses, three notebooks, Direct Lake model and PBIR report
  definitions, without running jobs or copying source data.
- Upload the four CSVs below, then `/brp-process`:
  [processing prompt](.github/prompts/brp-process.prompt.md). Validates inputs,
  runs Silver -> training -> reporting -> Gold, then refreshes and validates
  the target Power BI model/report using the available Power BI tools.

These are agent workflows around deterministic plan/apply scripts, not an
unattended service. Resource creation and notebook processing require reviewed
approval hashes; model refresh requires an explicit request. Power BI connection
credentials and browser sign-in can require user intervention. Definition
deployment, successful processing, live DAX validation and rendered-report
validation are separate milestones; never infer one from another.
Data Agent deployment is not included.

## Step 1: bootstrap

The standard-library Python orchestrator in [deployment/deploy.py](deployment/deploy.py)
implements **workspace/lakehouse bootstrap and notebook import**. Separate
processing and reporting commands are documented below. The default stage is
`bootstrap`. Python 3.10+ and an authenticated Azure CLI are required;
there are no additional Python dependencies for this command.

1. Copy [deployment/target.example.json](deployment/target.example.json) to
   `.deployment/target.json` and enter the actual tenant and active capacity IDs.
2. Authenticate with `az login --tenant <target-tenant-id> --allow-no-subscriptions`.
3. Generate a read-only plan from the repository root:

```powershell
python -B .\deployment\deploy.py plan --config .\.deployment\target.json --plan .\.deployment\plan.json
```

Review the resolved capacity, target names, create/reuse actions and limitations.
The plan uses GET requests only. Being able to list a capacity/workspace does
**not** prove create/assignment permissions. Workspace creators need permission
to create workspaces and assign the selected capacity; existing workspace
access must permit item creation.

Only after explicit approval, copy the exact `approval_id` into:

```powershell
python -B .\deployment\deploy.py apply --config .\.deployment\target.json --plan .\.deployment\plan.json --approve "<approval_id>" --journal .\.deployment\bootstrap-01.json
```

`apply` rediscovers the target and rejects a changed plan before any write.
It creates the workspace on the selected existing capacity and three
schema-enabled lakehouses, then verifies their IDs, names and schema support.
It does not provision/resume capacities, change an existing workspace's capacity,
copy data, import/run notebooks, train models, or deploy Power BI/Data Agent items.
Schema-enabled lakehouse creation currently uses the API's preview
`creationPayload.enableSchemas` option; tenant support must be verified by apply.

The original BRP2 workspace ID and names BRP/BRP2 are protected. A workspace
name collision is rejected unless its exact `workspace_id` is explicitly added
to configuration. Existing lakehouses are reused only if type and schema support
can be verified. No update, overwrite or delete API is implemented. A rerun
recognizes the same-name SQL endpoints automatically created alongside lakehouses.
An endpoint without its matching lakehouse remains a blocking collision. A rerun
can be a no-op after explicitly selecting the existing workspace and approving
a new plan; it is not an all-or-nothing transaction.

The local journal records pending operations and resolved IDs after each write.
If creation times out or fails, inspect the journal AND Fabric: a request may
have succeeded even if its response was lost. There is no automatic rollback,
blind POST retry or deletion. For recovery, add the verified workspace ID to
configuration, generate a fresh plan, review existing resources, then use a new
journal. Partial resources remain visible rather than being silently removed.

Local configuration, plans and journals belong in ignored `.deployment/`, never
in the public repository. Tokens are obtained from Azure CLI at request time,
remain in memory and are not written to these files. Fabric URLs are restricted
to the official HTTPS v1 endpoint, including pagination and operation polling.

For prompt-driven use, the agent should map the user's requested names into
configuration, invoke `plan`, summarize it, and stop for approval before `apply`.
Use the subsequent commands for notebook import, reporting definitions and
processing. A successful bootstrap alone does not validate those stages.

API references:
[Create workspace](https://learn.microsoft.com/en-us/rest/api/fabric/core/workspaces/create-workspace),
[Create lakehouse](https://learn.microsoft.com/en-us/rest/api/fabric/lakehouse/items/create-lakehouse).

## Step 2: import notebooks without execution

After bootstrap, add the resolved `workspace_id` to local configuration.
All three lakehouses must exist with schema support on the selected capacity.
Set `notebook_parameters` as shown in the example, with a fresh `run_prefix`.
The generated run IDs are `silver_<prefix>`, `train_<prefix>` and `report_<prefix>`.
This makes the Silver -> training -> reporting -> publisher handoff explicit.
Import itself does not reserve run folders; use a new prefix for a new execution.

Generate and review a notebook-only plan:

```powershell
python -B .\deployment\deploy.py plan --stage notebooks --config .\.deployment\target.json --plan .\.deployment\notebooks-plan.json
```

After approval of that plan:

```powershell
python -B .\deployment\deploy.py apply --stage notebooks --config .\.deployment\target.json --plan .\.deployment\notebooks-plan.json --approve "<approval_id>" --journal .\.deployment\notebooks-01.json
```

The stage imports exactly the three checked-in notebook sources using
`fabricGitSource` definitions. It renders target-specific copies in memory:
source files and source item identities (`.platform`) are not copied/modified.
Each notebook receives its target workspace/lakehouse IDs, lakehouse names,
default and known lakehouse bindings, and consistent run IDs in the parameter
cell. PySpark and Python 3.12 kernels, configure magic, parameter-cell markers,
feature engineering and model logic are preserved.

Optional overrides in `notebook_parameters` are `snapshot_date`,
`source_directory`, `silver_directory`, `results_directory`, `errors_directory`,
`silver_schema`, `reporting_schema`, `experiment_name`, and `model_name`.
Unspecified values retain checked-in defaults. `run_prefix` is required;
arbitrary Python expressions and arbitrary parameter keys are rejected.
ML is imported with `mode="train"` and `demo_run_id="train_<prefix>"`.
For later reporting execution, override `mode="reporting"` and
`demo_run_id="report_<prefix>"`; its `report_source_run_id` already points
to the training run. No model or experiment is created by importing definitions.

The plan binds approval to source hashes, generated parameters, dependencies and
the current target state. After import, definitions are retrieved from Fabric
and compared by Python AST, cell sequence/type, configure magic, Markdown,
kernel and lakehouse bindings. Harmless quote/newline formatting is ignored.
Existing semantically matching notebooks are reused; a different definition or
item-type/name collision is rejected, never overwritten. Replan after any
source/configuration change. Partial imports are journaled for safe recovery.

Notebook discovery uses the **read-only POST `getDefinition`** API, which can
return an asynchronous operation; bootstrap discovery still uses GET only.
When Fabric returns a regional operation Location, the orchestrator uses the
validated `x-ms-operation-id` to poll the public Fabric API instead; it does not
forward credentials to the regional host.
Neither stage invokes notebook execution/job endpoints. Notebook execution,
source-data transfer, ML training, semantic models, reports and Data Agents are
not implemented by this stage and are not validated merely by successful import.

API reference:
[Notebook definitions](https://learn.microsoft.com/en-us/rest/api/fabric/articles/item-management/definitions/notebook-definition).

## Step 3: deploy Power BI definitions

[deployment/reporting.py](deployment/reporting.py) renders all checked-in TMDL
and PBIR parts in memory. It remaps the shared OneLake URL, configured Gold
schema and report's `byConnection` semantic-model ID without modifying source
files or visuals. It does not refresh data or run notebooks.

```powershell
python -B .\deployment\reporting.py plan --config .\.deployment\target.json --plan .\.deployment\reporting-plan.json
python -B .\deployment\reporting.py apply --config .\.deployment\target.json --plan .\.deployment\reporting-plan.json --approve "<approval_id>" --journal .\.deployment\reporting-01.json
```

The model is created first, then the report binds to its resolved ID.
Definitions are retrieved after creation; model tables/columns/measures are
checked against report field bindings. Matching definitions are reused;
differing items and name/type collisions block deployment. Server readback
differences are preserved locally beside the journal for diagnosis, not
silently accepted. No overwrite/delete is implemented.

An empty deployment is metadata-only: no report data can appear until Gold
tables exist and the model has been successfully refreshed/framed. Do not
introduce placeholder data to bypass a metadata-deployment error.

## Step 4: upload CSVs and process

Upload these files into Bronze
`Files/FAIL_RATES_PREDICTION_SANITIZED_DATASETS/`
(or the configured `source_directory`), not the lakehouse root:

- `SANITIZED_RAW_CLAIMS.csv`
- `SANITIZED_RAW_VEHICLE_DATA.csv`
- `SANITIZED_RAW_MODEL_FAMILY.csv`
- `SANITIZED_RAW_FLEET_OBSERVATIONS.csv`

**No Bronze Delta tables are required.** Silver reads the fleet-observation
and model-family CSVs directly. The other two files are present for the wider
scenario but are not consumed by this mileage feature-preparation notebook.

Set a fresh `notebook_parameters.run_prefix` and the intended `snapshot_date`.
[deployment/process.py](deployment/process.py) passes all parameters at runtime:
changing the prefix does not require updating notebook definitions. It still
checks the algorithm, cell order, kernels, target attachments and fixed target
IDs. Stored defaults and runtime parameters are both bound into plan approval.

```powershell
python -B .\deployment\process.py plan --config .\.deployment\target.json --plan .\.deployment\processing-plan.json
python -B .\deployment\process.py run --config .\.deployment\target.json --plan .\.deployment\processing-plan.json --approve "<approval_id>" --journal .\.deployment\processing-01.json
```

Preflight checks nonempty files, required CSV headers, source sizes/ETags,
unused run folders and existing target tables. It rechecks immediately before
execution; it does not freeze uploaded files or validate every row. Do not
modify input files while processing. Full business/data validation remains in
the notebooks.

The four jobs run strictly in order:

1. Silver preparation (Spark), including managed Silver Delta and exported data.
2. ML training (Jupyter, two cores), evaluation, registration and reload verification.
3. ML reporting (Jupyter, two cores), reusing the exact registered model version.
4. Gold publication (Spark), creating six reporting Delta tables.

Each job must complete and provide matching workspace/run/model evidence before
the next starts. Existing Silver/Gold tables are blocked by default. To replace
them, explicitly approve that operation, add
`"processing": {"allow_table_overwrite": true}` to local configuration, then
generate/review a fresh plan naming the affected tables. Output folders are
never replayed even with that option. A fresh prefix is always required.

Training incurs compute and creates an experiment/model/version. Table
publication is sequential, **not atomic**: a failure can leave partial writes.
The journal preserves accepted job handles and evidence; timeout does not cancel
the job. Inspect its actual status before recovery, never blindly resubmit.
Use new plan/journal paths for subsequent attempts.

Finally follow `/brp-process` to refresh/frame the target Direct Lake model,
compare live DAX counts/metrics/lineage with the journal, verify the report's
model binding and inspect all report pages. Credential failures require
connection configuration in Power BI Service; no automatic credential workaround
is implemented. A processing journal marked `succeeded` covers notebook
processing only, not Power BI refresh or visual validation.

When modeling tools are unavailable, the standard-library REST fallback is:

```powershell
python -B .\deployment\reporting.py validate --config .\.deployment\target.json --processing-journal .\.deployment\processing-01.json --reporting-journal .\.deployment\reporting-01.json --journal .\.deployment\powerbi-validation-01.json --refresh
```

Use the actual successful journal paths (including any recovered deployment).
`--refresh` explicitly requests one enhanced, transactional model refresh with
zero automatic retries. Its accepted ID is journaled before polling the
canonical Power BI API; regional Locations are never followed with credentials.
Without that flag the command is read-only. It first verifies current reporting
definitions/IDs, then checks report binding, all six live DAX counts, publication
run/model version and numeric metrics against the successful processing journal.
This is not visual validation: the output explicitly records
`visual_validation: "not_performed"`. Open the report separately.

## Parameterized notebook deployment

Each notebook has a Fabric **parameters cell**. A deployment agent must resolve
workspace/lakehouse names to IDs, update notebook dependency metadata, then inject
the parameters through the Fabric execution mechanism. Changing a Python variable
does **not** change an attached lakehouse.

The checked-in defaults describe the existing BRP2 environment. They are retained
for compatibility, not intended as defaults for another workspace. Always override
the target IDs. No deployment, resource creation or data transfer happens just by
changing these notebooks.

### Parameters

| Notebook | Parameters to supply for a new target |
|---|---|
| `POC_NB_PREPARE_SILVER` | `medallion_workspace_id`, `bronze_lakehouse_id`, `silver_lakehouse_id`, `silver_lakehouse_name`, `source_directory`, `silver_directory`, `silver_schema`, `silver_run_id`, `snapshot_date` |
| `POC_NB_VEHICLE_USAGE` | `medallion_workspace_id`, `silver_lakehouse_id`, `gold_lakehouse_id`, `silver_directory`, `silver_run_id`, `results_directory`, `errors_directory`, `experiment_name`, `model_name`, `snapshot_date`, `mode`, `demo_run_id`, `report_source_run_id` |
| `POC_NB_PUBLISH_REPORTING` | `medallion_workspace_id`, `gold_lakehouse_id`, `gold_lakehouse_name`, `results_directory`, `reporting_schema`, `report_run_id` |

`source_directory` remains a compatibility parameter in the ML notebook; that
notebook reads Silver, not source CSVs. Source CSV filenames and published table
names remain unchanged. The deterministic 100,000-row sample, holdout and model
hyperparameters are intentionally unchanged.

Directory parameters are relative to `Files`, use `/` between segments, and allow
letters, digits, underscores and hyphens. Absolute paths, empty segments and `..`
are rejected. Lakehouse/schema names are quoted in SQL; backticks and control
characters are rejected. Experiment/model names use letters, digits, `_` and `-`.
Keep schemas `POC_silver` and `POC_reporting` unless downstream definitions are
also adapted.

### Required attachment and execution order

1. Create or resolve three **distinct lakehouses in the target workspace**.
   Silver and Gold require schema support. Update the top-level notebook metadata
   (`default_lakehouse`, `default_lakehouse_name`,
   `default_lakehouse_workspace_id`, and `known_lakehouses`) before deployment.
   Attach Bronze by default to PREPARE_SILVER; attach Gold by default to
   VEHICLE_USAGE and PUBLISH_REPORTING. Attach Silver as needed for Spark catalog
   access. Preserve notebook kernels: PySpark for preparation/publication and
   Python for ML. Ensure their dependencies are available.
2. Provide the approved CSVs under Bronze `Files/<source_directory>`. Definitions
   in this repository do not include source data.
3. Execute PREPARE_SILVER with a fresh `silver_run_id`, for example `silver_demo01`.
4. Execute VEHICLE_USAGE with `mode="train"`, that same `silver_run_id`, and a fresh
   `demo_run_id`, for example `train_demo01`. This creates a model version in the
   target workspace. Do not copy model files out of the source environment.
5. Execute VEHICLE_USAGE again with `mode="reporting"`,
   `report_source_run_id="train_demo01"` and `demo_run_id="report_demo01"`.
   Keep the same snapshot, model/experiment names, Silver publication and
   directories. It reloads the recorded model version without retraining.
6. Execute PUBLISH_REPORTING with `report_run_id="report_demo01"` and the same
   Gold target and results directory.

The notebooks fail before processing if the actual workspace/default attachment
does not match the supplied IDs. Preparation also checks that the Silver name
resolves to its supplied ID. Local Python file access still uses
`/lakehouse/default/Files`, **after** attachment validation; cross-lakehouse access
uses explicit IDs. Runtime context is read-only and cannot be changed by parameters.

New evidence JSON includes target provenance. Training/reporting checks retain
compatibility with historical evidence lacking the additional workspace/directory
fields; when present, these fields must match. Existing model, Silver and Gold
identity checks remain mandatory.

### Example prompt contract

> In target workspace DEMO_TEST, use DEMO_RAW / DEMO_CLEAN / DEMO_RESULTS as
> Bronze / Silver / Gold. Resolve their IDs, remap notebook metadata, and inject
> the documented parameters. Use approved source data and fresh run IDs.
> Train Target_Model in Target_Experiment, prepare reporting results, then
> publish only after approval of the overwrite plan. Do not modify BRP2.
> Report the resolved IDs, run IDs, model version and validation results.

This full-scenario contract is not yet a complete deployment command. Bootstrap
and notebook import are implemented above. The agent must separately implement
data transfer/execution and remap semantic-model connections, report bindings
and Data Agent sources; those definitions have not been changed.

### Writes and reruns

Silver and reporting candidate folders reject reused run IDs. Published Silver
and Gold Delta tables **still use overwrite**, as in the original implementation:
a new run ID is not a new table namespace. Require approval before executing
against existing target tables. Gold's six tables are written sequentially, not
as one atomic release; a failed publication can leave partially updated tables.
Do not treat rerunning this chain as an idempotent workspace deployment.

The ML notebook's `errors_directory/last_error.json` records handled runtime
errors after target validation and may replace the previous diagnostic. Invalid
deployment configuration fails directly without writing diagnostics to the
potentially wrong lakehouse. No credentials or tokens belong in parameter cells,
deployment prompts or Git.

## Local validation

With pandas and numpy available, run from the repository root:

```powershell
python -m unittest discover -s tests -v
```

Tests execute extracted notebook functions with mocked Fabric services, validate
syntax and exercise renamed targets, invalid attachments/paths and publication
lineage. Orchestrator tests cover source preflight, fresh runtime run IDs,
overwrite refusal, sequential job failure, model/report rendering and reuse,
binding checks, refresh failure handling and live-data comparisons. They do not
execute notebooks' top-level training/publication cells.

The workflow was also exercised in an isolated F16 workspace: model metadata
was created before the first Gold publication; all four notebook jobs completed;
six Gold tables, registered-model lineage, Direct Lake refresh, report binding,
row counts and metrics were verified through live APIs/DAX. This smoke test is
not a guarantee of other tenants' permissions, capacity/runtime availability or
data quality. Browser visual validation still requires an authenticated Fabric
session. The PBIR validator reported zero errors and a warning that the
referenced visualContainer schema could not be fetched; that portion of schema
validation is therefore incomplete.