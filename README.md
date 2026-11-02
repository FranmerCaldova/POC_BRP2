# POC_BRP2

Fabric proof of concept for cumulative vehicle mileage estimation, not failure prediction.

## Prompt deployment: step 1 bootstrap

The standard-library Python orchestrator in [deployment/deploy.py](deployment/deploy.py)
implements **only workspace/lakehouse planning and bootstrap**, not the complete
solution deployment. Python 3.10+ and an authenticated Azure CLI are required;
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
Do not treat an instruction in a prompt as proof that a complete BRP deployment
is implemented. Notebook/dependency remapping and end-to-end execution are later steps.

API references:
[Create workspace](https://learn.microsoft.com/en-us/rest/api/fabric/core/workspaces/create-workspace),
[Create lakehouse](https://learn.microsoft.com/en-us/rest/api/fabric/lakehouse/items/create-lakehouse).

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

This is a contract for a deployment agent, not an implemented deployment command.
These changes only parameterize the notebooks. The agent must separately remap
semantic-model connections, report bindings and Data Agent sources; these files
have not been changed.

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
lineage. They do not execute the notebooks' top-level training/publication cells.
An actual Fabric run in a separate workspace remains required to validate
Spark catalog resolution, permissions, MLflow and target data end to end.