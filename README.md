# POC_BRP2

Fabric proof of concept for cumulative vehicle mileage estimation, not failure prediction.

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