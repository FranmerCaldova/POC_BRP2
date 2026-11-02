---
description: "Deploy the isolated BRP workspace, lakehouses, notebooks and Power BI definitions without processing data."
agent: "agent"
argument-hint: "Target workspace name, tenant, active capacity and optional lakehouse names"
---

Deploy this repository's cumulative-mileage POC into the requested isolated
Fabric workspace. Follow the [deployment guide](../../README.md).

1. Verify Python 3.10+, Azure CLI authentication, target tenant and an existing
   active capacity. Resolve names with live APIs; never guess IDs or provision a
   capacity. Never write to the original BRP2 workspace.
2. Keep target configuration, plans and journals exclusively under ignored
   `.deployment/`. Start with [the example](../../deployment/target.example.json).
   Obtain missing target choices from the user. Never copy source CSVs, tokens
   or source workspace IDs into the public repository.
3. Use [deploy.py](../../deployment/deploy.py) `plan` then `apply` for `bootstrap`.
   Summarize creates/reuses, capacity and cost implications before requesting
   approval. Use the exact approval hash and a new journal. After creation,
   persist the verified workspace ID in local configuration.
4. Repeat the plan/review/approve/apply workflow with `--stage notebooks`.
   Verify all three imported definitions and target lakehouse attachments.
5. Use [reporting.py](../../deployment/reporting.py) `plan` then `apply` to create
   the Direct Lake semantic model and the PBIR report. Preserve the checked-in
   model, measures and visuals; remap target OneLake IDs and report/model binding
   only. Verify definitions read back from Fabric and all report field bindings.
   Do not refresh or query an empty model, create placeholder tables, or run
   notebooks merely to make deployment appear complete. If Fabric refuses
   metadata deployment before data exists, report that precise limitation.
6. Never overwrite differing existing items, delete resources, reassign capacity
   or blindly retry an accepted POST. On failure inspect the journal and actual
   target state, replan and request approval before a new operation.
7. Return workspace/report links, verified item IDs and any unresolved errors.
   Explain that definition deployment is not proof of data availability or
   report rendering. No Data Agent deployment is included.

Finish by directing the user to upload these four files into the **Bronze
lakehouse's Files**, beneath the configured `source_directory`
(`FAIL_RATES_PREDICTION_SANITIZED_DATASETS` by default):

- `SANITIZED_RAW_CLAIMS.csv`
- `SANITIZED_RAW_VEHICLE_DATA.csv`
- `SANITIZED_RAW_MODEL_FAMILY.csv`
- `SANITIZED_RAW_FLEET_OBSERVATIONS.csv`

Do not create Bronze Delta tables. After upload, use `/brp-process`.
