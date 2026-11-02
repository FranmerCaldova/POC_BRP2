---
description: "Validate uploaded BRP CSVs, execute Silver, training, reporting and Gold, then validate the Power BI report."
agent: "agent"
argument-hint: "Local target configuration, snapshot date and explicit permission if replacing existing tables"
---

Process this repository's uploaded CSVs through to Power BI in the previously
deployed target workspace. Follow the [processing guide](../../README.md).

1. Load the ignored local target configuration and verify the Azure CLI tenant,
   workspace, capacity, lakehouse IDs and notebook definitions. Never use BRP2
   as the write target. Do not copy data or create Bronze tables.
2. Select a fresh `notebook_parameters.run_prefix` and the agreed business
   `snapshot_date`. Runtime parameters are passed by the orchestrator; do not
   overwrite imported notebooks to change run IDs.
3. Use [process.py](../../deployment/process.py) `plan`. Validate four nonempty
   CSVs in the configured input folder, required headers, source ETags,
   unused output folders, target bindings and existing tables. The preflight
   validates headers, not every data row.
4. Present the four sequential jobs, compute/model-registration costs, source
   files and tables that would be replaced. Existing Silver or Gold tables
   require explicit user permission and `processing.allow_table_overwrite=true`.
   A request to process files is not implicit approval to replace tables.
5. After approval, use `run` with the exact hash and a fresh journal. Wait for
   Silver, ML training, ML reporting and Gold publication in that order. Require
   each completed job AND its target/run/model-version evidence before advancing.
   Reuse the registered model for reporting; do not retrain in reporting mode.
6. Stop downstream work on any failed job, inconsistent evidence or timeout.
   Preserve the accepted job handle and partial outputs. Never blindly replay
   a POST or reuse an output folder; inspect the exact job before recovery.
   Gold publication is sequential, not atomic.
7. Verify or deploy the target-bound reporting definitions with
   [reporting.py](../../deployment/reporting.py), using its separate plan and
   approval workflow. Never overwrite a differing model/report.
8. Connect to the deployed semantic model with the available Power BI modeling
   tools, refresh/frame Direct Lake and query the live data. If those tools
   cannot connect, use `reporting.py validate --refresh` with the successful
   processing/reporting journals and a fresh output journal, as documented.
   Never accept license terms on the user's behalf. Compare all six
   table counts and the publication run/model version with the processing
   journal; compare MAE, baseline MAE, RMSE and improvement with training evidence.
   Verify the report is bound to this exact target model. If refresh reports
   credential errors, STOP and instruct the user to configure the connection in
   Power BI Service; do not retry or work around credentials.
9. Open the target report and verify its three pages render. If browser access
   is unavailable, state that visual validation remains unverified, independently
   of API/DAX checks.

Return the target workspace/report links, actual run IDs, model version, row
counts, metrics, journal locations and validation results. Clearly separate
completed processing, metadata deployment, live model validation and visual
validation. Never report end-to-end success when a later stage is blocked.
