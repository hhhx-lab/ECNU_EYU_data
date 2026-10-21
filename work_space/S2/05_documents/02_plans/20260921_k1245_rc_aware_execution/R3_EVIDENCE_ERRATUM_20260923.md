# K1245 r3 fallback attempt: evidence erratum

状态：`experimental_unvalidated`；`operator_approved=false`；`formal_gate_status=not_run_not_passed`。

The field names `source_rc27_manifest_identity_sha256` and `source_small8_manifest_identity_sha256` in the frozen r3 fallback plan are inaccurate: the bound values were copied from r2 `POOL_AND_EMPIRICAL_VALIDATION.manifests.*.sha256`, which are the manifest **file** hashes. The fallback gate itself only checked those file hashes against the same immutable r2 validation record; no fallback criterion depended on the manifest's canonical identity.

Read-only inspection of the actual r2 RC27 manifest established:

- file SHA256: `4bef77f255a1a5fa2ba65fa22b58c84191634859b4040eacf99e13f99059c35c`
- embedded canonical manifest identity: `a9b74b410838f8fcba50d66d52c7751bef2b07ddc94f2f774eabaed8c98701ac`

Do not edit the frozen r3 plan or its evidence. The K1 config child plan `K1_FALLBACK_CONFIG_PLAN_20260923_R3A.json` binds both identities explicitly. The earlier smoke child plan with the mislabeled identity is superseded by `K1_RC_SMOKE_EXECUTION_PLAN_20260923_R3A.json` and must not be used for launch.

Two K1 configuration operations failed closed before config creation or job submission. The first exposed a missing `mkdir -p` for nested runtime directories; the second correctly rejected a file-SHA/canonical-identity mismatch. Both failure audits remain in the r3 local operator directory. No training or smoke job was submitted by either operation.
