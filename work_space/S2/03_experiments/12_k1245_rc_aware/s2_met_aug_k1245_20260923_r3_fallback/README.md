# K1245 independent F fallback gate, r3

Status: `experimental_unvalidated`; `operator_approved=false`; `formal_gate_status=not_run_not_passed`.

This is a new, isolated recovery attempt. It does not alter r1's failed strict gate or r2's completed pool evidence. The original strict result remains `F_STRICT_PREREG_GATE=false`.

The fallback is an engineering eligibility check, not a significance test and not evidence of model benefit. It requires the frozen 1,035-case training denominator, nondecreasing small-lesion fraction over the six preregistered burden bins, nondecreasing case-level sampling probabilities, probability mean 0.20 (absolute tolerance 1e-4), and maximum probability no greater than 0.60. Confidence intervals are descriptive only and are not a pass criterion.

Passing this fallback can only make a K1 exploratory configuration eligible for the separately required RC smoke and 24-case manual review. It does not pass K1's scientific gate. K5 remains blocked until K1 and K2 independently pass their preregistered scientific gates.

The machine plan is `K1245_F_FALLBACK_EXECUTION_PLAN_20260923_R3.json`. K1's corrected smoke contract is `K1_RC_SMOKE_EXECUTION_PLAN_20260923_R3C.json`. No training authorization is implied; smoke output must remain on hold until 24/24 manual reviews are recorded independently.
