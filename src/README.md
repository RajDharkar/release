# Released code

`loaders/source6_pd_poetry*.py` parse the four raw volumes into the initial AAVE
and SAE poetry pools. `reclassify_pd_poetry_density.py` and
`reclassify_hughes_2pct_and_fix_ids.py` apply the two documented review passes.
Together they reproduce the 212 AAVE and 521 SAE retained pools from the named
raw inputs.

`perturb/rq1_proxy_sae521_multivalue.py` records the canonical human-SAE
Multi-VALUE transformation. `analysis/prepare_rq3_multivalue_v03.py` and
`analysis/analyze_rq3_multivalue_v03.py` cover the version-0.3 machine rewrite
analysis.

The `analysis/final_rq*.py`, robustness, operating-point, measured-profile, and
length-sensitivity programs are the implementations behind the frozen result
tables. `analysis/metrics.py` and `analysis/grammatical_proxy_final.py` contain
the released eye-dialect and ten-feature residual-grammar measurements.

The analysis programs use repository-relative paths. Some detector caches,
model checkpoints, raw source volumes, and the pinned Multi-VALUE environment
are not bundled; their identifiers and hashes remain in the provenance and
result artifacts so absent dependencies fail visibly rather than being guessed.
