## Contributions

**An indicator that cannot see its own failure mode, and a correction.** Attainment
computed against an extended-budget reference front rises with problem size on our
ablation, from 0.945 at 10 decision variables to 0.964 at 94. Measured against the
closed-form attainable range instead, coverage of the reachable return range falls from
61.3% to 10.2% and of the risk range from 34.0% to 4.3%, with Spearman rho = -0.93 over
75 instances. The two readings are contradictory and only one is checkable against
ground truth. We give the coverage measure, a guard that refuses to report attainment
without it, and the condition that flags the pathology: high attainment beside low
coverage.

**A demonstration that this changes a substantive conclusion.** The ablation was
designed to test whether supervised preselection earns its place. On the attainment
reading it does not, on any axis. On the coverage reading the tractability claim holds
and the quality claim still fails: preselection genuinely makes the search easier, and
separately, the screening signal is bad enough that the easier search does not pay for
it. Those are separable findings, and an indicator that conflates them would have
suppressed one.

**An inferential hazard in blocked experimental designs.** Fitting a mixed model over
21,275 cell-months with the block as a random effect yields standard errors 6 to 9 times
smaller than resampling whole blocks. The point estimates agree exactly; only the
precision differs. A random intercept removes a block's common level but not the
cross-sectional dependence among configurations that share underlying data, so the model
treats correlated cells as replicates. We report both and let the pre-registered
block-paired tests govern.

**A negative empirical result, reported in full.** Across 6 screening arms over 18
monthly blocks, none beats passing the whole admitted universe to the optimiser: 12.0%
annualised for the no-screening control against 7.9% for the best arm and 0.1% for the
worst, at 50 basis points. Return attribution isolates the mechanism -- the selection
component is negative for every screening arm, between -7.3% and -3.1%, and exactly zero
for the control, which validates the decomposition. No effect exceeds the design's
detectable bound of 12.7% to 17.1%, so these are reported as unresolved rather than as
null.

**A replication package.** The pipeline is specified before execution, runs from a
cached dataset of record with the network unavailable, and regenerates every table,
figure and quoted number. Configuration drift is detectable: seeds derive from a
configuration fingerprint that is checked against the manifest, a property added after
the drift occurred undetected during development.
