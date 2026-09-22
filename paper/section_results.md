## Results

### The two measures disagree in direction

Across 450 ablation instances, attainment against an extended-budget reference rises
monotonically with problem size, from 0.945 at 10 decision variables to 0.964 at 94.
Read alone, this says the larger instances are solved as completely as the smaller ones
and that a fixed-budget comparison across sizes is sound.

Coverage says the opposite. Over 75 instances, the fraction of the attainable return
range the front spans falls from 61.3% at 10 variables to 10.2% at 93, and the risk
range from 34.0% to 4.3%. The rank correlation between problem size and return coverage
is -0.93. Because the asset pools are nested, the attainable range can only widen with
size -- and it does, while the range the optimiser spans narrows.

The effect is visible in the portfolios themselves rather than only in the indicator. On
the no-screening control at the median evaluation month, the three extracted profiles
hold nearly the entire universe at a maximum weight close to a thirtieth, and their
expected returns differ by less than a tenth of a percentage point per month. The
aggressive and conservative profiles are, in substance, the same portfolio. The
concentration ratio against equal weight stays roughly constant across sizes, so the
optimiser is finding structure; it is the span that collapses.

### Search cost does not grow with the problem

Wall-clock rises from 2.1 to 4.1 seconds between 10 and 94 variables, a factor of 2.0,
while the covariance matrix grows by a factor of 88. Preselection is therefore not
buying computation at this scale, whatever it may buy in search quality.

### The system's own performance

No screening arm beats the no-screening control. At 50 basis points the control returns
12.0% annualised against 7.9% for the best machine-learning arm, neural network, and
0.1% for the worst, random forest. Return attribution locates the cause: the selection
component is negative for every screening arm, between -7.3% and -3.1% annualised, while
the same decomposition returns exactly zero for the control.

Against the comparator suite the ordering is unfavourable to the proposed architecture.
The cross-sectional momentum portfolio returns 28.2%, the local index 19.4%, and the
sovereign short-rate instrument 7.8%, all above every screening arm and the first two
above the control.

### Nothing separates statistically, and the design says why

None of the confirmatory hypotheses rejects, and every estimated effect falls below the
detectable bound of 12.7% to 17.1%. The hierarchical tests, each holding the other
factors at their best level, return probabilities between 0.42 and 0.68. The critical
difference over the pre-registered headline set of 6 cells is 1.78 ranks against an
observed spread far smaller, so no pair is separated.

These are statements about resolution, not about equivalence, and we report them as
such.

### Inference is sensitive to how dependence is handled

Fitting a mixed model over 21,275 cell-months with the evaluation block as a random
effect yields standard errors 6 to 9 times smaller than resampling whole blocks. The
point estimates agree; only the precision differs. Cells within a block hold overlapping
instruments and move together, and a random intercept removes the block's common level
while leaving that dependence intact, so the model treats correlated configurations as
replicates. The block-paired tests, named confirmatory in advance, govern; the model is
reported as descriptive.

The labelling-by-allocator interaction, which is the direct test of whether an apparent
labelling advantage is a concentration effect in disguise, has 10 terms and none reaches
significance even at the model's overstated precision.

### Robustness

Substituting a shrinkage covariance estimator moves the reported contrasts by at most
0.7% annualised, and no contrast reverses sign. The estimator is therefore not promoted
to a design factor.
