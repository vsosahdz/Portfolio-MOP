## Threats to validity, and what did not survive

This section reports what limits the results above and what did not survive measurement,
at the same length as the findings, because a reader cannot weigh a claim without
knowing which of its neighbours failed.

**A conclusion of this paper reverses one we had already drawn.** An earlier reading of
the ablation concluded that preselection buys neither computational cost nor
tractability, on the evidence that attainment rises with problem size. The cost half of
that survives; the tractability half was an artefact of the instrument, and the coverage
measure of Section~\ref{sec:the-proposal} exists because of it. We report the reversal
rather than the corrected conclusion alone, since the failure mode is the paper's
subject and we were subject to it.

**An earlier version of this study had a one-month look-ahead and produced returns that
refuted themselves.** Monthly features aggregate daily indicators of their own month,
and the screening stage predicted with month $E$'s features for a portfolio held through
month $E$. The correlation between a month's feature and that month's return is above
nine tenths, and against the following month's return below a tenth. The resulting arms
returned between fifty and sixty-five per cent annualised with monthly Sharpe above
four, against a momentum bar of 28.2%. Selection now uses the preceding month's
features. The guard that should have caught it did not, because the defect lived in the
join between the selection month and the holding month rather than in either alone.

**The package did not regenerate its own results, and nothing noticed.** The
configuration fingerprint that seeds the optimiser had drifted from the one the stored
results were produced under. Deterministic allocators matched bit for bit, so only the
genetic replicates differed, and the headline figures moved by at most half a percentage
point -- close enough that no analysis looked wrong. The current results are regenerated
from the shipped configuration, and the fingerprint is now checked against the manifest.

**Two further defects inverted intermediate results before they were found.** A momentum
comparator ranked over all price history rather than the month's admitted universe,
inflating its return by eighteen percentage points; and the Sharpe ratio was annualised
by twelve in one hypothesis test, which is not a monthly rate, producing a reported
effect of several hundred per cent. Both are fixed and both now carry a guard.

**Coverage of the design is the binding limit on every performance claim.** 18
evaluation blocks give a detectable effect of 12.7% to 17.1% annualised, and no effect
we estimated exceeds it. The screening comparison, the labelling comparison and the
concentration-cap comparison are therefore unresolved rather than null, and we do not
claim equivalence anywhere. The coverage result does not rest on that bound: it is a
measurement against a computed quantity, replicated across months and seeds.

**One screener was omitted with its cost measured rather than estimated.** A
contrast-pattern classifier was excluded after its measured runtime reached one hundred
and thirty-seven hours on this grid. Injecting a hypothetical seventh arm at an
implausibly favourable return moves the pooled screening effect by under three
percentage points and leaves it well inside the detectable bound, so the omission cannot
carry the screening conclusion.

**Three limits lie outside what this study can address.** Point-in-time exchange
membership is not obtainable for this market, so the universe is screened from price
history and the survivorship exposure is declared with its measurement rather than
eliminated. Transaction costs are applied after optimisation rather than inside the
objective, so the optimiser never trades return against the cost of achieving it. And
costs are charged at a flat spread, which flatters the screened arms relative to the
control because they hold smaller and less liquid names -- a bias that runs against this
paper's own conclusion rather than toward it.
