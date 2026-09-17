## Limitations

**Resolution.** Conclusions rest on 18 monthly blocks, giving a minimum detectable
effect of 12.7% to 17.1% annualised. No factor effect in the study exceeds that bound.
The hierarchical Friedman tests return p between 0.42 and 0.68, and the critical
difference over the headline set is 1.78 ranks against a spread far smaller. These are
statements about the design, not about the world, and a longer window is the single
change that would most improve the study.

**Dependence and the mixed model.** Fitting the mixed model over 21,275 cell-months
yields standard errors between 6 and 9 times smaller than resampling whole months,
because cells within a month hold overlapping stocks and a single random intercept
removes only the month's common level. The point estimates agree; the apparent
significance does not survive. The month-paired tests are confirmatory, as
pre-registered, and the model is reported as descriptive. The labeling-by-allocator
interaction -- the direct test of whether an apparent labeling advantage is really a
concentration effect -- has 10 terms and none reaches significance even at the model's
inflated precision.

**Survivorship.** The universe is screened point-in-time from price history rather than
from an exchange listing record, because the exchange's listing endpoints were
unavailable and no archived snapshot exists. No instrument ceased trading during the
evaluation window. That is a lower bound on the exposure, not a proof of its absence.

**Costs.** Transaction costs are applied after optimisation rather than inside the
objective, so the optimiser never trades return against the cost of achieving it. Costs
are also charged at a flat spread across instruments. Real spreads widen on the smaller
and less liquid names, which the screening arms select more often than the control does,
so the flat assumption flatters the screened arms relative to the control -- the bias
runs against the study's own conclusion rather than toward it.

**Covariance estimator.** The shrinkage sensitivity arm moves the reported contrasts by
at most 0.7% annualised and no contrast reverses sign. The arm covers a subset of
screeners and one labeling strategy, so this is an argument from the observed effect
size rather than a proof of insensitivity across the full grid.

**Scope.** The study covers one exchange over one window with one feature dictionary.
Nothing here establishes that supervised screening subtracts value in general; it
establishes that it did so here, under a protocol designed in advance to be able to
detect the opposite.
