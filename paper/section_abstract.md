## Abstract

Intelligent decision-support systems built on multi-objective optimisation are routinely
validated with indicators computed against a reference front produced by the same
optimiser under a larger budget. We show this validation is blind by construction to any
failure the two runs share, and propose one that is not: where a problem's objective
extremes have a closed form, the fraction of the attainable range a front spans is
measurable against ground truth.

The difference is decisive. Over 450 instances spanning 10 to 94 decision variables, the
conventional indicator *rises* with problem size, from 0.945 to 0.964, suggesting large
instances are solved as completely as small ones. Our diagnostic *falls*, from 61.3% to
10.2% of the attainable return range and 34.0% to 4.3% of the risk range, with Spearman
rho = -0.93: the search degrades sharply with dimension, and the standard indicator
reports the opposite.

We apply it to a screening-plus-optimisation system for equity portfolio selection on an
emerging market: 6 screening arms, including gradient boosting and a tabular foundation
model, over 18 monthly blocks against deterministic, index and published baselines. The
diagnostic separates two claims usually made together: preselection does reduce search
difficulty, and it does not improve the portfolio. No screening arm beats passing the
whole universe to the optimiser, and return attribution locates the cause in a negative
selection component. A mixed model over 21,275 cell-months also understates standard
errors 6- to 9-fold against block resampling. Every artefact regenerates offline from a
cached dataset of record.
