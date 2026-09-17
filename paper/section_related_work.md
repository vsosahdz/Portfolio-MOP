## Related work

**Performance assessment in multi-objective optimisation.** The hypervolume indicator
(Zitzler and Thiele, 1999) and the attainment function (Fonseca and Fleming, 1996) are
the standard instruments for comparing stochastic multi-objective optimisers, and both
require a point of comparison. Where the true Pareto front is known -- on synthetic
benchmarks constructed for the purpose -- that point is exact. Where it is not, which is
every real application, practice substitutes a reference front assembled from the
algorithms under study, often from an extended-budget run of the same algorithm. The
substitution is usually described as a practical necessity and treated as benign.

This paper is about a case where it is not benign. A reference front produced by the
same optimiser inherits whatever that optimiser cannot do, so an indicator computed
against it measures consistency rather than quality, and reports success precisely when
the failure is systematic. The problem is distinct from the well-studied sensitivity of
hypervolume to its reference point, which concerns the scaling of a comparison rather
than its blindness.

**Statistical comparison across problems.** Comparing optimisers over multiple problem
instances is conventionally handled with rank-based tests and critical-difference
diagrams (Demšar, 2006). We use that machinery and report the interval it implies,
because over a small number of blocks the interval is wide enough that adjacency in such
a diagram carries almost no information -- a property of the design that a diagram
without its interval invites readers to ignore.

**Portfolio selection as a multi-objective problem.** The mean-variance formulation
(Markowitz, 1952) is the canonical two-objective instance, and evolutionary approaches
to it are well established; NSGA-III (Deb and Jain, 2014) is the algorithm used here.
The formulation suits our purpose for a reason independent of finance: under a simplex
constraint both extremes are analytic, so the attainable range is computable and the
indicator can be checked against ground truth rather than against itself.

**Preselection and dimensionality reduction.** Applications commonly reduce the decision
space before optimising, and supervised preselection -- training a classifier to admit
or reject candidates -- is a frequent choice, justified on two grounds: that it improves
the solution by removing bad candidates, and that it makes the search tractable by
reducing decision variables. Our ablation tests both claims independently, and finds
that they come apart.

**Comparators.** The empirical arms are placed against deterministic baselines (equal
weighting, inverse volatility, minimum variance, maximum Sharpe), against the reference
series an investor would hold instead -- the CETES 28-day instrument at 7.8% annualised,
the local index at 19.4%, and the S&P 500 converted to pesos -- and against adapted
published approaches: a single-objective genetic algorithm maximising the Sharpe ratio,
an L1-regularised mean-variance formulation with shrinkage covariance, and a
cross-sectional momentum portfolio. Momentum is the demanding one, at 28.2%, above every
arm the studied method produces.
