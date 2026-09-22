## The diagnostic

### What a self-produced reference cannot show

Let an optimiser produce a front $F$ on an instance, and let $R$ be a reference front
produced by the same optimiser under a larger budget. Any indicator of the form $I(F,
R)$ is a statement about the relationship between two samples from the same procedure.
If the procedure cannot reach a region of the objective space at all, neither $F$ nor
$R$ contains points there, the region is absent from the comparison, and $I(F, R)$ is
unaffected by its absence. The indicator does not fail; it answers a different question
than the one the analyst intends.

This matters most exactly where reassurance is most wanted. As an instance grows, a
search that degrades will produce a narrower $F$ -- and a correspondingly narrower $R$.
The ratio can rise while both collapse.

### Coverage against a computable bound

Where the extremes of each objective can be computed in closed form, the comparison need
not be self-referential. For the constrained mean-variance problem over a simplex, the
maximum expected return is attained by placing the whole budget on the single
highest-mean asset, and the minimum variance is the analytic minimum-variance portfolio.
Both are exact and neither involves the optimiser.

Writing $[\,a_j, b_j\,]$ for the attainable range of objective $j$ and $[\,\hat{a}_j,
\hat{b}_j\,]$ for the range the produced front spans, coverage is

$$ C_j(F) \;=\; \frac{\hat{b}_j - \hat{a}_j}{b_j - a_j}. $$

Coverage is bounded above by one, is comparable across instance sizes because it is
normalised by what is attainable on each, and cannot be moved by a bias the produced and
reference fronts share, because no reference front enters it.

The requirement is that the extremes be computable, not that the whole front be known.
This is weaker than it first appears: it holds whenever each objective, taken alone, is
optimised by a solution with a closed form, which covers linear objectives over a
simplex, convex quadratic objectives with analytic minimisers, and any objective whose
single-objective optimum is available from a solver the analyst already trusts.

### The guard condition

Coverage does not replace attainment, which answers a legitimate question about budget
sufficiency. It constrains how attainment may be read. We report the pair, and flag the
configuration in which the two disagree in the dangerous direction: attainment at or
above nine tenths beside coverage below one quarter. That combination says the search is
consistent with itself and narrow against the problem, which is the signature this paper
is about. A reported attainment unaccompanied by coverage is, on this evidence, not
interpretable.
