## Introduction

A decision-support system built on multi-objective optimisation presents its user with a
set of trade-offs rather than a single answer, and the value of the system rests on that
set being a fair picture of what is achievable. Validating it is therefore a question
about the produced front, not about the objective values of any one solution, and the
field answers that question with indicators: hypervolume, inverted generational
distance, attainment. Each requires something to compare against.

On synthetic benchmarks the comparison is exact, because the true front is known by
construction. In an application it is not known, and practice substitutes a reference
front assembled from the algorithms under study -- commonly the same algorithm run under
a larger budget. The substitution is treated as a practical necessity with no
consequences for what the indicator means.

This paper shows the consequence is real and can invert a conclusion. A reference front
produced by the same optimiser inherits whatever that optimiser cannot reach, so an
indicator computed against it reports agreement between two runs rather than quality.
Where the search systematically fails to reach part of the objective space, both runs
fail the same way, the indicator sees nothing, and the system passes validation while
presenting its user with a fraction of the available trade-offs.

We make the failure measurable. For problems whose objective extremes have a closed form
-- which includes the constrained mean-variance formulation and any problem where a
linear objective is optimised over a simplex -- the attainable range can be computed
rather than estimated, and the fraction of it a front spans is a quantity no shared bias
can move. We call it coverage, give the condition under which a reported attainment is
untrustworthy, and show that on our problem the two measures disagree in direction
across the full range of problem sizes tested.

The setting is a screening-plus-optimisation system for equity portfolio selection: a
supervised classifier admits or rejects candidate instruments, and a multi-objective
optimiser allocates over what survives. This architecture is common, and the
preselection step is justified on two grounds that are usually asserted together -- that
it improves the portfolio, and that it makes the optimisation tractable by reducing
decision variables. Our diagnostic separates them, and they come apart: the tractability
claim holds and the quality claim fails.

Section headings below follow the order in which the argument is built. We state the
diagnostic and its guard condition, describe the system and the evaluation protocol,
report what both measures say about problem size, report the system's own performance
against deterministic, index and published comparators, and close with what the design
can and cannot resolve.
