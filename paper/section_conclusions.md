## Conclusions

An indicator computed against a reference the system produced itself cannot detect a
failure the system makes consistently. We gave a case where this is not a theoretical
concern: on a constrained portfolio problem, attainment against an extended-budget
reference rises with problem size while coverage against the closed-form attainable
range falls by a factor of roughly six, and the two readings support opposite
conclusions about whether preselection is needed for tractability.

The correction is cheap where it applies. Coverage requires only that each objective's
extreme be computable alone, which holds for a wide class of constrained formulations,
and it is bounded, normalised and immune to any bias the produced and reference fronts
share. We recommend it be reported alongside attainment rather than instead of it, and
we give the condition -- high attainment beside low coverage -- under which an
attainment figure should not be believed.

For the system studied, the diagnostic separated two claims that are usually made
together. Supervised preselection does make the search easier, measurably and in the
direction its proponents assert. It does not improve the portfolio: no screening arm
beat passing the whole admitted universe to the optimiser, and return attribution
located the shortfall in a negative selection component present in every arm. A
practitioner who wants the search benefit should seek it from dimension reduction that
does not condition on a predicted return.

We are explicit about what this study cannot settle. 18 evaluation blocks give a
detectable effect of 12.7% to 17.1% annualised, and no effect we estimated exceeds it,
so the performance comparisons are unresolved rather than null. The coverage result does
not depend on that bound: it is a measurement against a computed quantity, replicated
across months and seeds, and its direction is unambiguous.
