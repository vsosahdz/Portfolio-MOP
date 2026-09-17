## Does preselection justify itself?

The original formulation motivates supervised screening on two grounds: that it improves
the portfolio by removing instruments predicted to fall, and that it makes the
optimisation tractable by reducing the number of decision variables. The ablation tests
both, over 450 instances spanning 10 to 94 decision variables with no-preselection as an
explicit level.

**On portfolio quality, the argument fails.** The no-screening control returns 12.0%
annualised against 7.9% for the best screening arm, a gap of 4.1%. The gap is below the
design's resolution and is therefore not a significant difference, but it points the
wrong way for the method, and the attribution decomposition shows why: the selection
component is negative for every screener.

**On computational cost, the argument also fails, and by a wide margin.** Wall-clock
rises from 2.1 to 4.1 seconds between 10 and 94 variables -- a factor of 2.0 -- while
the covariance matrix grows by a factor of 88. The quadratic growth in problem size does
not translate into a proportionate cost, so preselection is not buying compute.

**On tractability, the argument holds, and this study initially missed it.** The obvious
metric says otherwise: attainment against an extended-budget reference rises with
problem size. That metric is self-referential, because the reference front comes from
the same optimiser. Against the closed-form attainable range -- maximum return is the
whole budget in the highest-mean asset, minimum variance is analytic -- the picture
inverts. Return coverage falls from 61.3% at 10 variables to 10.2% at 93; risk coverage
falls from 34.0% to 4.3%. The search genuinely degrades with dimension.

The two findings are separable and both are reported. Preselection does make the
optimisation easier in a way that is real and measurable. It does not follow that
preselection helps, because the screening signal is negative enough to outweigh what the
smaller search space buys. A reader who wants the tractability benefit without the
selection penalty should look at dimension reduction that does not condition on a
predicted return.
