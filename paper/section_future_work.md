## Future work

**Cost-aware optimisation.** The most direct extension is to move transaction costs
inside the objective rather than applying them afterward. Realised turnover is high
enough that the optimiser is currently choosing portfolios it would not choose if it
were charged for reaching them, and a third objective or a turnover penalty would let
the front express that trade-off rather than have it imposed after the fact.

**Search that reaches the extremes.** The coverage result gives a concrete target: at 93
decision variables the optimiser spans 10.2% of the attainable return range.
Reference-direction methods, decomposition-based algorithms, or simply seeding the
population with the analytic extremes would attack this directly, and the closed-form
bound gives an unambiguous measure of progress that attainment against a self-produced
reference cannot provide.

**Dimension reduction without return prediction.** The ablation separates a real
tractability benefit from a harmful selection signal. Clustering, sector-balanced
sampling, or liquidity-based admission would reduce the search space without
conditioning on a predicted return, and would test whether the benefit survives when the
penalty is removed.

**Longer windows and more blocks.** The design's resolution is the binding constraint on
every conclusion. Extending the evaluation window is the change that most improves what
can be claimed, and the protocol is specified so that this requires no methodological
change.

**Alternative labels.** All three labeling strategies condition on realised returns over
a one-month horizon. Labels defined on risk-adjusted outcomes, on longer horizons, or on
relative rather than absolute performance would test whether the negative selection
component is a property of supervised screening or of this particular target.
