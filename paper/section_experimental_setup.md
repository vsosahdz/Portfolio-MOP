## Experimental setup

**Market and window.** Equities on the Mexican Stock Exchange, evaluated monthly over 18
blocks from 2025-01 to 2026-06. The universe is screened point-in-time on liquidity and
data quality, admitting between 88 and 101 instruments per month. Price history reaches
further back than the evaluation window to supply indicator warm-up, the expanding
training pool and the covariance estimate.

**Screening arms.** 6 arms, comprising a no-screening control that passes the whole
admitted universe to the optimiser and machine-learning screeners spanning support
vector machines, random forests, neural networks, gradient boosting and a tabular
foundation model. Each is tuned by inner cross-validation on a classification metric,
never on portfolio return, so the portfolio outcome stays out of sample. Three labelling
strategies are crossed with the screeners.

**Optimisation.** NSGA-III on the two-objective mean-variance problem, with expected
return and ex-ante risk estimated from trailing daily returns. Three portfolios are
extracted from each front -- minimum risk, knee and maximum return -- and evaluated
under two concentration caps, the looser of which reproduces the unconstrained
formulation. Independent seeds are replicated per cell, giving 21,275 evaluated
cell-months in the factorial.

**Evaluation.** Portfolios are priced on executable prices, charged round-trip
transaction costs under four scenarios, and reported at 50 basis points as the primary
scenario; the zero-cost scenario is reported only as an upper bound. Returns are
decomposed into market, selection, allocation and a currency tilt by an identity whose
residual is zero by construction, which the no-screening control exercises as an
internal check.

**Comparators.** Three tiers: deterministic analytic allocators over the same universe;
reference series an investor would hold instead, including the local index and the
sovereign short-rate instrument; and adapted published approaches, among them a
single-objective genetic algorithm on the Sharpe ratio and a cross-sectional momentum
portfolio.

**Ablation.** The search-space ablation truncates each screener's ranked output to fixed
sizes and adds no-preselection as an explicit level, giving 450 instances from 10 to 94
decision variables. Both attainment against an extended-budget reference and coverage
against the closed-form range are recorded per instance.

**Statistical protocol.** The confirmatory hypotheses, their contrasts and their metrics
are fixed before execution; everything else is labelled exploratory. Tests are paired on
the evaluation block, corrected across the confirmatory family, and reported with the
minimum effect the design can detect, which is 12.7% to 17.1% annualised. A result below
that bound is reported as unresolved, never as evidence of equivalence.
