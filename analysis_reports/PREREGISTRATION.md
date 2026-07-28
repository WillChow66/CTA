# Pre-registration: does Section 6.2's "declarative over procedural" design rule predict anything?

Written 2026-07-25, BEFORE any rating was collected and BEFORE any
skill-score-to-outcome statistic was computed. Frozen; not edited afterwards.

## Motivation

Sec 6.2 of the CTA submission states:

> skills should describe properties of correct outputs, not procedures to produce
> them. Procedural skills compete with the agent's default loop and create the
> very SIPs that depress dP. Declarative "what does correct look like" skills are
> less likely to over-anchor the agent or trigger premature closure.

This rule was derived from Finding #2 (bucket -> dominant SIP signature). Finding
#2 does not survive a per-task test (bucket-label permutation G=4.50, p=0.84;
ceiling SA-vs-EP paired sign test p=1.000; the argmax flips SA->EP if the ceiling
cutoff moves 0.90 -> 0.80). Sec 6.2 therefore currently has no surviving
empirical basis. Reviewer 3|3 Q4 asks exactly this.

The claim is testable directly and was never tested directly: the independent
variable (how procedural a skill document is) is a property of the 49 SKILL.md
files, which we have; the dependent variables are the per-task outcomes, which we
also have.

## Design

Blind LLM rating of the independent variable, then correlation against outcomes.

Unit of analysis: task/skill, n = 49 (one instance per skill, r=1).

### Independent variable

`procedural_score` in [0,1] for each of the 49 SKILL.md documents.
0 = purely declarative (describes properties/criteria a correct output must
satisfy), 1 = purely procedural (prescribes an ordered sequence of actions).

Secondary independent variable: `has_terminal_completion_step`, binary. True iff
the document ends by directing the agent to perform a wrap-up action (commit,
document, changelog, PR, mark done, report completion). This is the paper's
Case 1 premature-closure mechanism, operationalised as a document property.

Raters: `gemini` and `claude-opus-4.8` via the NAIRR gateway, two independent
runs each (4 passes total). Primary score = mean over all completed passes.
Per-rater scores reported as robustness.

BLINDING: each rating call receives ONLY the skill document text and the rubric.
No pass rates, no dP, no SIP counts, no token counts, no task name, no mention of
CTA, SIPs, or that an experiment exists. The rater cannot condition on outcomes.

### Dependent variables (from cta_output/cta_combined_results.json and
### config/cta_task_metadata.json, per task)

1. `pass_rate_delta` (dP)
2. `surface_anchoring` count (SA)
3. destructive share = (SA + CB) / total_SIPs
4. `total_sips`
5. `token_overhead_ratio`

### Directional hypotheses (Sec 6.2 predicts)

- H1: procedural_score correlates NEGATIVELY with dP.
- H2: procedural_score correlates POSITIVELY with SA count.
- H3: procedural_score correlates POSITIVELY with destructive share.
- H4 (mechanism, secondary): has_terminal_completion_step = True tasks have
  higher SA / destructive share than False tasks.

Exploratory, no signed prediction: total SIPs, token_overhead_ratio.

All three primary tests are ONE-SIDED in the direction Sec 6.2 predicts, since
the paper makes a signed claim. Two-sided p also reported.

### Statistics

Spearman rho (primary; the SIP and token distributions are heavy-tailed) and
Pearson r (secondary), each with a 95% percentile bootstrap CI (10,000 resamples
of tasks) and a permutation p-value (20,000 label shuffles of procedural_score).
Holm correction across the three primary tests (H1-H3).

Reliability is reported first and gates interpretation: Pearson + Spearman
between raters on procedural_score, Cohen's kappa on the binary, and intra-rater
test-retest from the two runs per rater. If between-rater agreement is poor, the
construct Sec 6.2 rests on is not reliably measurable, and that is the finding,
independent of any correlation.

Correlations are also reported against the reliability ceiling
sqrt(reliability), since an unreliable predictor cannot correlate with anything.

### Power warning (stated in advance, not after seeing a null)

45 of 49 tasks have dP = 0 exactly. The dP variable has almost no variance: it is
a near-constant. The H1 test is therefore close to powerless and a null result on
H1 is NOT evidence that procedural skills are harmless. This is stated here, in
advance, so that the H1 null cannot be reported as if it were evidence of
absence. H2 and H3 have real variance across tasks and are the informative tests.

### Decision rule, fixed in advance

- All three primary correlations null, with adequate reliability and adequate
  variance in the DV -> Sec 6.2 is unsupported by the paper's own data and must
  be softened to a hypothesis in the rebuttal.
- Correlations in the direction Sec 6.2 predicts, surviving Holm -> Sec 6.2 is
  supported by a test independent of Finding #2, and that test should be added.
- Correlations in the OPPOSITE direction -> Sec 6.2 is contradicted and must be
  withdrawn, not merely softened.
- Poor inter-rater reliability -> Sec 6.2's construct is not operationalisable;
  report as unmeasurable rather than as supported or refuted.

Nothing below the reliability gate will be reported selectively: all five DVs x
both correlation types x per-rater breakdown are dumped to
sec62_correlations.json regardless of outcome.
