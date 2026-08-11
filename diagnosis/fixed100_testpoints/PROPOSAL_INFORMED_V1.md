# Informed proposal kernel V1

This branch changes only the single-try tree proposal kernel used by
`DefaultBART` and default-sampler `ParallelTemperingBART`.  The legacy kernel
remains the default.  MTMH, move-type probabilities, PT ladder controls,
initialization, priors, likelihoods, and saved predictive metrics are unchanged.

Select the experiment with:

```text
--short-methods default default_pt
--proposal-kernel-default informed_v1
```

## Exact proposal definitions

### Grow

The grow proposal is a defensive mixture

```text
q_grow = (1 - w_grow) q_uniform + w_grow q_informed.
```

`q_uniform` is the original leaf / variable / cutpoint proposal.  The informed
component favors leaves with high partial-residual variation and cutpoints with
high normalized one-split SSE reduction.  The variable probabilities remain the
model's current split-rule prior probabilities (`s`, or uniform without the
Dirichlet prior).  Invalid proposals become self-transitions; V1 never retries
until success.

The grow transition term includes the split-rule prior explicitly:

```text
log p(rule) + log q_prune(reverse) - log q_grow(forward).
```

### Prune

Prune samples uniformly from current terminal split nodes.  Its reverse
probability is evaluated under the exact informed grow mixture in the pruned
tree:

```text
-log p(old rule) + log q_grow(reverse) - log q_prune(forward).
```

This is the required reverse correction; grow and prune are not tuned as
independent unrelated proposals.

### Change

Change is an 80/20 mixture by default:

- local: keep the variable and move within five cutpoint-grid positions;
- global: draw a variable and cutpoint from the original split-rule prior.

Both mixture densities and the old/new rule-prior ratio are included in the MH
transition term.  The uniformly selected internal-node probability cancels
because a change move preserves the number of internal nodes.

### Swap

V1 enumerates parent-child split pairs, simulates each swap, retains only pairs
that leave every active leaf nonempty, and samples uniformly from that valid
set.  The transition term is

```text
log(number of valid forward pairs) - log(number of valid reverse pairs).
```

## Instrumentation

Per method and chain, `proposal_diagnostics` records selected, feasible, and
accepted counts; informed/uniform or local/global component counts; failure
reasons; and summary statistics of proposal log-transition terms.  PT numerical
swap failures are rejected, counted, and saved separately rather than aborting
an otherwise valid run.

## Default V1 constants

```text
grow_informed_weight = 0.8
leaf_score_strength = 4.0
threshold_score_strength = 4.0
min_leaf = 1
change_local_weight = 0.8
change_local_radius = 5
move probabilities = grow 0.25, prune 0.25, change 0.40, swap 0.10
```

These are a first controlled experiment, not claimed optimal settings.
