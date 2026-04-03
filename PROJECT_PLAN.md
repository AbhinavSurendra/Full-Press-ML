# Full Press ML Project Plan

## Project Direction

This plan is based primarily on the newer project presentation. The project should be framed as an NBA tracking-data problem, not as a broad sports analytics project.

The central question is:

**How accurately can we predict the outcome of an NBA possession from player-tracking data?**

The presentation already points to a sensible modeling path:

- clean and structure the tracking data
- build an `XGBoost` baseline
- test an `LSTM` as the main temporal model
- treat a `GNN` as a stretch goal only

## Recommended Scope

The most achievable version of the project is a **5-class possession outcome classification task**. Each example should correspond to one possession, with labels such as:

1. made 2
2. made 3
3. missed shot
4. turnover
5. free throws

This is narrow enough to finish well and broad enough to produce interesting analysis.

The strongest framing is not simply “classify the possession outcome,” but rather:

**How early in a possession can we predict the final outcome?**

That makes the project more compelling and better aligned with the presentation’s interest in temporal and spatial structure. Instead of using the full possession, the main experiments should use only the first few seconds of tracking data and compare performance across multiple horizons, such as 2, 4, 6, and 8 seconds.

## Achievable Deliverable

A strong and realistic final project would include:

- a reproducible pipeline from raw tracking data to possession-level examples
- a clean label definition for the 5 outcome classes
- engineered spatial and temporal features from the first `k` seconds of each possession
- a simple baseline such as multinomial logistic regression
- a stronger tabular baseline such as `XGBoost`
- one temporal model, ideally an `LSTM`
- evaluation using train/validation/test splits by game

This is enough for a complete and defensible course project. Trying to do too many architectures before the pipeline is stable would be a mistake.

## What To Prioritize

The main technical risk is not the model. It is the data pipeline. The hardest parts of the project are likely to be:

- possession segmentation
- label cleaning
- frame alignment
- preventing train/test leakage
- designing useful features from player and ball coordinates

Because of that, the project should prioritize data quality and evaluation discipline over model count.

If time becomes tight, the order of importance should be:

1. possession extraction and label quality
2. engineered-feature baseline
3. horizon-based evaluation
4. sequence model
5. stretch models

## Modeling Plan

### Baseline

Start with a possession-level tabular dataset built from the first `k` seconds of each possession. Useful features will likely include:

- offensive spacing
- nearest-defender distance
- ball position and movement summaries
- player speed summaries
- shot clock and time context, if available
- pass count or action count within the observed prefix

Train:

- multinomial logistic regression
- `XGBoost`

These models should establish whether the feature pipeline is working and which outcomes are easiest to separate.

### Temporal Model

Once the baseline pipeline is stable, train an `LSTM` on possession prefixes represented as short sequences of tracking frames or aggregated time buckets. The goal is to test whether temporal order adds predictive value beyond engineered static summaries.

### Stretch Goal

A `GNN` can be justified only if the rest of the project is already complete. It should not be treated as a required component.

## Evaluation Plan

The evaluation should be designed around the main contribution: early prediction.

At minimum, report:

- macro F1
- weighted F1
- per-class precision and recall
- confusion matrix
- calibration or confidence reliability, if feasible

The most important comparisons are:

- performance at different prefix lengths
- baseline vs temporal model
- which classes become predictable earliest

It would also be useful to test temporal generalization by training on earlier games and testing on later games, if the dataset split supports that cleanly.

## Novel Contribution

The most realistic novelty is in the framing and analysis, not in inventing a new architecture.

The best novel angle is:

**early possession outcome prediction from partial tracking data**

This gives the project a clear analytical contribution:

- measuring how predictability changes over a possession
- identifying which possession outcomes become obvious early
- showing whether temporal models help more at shorter horizons
- connecting the project to real-time game inference without needing to build a live system

Additional manageable novelty can come from feature design, especially spatial features such as spacing, defensive pressure, and paint congestion.

What should be avoided:

- comparing many models without a strong evaluation story
- forcing in a GNN just to appear advanced
- expanding the project beyond NBA possession modeling

## Recommended Work Plan

### Phase 1: Data Understanding

- inspect the dataset schema
- identify how possessions begin and end
- confirm the availability of labels, shot clock, and event markers
- manually inspect a sample of possessions for edge cases

### Phase 2: Dataset Construction

- build the possession extraction pipeline
- map terminal events into the 5-class label space
- extract possession prefixes at several time horizons
- create a clean training table for baseline models

### Phase 3: Baselines

- train logistic regression
- train `XGBoost`
- compare performance across time horizons

### Phase 4: Sequence Modeling

- train an `LSTM` on possession prefixes
- compare against the tabular baseline

### Phase 5: Analysis

- perform per-class error analysis
- inspect calibration and model confidence
- analyze which features matter most in the tabular model
- summarize where early prediction works and where it fails

## Minimum, Strong, And Stretch Outcomes

### Minimum Viable Project

- clean possession dataset
- one simple baseline
- one strong tabular baseline
- clear horizon-based evaluation

### Strong Final Project

- everything above
- `LSTM` comparison
- feature importance analysis
- calibration or temporal generalization analysis

### Stretch Project

- `GNN`
- live inference demo
- counterfactual or decision-value analysis

## Final Recommendation

The project should stay tightly focused on **NBA possession outcome prediction using 2015-2016 tracking data**. The achievable and most compelling version is a 5-class possession-end prediction task with the main contribution framed around **how early the final outcome can be predicted from partial tracking data**.

That scope is coherent, technically defensible, and novel enough for a strong course project without overreaching.
