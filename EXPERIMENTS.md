# Validation experiments

## What was measured

The provided test files have no labels, so test accuracy cannot be measured. These numbers come only from the training ground truth.

Validation holds out the 53,724 Source 1 records whose addresses match the configured Iowa/Rajasthan region pattern. It includes all known linked query records plus queries whose addresses mention those regions. The same 3% deterministic sample of non-validation queries was used for training in both runs. Both models used 61 pair features, 250 LightGBM rounds, and kept 8 candidates per query for model scoring.

`Candidate recall` is the share of known validation matches that survived blocking and the 8-candidate shortlist. `Macro F0.5` is the challenge metric, which puts more weight on precision than recall. The decision threshold was selected on this validation set.

## Results

| Initial candidates retrieved | Candidates scored per query | Best threshold | Validation macro F0.5 | Candidate recall |
| ---: | ---: | ---: | ---: | ---: |
| 20 | 8 | 0.65 | 0.9467 | 94.36% |
| 40 | 8 | 0.70 | 0.9497 | 94.88% |
| 60 | 8 | 0.65 | 0.9514 | 95.20% |
| 80 | 8 | 0.65 | **0.9520** | **95.44%** |

Retrieving 40 candidates improved macro F0.5 by 0.0030 (0.30 percentage points) and candidate recall by 0.52 percentage points over the 20-candidate baseline. Retrieval 60 improved them by 0.0047 (0.47 percentage points) and 0.84 percentage points; retrieval 80 improved them by 0.0053 (0.53 percentage points) and 1.08 percentage points. The final shortlist stays at 8, so the wider setting adds search work rather than increasing the output list size. The gain from 60 to 80 was only 0.0006 F0.5 (0.06 percentage points), so the extra search cost should be considered before deployment.

## Limits

- This is one geographic holdout using Iowa and Rajasthan; results may differ in other regions.
- The threshold was tuned on this same holdout, so the score is an estimate and may be optimistic.
- No labeled test set was supplied, so these are not test or leaderboard scores.
- The 40-candidate setting is a promising measured improvement, not proof that every prediction is correct.
