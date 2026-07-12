# Fair Matched-Budget Arxiv Retrieval

Two locked query seeds, 100 queries each. Every training run used exactly 9,000 optimizer steps. FullCov is primary; recall is secondary.

## Best-selected checkpoints by objective

| Objective | Retrieval | Budget | Training replicates | FullCov | Rate | Recall | Avg max true rank |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| control | locked dynamic | 50 | 2 | 171/400 | 0.427 | 0.8557 | 74.02 |
| cvar | locked dynamic | 50 | 1 | 82/200 | 0.410 | 0.8229 | 74.52 |
| final | locked dynamic | 50 | 2 | 187/400 | 0.468 | 0.8643 | 58.97 |
| topk | locked dynamic | 50 | 1 | 78/200 | 0.390 | 0.8237 | 81.53 |
| control | locked dynamic | 75 | 2 | 248/400 | 0.620 | 0.9224 | 74.02 |
| cvar | locked dynamic | 75 | 1 | 116/200 | 0.580 | 0.9038 | 74.52 |
| final | locked dynamic | 75 | 2 | 264/400 | 0.660 | 0.9298 | 58.97 |
| topk | locked dynamic | 75 | 1 | 115/200 | 0.575 | 0.9037 | 81.53 |
| control | locked dynamic | 100 | 2 | 315/400 | 0.787 | 0.9583 | 74.02 |
| cvar | locked dynamic | 100 | 1 | 154/200 | 0.770 | 0.9522 | 74.52 |
| final | locked dynamic | 100 | 2 | 321/400 | 0.802 | 0.9590 | 58.97 |
| topk | locked dynamic | 100 | 1 | 153/200 | 0.765 | 0.9545 | 81.53 |
| control | fixed | 20 | 2 | 87/400 | 0.217 | 0.6620 | 74.02 |
| cvar | fixed | 20 | 1 | 38/200 | 0.190 | 0.6240 | 74.52 |
| final | fixed | 20 | 2 | 104/400 | 0.260 | 0.7012 | 58.97 |
| topk | fixed | 20 | 1 | 33/200 | 0.165 | 0.5960 | 81.53 |
| control | fixed | 50 | 2 | 156/400 | 0.390 | 0.8319 | 74.02 |
| cvar | fixed | 50 | 1 | 85/200 | 0.425 | 0.8247 | 74.52 |
| final | fixed | 50 | 2 | 195/400 | 0.487 | 0.8740 | 58.97 |
| topk | fixed | 50 | 1 | 71/200 | 0.355 | 0.7982 | 81.53 |
| control | fixed | 100 | 2 | 264/400 | 0.660 | 0.9384 | 74.02 |
| cvar | fixed | 100 | 1 | 132/200 | 0.660 | 0.9338 | 74.52 |
| final | fixed | 100 | 2 | 328/400 | 0.820 | 0.9676 | 58.97 |
| topk | fixed | 100 | 1 | 128/200 | 0.640 | 0.9286 | 81.53 |

## Individual final-objective checkpoints

| Checkpoint | Retrieval | Budget | FullCov | Recall | Avg max true rank |
| --- | --- | ---: | ---: | ---: | ---: |
| fair_final_seed7101_best | locked dynamic | 50 | 95/200 | 0.8667 | 58.88 |
| fair_final_seed7101_best | locked dynamic | 75 | 133/200 | 0.9303 | 58.88 |
| fair_final_seed7101_best | locked dynamic | 100 | 160/200 | 0.9580 | 58.88 |
| fair_final_seed7101_best | fixed | 20 | 51/200 | 0.6995 | 58.88 |
| fair_final_seed7101_best | fixed | 50 | 96/200 | 0.8714 | 58.88 |
| fair_final_seed7101_best | fixed | 100 | 165/200 | 0.9665 | 58.88 |
| fair_final_seed7101_final | locked dynamic | 50 | 97/200 | 0.8627 | 59.21 |
| fair_final_seed7101_final | locked dynamic | 75 | 132/200 | 0.9317 | 59.21 |
| fair_final_seed7101_final | locked dynamic | 100 | 161/200 | 0.9600 | 59.21 |
| fair_final_seed7101_final | fixed | 20 | 48/200 | 0.6860 | 59.21 |
| fair_final_seed7101_final | fixed | 50 | 95/200 | 0.8719 | 59.21 |
| fair_final_seed7101_final | fixed | 100 | 164/200 | 0.9650 | 59.21 |
| fair_final_seed7102_best | locked dynamic | 50 | 92/200 | 0.8620 | 59.05 |
| fair_final_seed7102_best | locked dynamic | 75 | 131/200 | 0.9292 | 59.05 |
| fair_final_seed7102_best | locked dynamic | 100 | 161/200 | 0.9600 | 59.05 |
| fair_final_seed7102_best | fixed | 20 | 53/200 | 0.7028 | 59.05 |
| fair_final_seed7102_best | fixed | 50 | 99/200 | 0.8766 | 59.05 |
| fair_final_seed7102_best | fixed | 100 | 163/200 | 0.9686 | 59.05 |
| fair_final_seed7102_final | locked dynamic | 50 | 94/200 | 0.8694 | 59.32 |
| fair_final_seed7102_final | locked dynamic | 75 | 132/200 | 0.9308 | 59.32 |
| fair_final_seed7102_final | locked dynamic | 100 | 160/200 | 0.9606 | 59.32 |
| fair_final_seed7102_final | fixed | 20 | 51/200 | 0.7022 | 59.32 |
| fair_final_seed7102_final | fixed | 50 | 100/200 | 0.8835 | 59.32 |
| fair_final_seed7102_final | fixed | 100 | 160/200 | 0.9643 | 59.32 |

## Paired control versus final FullCov

Wins are queries covered only by final; losses are queries covered only by control. The exact two-sided McNemar p-value pools both training replicates and both locked query seeds.

| Retrieval | Budget | Final wins | Final losses | Exact p-value |
| --- | ---: | ---: | ---: | ---: |
| locked dynamic | 50 | 40 | 24 | 0.05994 |
| locked dynamic | 75 | 52 | 36 | 0.1093 |
| locked dynamic | 100 | 32 | 26 | 0.5118 |
| fixed | 20 | 25 | 8 | 0.004551 |
| fixed | 50 | 58 | 19 | 9.775e-06 |
| fixed | 100 | 86 | 22 | 3.988e-10 |

## Query feasibility

| K | Impossible queries | Avg true partitions | Max true partitions |
| ---: | ---: | ---: | ---: |
| 20 | 0/200 | 6.37 | 17 |
| 50 | 0/200 | 6.37 | 17 |
| 100 | 0/200 | 6.37 | 17 |

Retrieval latency is not present in these CSVs and must not be claimed from this evaluation.
