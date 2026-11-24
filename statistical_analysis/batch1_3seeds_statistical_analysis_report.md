# Statistical analysis report — Batch 1 (seeds 42, 123, 2024)
## Data provenance
- Source (read-only): `results/multiseed_20261027/studies/**/summary.json`
- Curated export: `results/batch1_seeds_42_123_2024/`
- Algorithms: Standard TPE, Multivariate TPE, Gaussian Process
- Seeds: 42, 123, 2024 (paired)
- Datasets: 15 (9 classification + 6 regression)
- Generators per dataset: 7 (forest_diffusion excluded)
- Objective: maximise $J = 0.4\cdot\mathrm{Fidelity} + 0.4\cdot\mathrm{Utility} - 0.2\cdot\mathrm{Privacy\ risk}$
### Aggregation definition
For each (dataset, algorithm, seed), the seed-level score is the **mean** of the seven generator-specific study best objectives. The dataset-level score is then the **mean of the three seed-level scores**. Primary comparisons use this dataset-level mean (not the max across seeds).
### Validation notes
- Loaded rows: 945 / expected 945
-   Standard TPE seed=42: 105/105 studies with best J
-   Standard TPE seed=123: 105/105 studies with best J
-   Standard TPE seed=2024: 105/105 studies with best J
-   Multivariate TPE seed=42: 105/105 studies with best J
-   Multivariate TPE seed=123: 105/105 studies with best J
-   Multivariate TPE seed=2024: 105/105 studies with best J
-   Gaussian Process seed=42: 105/105 studies with best J
-   Gaussian Process seed=123: 105/105 studies with best J
-   Gaussian Process seed=2024: 105/105 studies with best J
- All 15 datasets × 3 algorithms have 3 seed-level scores — OK for Friedman.

## Descriptive statistics
| algorithm        | sampler          |   n_datasets |     mean |       std |   median |      min |      max |       iqr |       q1 |       q3 |   average_rank |   n_wins_exclusive_rank1 |   n_wins_including_ties |
|:-----------------|:-----------------|-------------:|---------:|----------:|---------:|---------:|---------:|----------:|---------:|---------:|---------------:|-------------------------:|------------------------:|
| Standard TPE     | standard_tpe     |           15 | 0.572234 | 0.0515008 | 0.581598 | 0.434117 | 0.638219 | 0.0536963 | 0.553215 | 0.606911 |        2.4     |                        2 |                       2 |
| Multivariate TPE | multivariate_tpe |           15 | 0.57428  | 0.0512226 | 0.598508 | 0.440461 | 0.637047 | 0.0570948 | 0.550743 | 0.607838 |        2.13333 |                        3 |                       3 |
| Gaussian Process | gp               |           15 | 0.578203 | 0.0509819 | 0.598334 | 0.437755 | 0.639396 | 0.0573189 | 0.556269 | 0.613588 |        1.46667 |                       10 |                      10 |

## Friedman test
- χ² = 6.9333, df = 2, p = 0.0312209
- Kendall's W = 0.2311
- Significant at α=0.05: **True**
- Note: Primary unit = dataset-level mean of 3 seed means (each seed mean = average best-J across 7 generators).

## Pairwise Wilcoxon signed-rank (Holm-corrected)
| comparison                           | algo_a           | algo_b           |   mean_a |   mean_b |   mean_delta_a_minus_b |   n_a_better |   n_b_better |   n_ties |   wilcoxon_stat |     p_raw |    p_holm | significant_holm_0.05   | direction                                  |
|:-------------------------------------|:-----------------|:-----------------|---------:|---------:|-----------------------:|-------------:|-------------:|---------:|----------------:|----------:|----------:|:------------------------|:-------------------------------------------|
| Standard TPE vs Multivariate TPE     | Standard TPE     | Multivariate TPE | 0.572234 | 0.57428  |            -0.00204568 |            6 |            9 |        0 |              42 | 0.330261  | 0.330261  | False                   | Multivariate TPE higher mean (numerically) |
| Standard TPE vs Gaussian Process     | Standard TPE     | Gaussian Process | 0.572234 | 0.578203 |            -0.00596876 |            3 |           12 |        0 |              19 | 0.0180664 | 0.0541992 | False                   | Gaussian Process higher mean (numerically) |
| Multivariate TPE vs Gaussian Process | Multivariate TPE | Gaussian Process | 0.57428  | 0.578203 |            -0.00392308 |            4 |           11 |        0 |              19 | 0.0180664 | 0.0541992 | False                   | Gaussian Process higher mean (numerically) |

## Average ranks
| sampler          | algorithm        |   average_rank |
|:-----------------|:-----------------|---------------:|
| gp               | Gaussian Process |        1.46667 |
| multivariate_tpe | Multivariate TPE |        2.13333 |
| standard_tpe     | Standard TPE     |        2.4     |

## Seed-level consistency (lower seed-std = more consistent)
| algorithm        | sampler          |   mean_of_dataset_seed_means |   mean_of_dataset_seed_stds |   median_of_dataset_seed_stds |   mean_cv |   median_cv |   n_datasets |
|:-----------------|:-----------------|-----------------------------:|----------------------------:|------------------------------:|----------:|------------:|-------------:|
| Standard TPE     | standard_tpe     |                     0.572234 |                  0.00909261 |                    0.00633358 | 0.0158695 |  0.0108477  |           15 |
| Multivariate TPE | multivariate_tpe |                     0.57428  |                  0.00565777 |                    0.0045946  | 0.0100549 |  0.00785164 |           15 |
| Gaussian Process | gp               |                     0.578203 |                  0.00593079 |                    0.00340546 | 0.0109527 |  0.00606811 |           15 |

## Research interpretation
- Highest mean objective: **Gaussian Process** (mean = 0.5782).
- Lowest average rank: **Gaussian Process** (average rank = 1.467).
- Friedman significant: **True** (p = 0.03122).
- **No pairwise comparison remains significant after Holm correction.** Numerical differences should not be interpreted as statistically confirmed superiority.
- Most consistent across seeds (lowest mean per-dataset seed std): **Multivariate TPE** (mean seed-std = 0.0057).
- Datasets where GP numerically beats both TPE variants: cancer, alzheimers, forest_cover, bank_marketing, cdc_diabetes, mushroom, magic_gamma, metro_interstate, concrete, real_estate.
- Datasets where Multivariate TPE numerically beats Standard TPE: 9/15 (cancer, alzheimers, forest_cover, wine_quality, cdc_diabetes, mushroom, metro_interstate, online_shopping, air_quality).

## Limitations
- Only **three** seeds; variability estimates and Wilcoxon/Friedman power are limited.
- Seed 42 Standard/Multivariate TPE include some imported legacy studies with fewer than 20 completed trials (partial budgets); best-J still recorded.
- Dataset-level scores average across seven generators; generator-specific behaviour may differ.
- Multiple testing corrected with Holm; absence of significance is not evidence of equivalence.

## Results and Discussion (paper-ready draft)

We compared three Bayesian optimisation algorithms—Standard TPE, Multivariate TPE and Gaussian Process (GP) sampling—on fifteen tabular datasets (nine classification and six regression). Each algorithm was executed with three paired random seeds (42, 123 and 2024). For every dataset–algorithm–seed combination, performance was summarised as the mean of the best objective values obtained across seven synthetic-data generators; dataset-level scores were then obtained by averaging the three seed-level summaries. The composite objective maximises fidelity and downstream utility while penalising privacy risk.

Descriptively, Gaussian Process attained the highest mean dataset-level objective (0.5782), followed by the remaining methods (Standard TPE: 0.5722; Multivariate TPE: 0.5743; Gaussian Process: 0.5782). Average ranks across datasets likewise favoured Gaussian Process (average rank 1.467). These numerical patterns indicate a modest ordering among samplers but must be interpreted jointly with formal hypothesis tests.

A Friedman test on the fifteen dataset-level means did detect a statistically significant difference among the three algorithms at α = 0.05 (χ² = 6.933, df = 2, p = 0.03122; Kendall's W = 0.231). Pairwise Wilcoxon signed-rank tests with Holm correction did not identify any significant pairwise differences. Accordingly, while one method may appear numerically preferable on mean objective or average rank, the present evidence does **not** support claiming statistically significant superiority of any single optimiser over the others under the primary analysis protocol.

Seed-level consistency was summarised by the mean of per-dataset standard deviations across the three seeds. Multivariate TPE exhibited the smallest average seed dispersion. Lower variability did not necessarily coincide with the highest mean performance, underscoring that stability and peak objective are distinct properties in this setting.

Dataset-wise inspection showed that Gaussian Process numerically outperformed both TPE variants on 10 dataset(s): cancer, alzheimers, forest_cover, bank_marketing, cdc_diabetes, mushroom, magic_gamma, metro_interstate, concrete, real_estate. Multivariate TPE exceeded Standard TPE on 9 of 15 datasets. These local wins illustrate heterogeneity across data regimes; however, after Holm correction no pairwise contrast reaches significance, so omnibus evidence of a difference among methods should not be over-interpreted as confirming a uniquely superior optimiser.

Several limitations apply. With only three seeds, estimates of variability and the power of rank-based tests are constrained. A subset of seed-42 TPE studies originates from earlier runs with incomplete trial budgets, although best objectives were retained. Finally, averaging across generators emphasises overall behaviour and may mask generator-specific optimiser advantages. Future work with additional seeds and generator-stratified analyses would strengthen inferential claims.

