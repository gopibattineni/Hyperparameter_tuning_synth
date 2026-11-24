# Statistical analysis report — 6 seeds (42, 123, 2024, 7, 99, 1337)
## Data provenance
- Source (read-only): `results/multiseed_20261027/studies/**/summary.json`
- Curated export: `results/multiseed_6seeds_42_123_2024_7_99_1337/`
- Algorithms: Standard TPE, Multivariate TPE, Gaussian Process
- Seeds: 42, 123, 2024, 7, 99, 1337 (paired)
- Datasets: 15 (9 classification + 6 regression)
- Generators per dataset: 7 (forest_diffusion excluded)
- Expected studies: 1890 (= 15×7×3×6)
- Objective: maximise $J = 0.4\cdot\mathrm{Fidelity} + 0.4\cdot\mathrm{Utility} - 0.2\cdot\mathrm{Privacy\ risk}$
### Aggregation definition
For each (dataset, algorithm, seed), the seed-level score is the **mean** of the seven generator-specific study best objectives. The dataset-level score is then the **mean of the 6 seed-level scores**. Primary comparisons use this dataset-level mean (not the max across seeds).
### Validation notes
- Loaded rows: 1890 / expected 1890
-   Standard TPE seed=42: 105/105 studies with best J
-   Standard TPE seed=123: 105/105 studies with best J
-   Standard TPE seed=2024: 105/105 studies with best J
-   Standard TPE seed=7: 105/105 studies with best J
-   Standard TPE seed=99: 105/105 studies with best J
-   Standard TPE seed=1337: 105/105 studies with best J
-   Multivariate TPE seed=42: 105/105 studies with best J
-   Multivariate TPE seed=123: 105/105 studies with best J
-   Multivariate TPE seed=2024: 105/105 studies with best J
-   Multivariate TPE seed=7: 105/105 studies with best J
-   Multivariate TPE seed=99: 105/105 studies with best J
-   Multivariate TPE seed=1337: 105/105 studies with best J
-   Gaussian Process seed=42: 105/105 studies with best J
-   Gaussian Process seed=123: 105/105 studies with best J
-   Gaussian Process seed=2024: 105/105 studies with best J
-   Gaussian Process seed=7: 105/105 studies with best J
-   Gaussian Process seed=99: 105/105 studies with best J
-   Gaussian Process seed=1337: 105/105 studies with best J
- All 15 datasets × 3 algorithms have 6 seed-level scores — OK for Friedman.

## Descriptive statistics
| algorithm        | sampler          |   n_datasets |     mean |       std |   median |      min |      max |       iqr |       q1 |       q3 |   average_rank |   n_wins_exclusive_rank1 |   n_wins_including_ties |
|:-----------------|:-----------------|-------------:|---------:|----------:|---------:|---------:|---------:|----------:|---------:|---------:|---------------:|-------------------------:|------------------------:|
| Standard TPE     | standard_tpe     |           15 | 0.575234 | 0.0503474 | 0.594201 | 0.43807  | 0.637783 | 0.0545849 | 0.553425 | 0.60801  |        2.33333 |                        3 |                       3 |
| Multivariate TPE | multivariate_tpe |           15 | 0.575745 | 0.0498487 | 0.597466 | 0.441081 | 0.635699 | 0.0542782 | 0.553825 | 0.608103 |        2.13333 |                        2 |                       2 |
| Gaussian Process | gp               |           15 | 0.579192 | 0.0501317 | 0.598663 | 0.440398 | 0.640422 | 0.055487  | 0.557143 | 0.61263  |        1.53333 |                       10 |                      10 |

## Friedman test
- χ² = 5.2000, df = 2, p = 0.0742736
- Kendall's W = 0.1733
- Significant at α=0.05: **False**
- Note: Primary unit = dataset-level mean of 6 seed means (each seed mean = average best-J across 7 generators).

## Pairwise Wilcoxon signed-rank (Holm-corrected)
| comparison                           | algo_a           | algo_b           |   mean_a |   mean_b |   mean_delta_a_minus_b |   n_a_better |   n_b_better |   n_ties |   wilcoxon_stat |     p_raw |    p_holm | significant_holm_0.05   | direction                                  |
|:-------------------------------------|:-----------------|:-----------------|---------:|---------:|-----------------------:|-------------:|-------------:|---------:|----------------:|----------:|----------:|:------------------------|:-------------------------------------------|
| Standard TPE vs Multivariate TPE     | Standard TPE     | Multivariate TPE | 0.575234 | 0.575745 |           -0.000511045 |            7 |            8 |        0 |              48 | 0.524475  | 0.524475  | False                   | Multivariate TPE higher mean (numerically) |
| Standard TPE vs Gaussian Process     | Standard TPE     | Gaussian Process | 0.575234 | 0.579192 |           -0.0039581   |            3 |           12 |        0 |              19 | 0.0180664 | 0.0361328 | True                    | Gaussian Process higher mean (numerically) |
| Multivariate TPE vs Gaussian Process | Multivariate TPE | Gaussian Process | 0.575745 | 0.579192 |           -0.00344706  |            5 |           10 |        0 |              16 | 0.0102539 | 0.0307617 | True                    | Gaussian Process higher mean (numerically) |

## Average ranks
| sampler          | algorithm        |   average_rank |
|:-----------------|:-----------------|---------------:|
| gp               | Gaussian Process |        1.53333 |
| multivariate_tpe | Multivariate TPE |        2.13333 |
| standard_tpe     | Standard TPE     |        2.33333 |

## Seed-level consistency (lower seed-std = more consistent)
| algorithm        | sampler          |   mean_of_dataset_seed_means |   mean_of_dataset_seed_stds |   median_of_dataset_seed_stds |   mean_cv |   median_cv |   n_datasets |
|:-----------------|:-----------------|-----------------------------:|----------------------------:|------------------------------:|----------:|------------:|-------------:|
| Standard TPE     | standard_tpe     |                     0.575234 |                  0.00697057 |                    0.00502784 | 0.0122505 |  0.00935313 |           15 |
| Multivariate TPE | multivariate_tpe |                     0.575745 |                  0.00652062 |                    0.00639224 | 0.0117354 |  0.0106053  |           15 |
| Gaussian Process | gp               |                     0.579192 |                  0.00559815 |                    0.00418232 | 0.0101513 |  0.00742755 |           15 |

## Research interpretation
- Highest mean objective: **Gaussian Process** (mean = 0.5792).
- Lowest average rank: **Gaussian Process** (average rank = 1.533).
- Friedman significant: **False** (p = 0.07427).
- Significant pairwise comparisons after Holm correction:
  - Standard TPE vs Gaussian Process: p_Holm = 0.03613; Gaussian Process higher mean (numerically)
  - Multivariate TPE vs Gaussian Process: p_Holm = 0.03076; Gaussian Process higher mean (numerically)
- Most consistent across seeds (lowest mean per-dataset seed std): **Gaussian Process** (mean seed-std = 0.0056).
- Datasets where GP numerically beats both TPE variants: cancer, alzheimers, forest_cover, bank_marketing, cdc_diabetes, mushroom, magic_gamma, metro_interstate, air_quality, real_estate.
- Datasets where Multivariate TPE numerically beats Standard TPE: 8/15 (cancer, alzheimers, forest_cover, wine_quality, cdc_diabetes, mushroom, metro_interstate, online_shopping).

## Limitations
- **6** seeds improve variability estimates relative to a 3-seed pilot, but Wilcoxon/Friedman power on 15 datasets remains modest.
- All reported studies target 20 completed trials per (dataset, generator, sampler, seed).
- Dataset-level scores average across seven generators; generator-specific behaviour may differ.
- Multiple testing corrected with Holm; absence of significance is not evidence of equivalence.

## Results and Discussion (paper-ready draft)

We compared three Bayesian optimisation algorithms—Standard TPE, Multivariate TPE and Gaussian Process (GP) sampling—on fifteen tabular datasets (nine classification and six regression). Each algorithm was executed with 6 paired random seeds (42, 123, 2024, 7, 99, 1337). For every dataset–algorithm–seed combination, performance was summarised as the mean of the best objective values obtained across seven synthetic-data generators; dataset-level scores were then obtained by averaging the 6 seed-level summaries. The composite objective maximises fidelity and downstream utility while penalising privacy risk.

Descriptively, Gaussian Process attained the highest mean dataset-level objective (0.5792), followed by the remaining methods (Standard TPE: 0.5752; Multivariate TPE: 0.5757; Gaussian Process: 0.5792). Average ranks across datasets likewise favoured Gaussian Process (average rank 1.533). These numerical patterns indicate a modest ordering among samplers but must be interpreted jointly with formal hypothesis tests.

A Friedman test on the fifteen dataset-level means did not detect a statistically significant difference among the three algorithms at α = 0.05 (χ² = 5.200, df = 2, p = 0.07427; Kendall's W = 0.173). Pairwise Wilcoxon signed-rank tests with Holm correction identified significant pairwise differences as tabulated. Accordingly, while one method may appear numerically preferable on mean objective or average rank, the present evidence supports pairwise claims only where Holm-adjusted tests are significant under the primary analysis protocol.

Seed-level consistency was summarised by the mean of per-dataset standard deviations across the 6 seeds. Gaussian Process exhibited the smallest average seed dispersion. Lower variability did not necessarily coincide with the highest mean performance, underscoring that stability and peak objective are distinct properties in this setting.

Dataset-wise inspection showed that Gaussian Process numerically outperformed both TPE variants on 10 dataset(s): cancer, alzheimers, forest_cover, bank_marketing, cdc_diabetes, mushroom, magic_gamma, metro_interstate, air_quality, real_estate. Multivariate TPE exceeded Standard TPE on 8 of 15 datasets. These local wins illustrate heterogeneity across data regimes; however, Holm-adjusted pairwise tests should be read jointly with the omnibus Friedman result.

Several limitations apply. With 6 seeds, variability estimates improve relative to a three-seed pilot, but power of rank-based tests on fifteen datasets remains limited. All studies target a 20-trial budget. Finally, averaging across generators emphasises overall behaviour and may mask generator-specific optimiser advantages. Generator-stratified analyses would further strengthen inferential claims.

