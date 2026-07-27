# Grid: полные результаты всех конфигов

Hard-сплит (`sapientinc/sudoku-extreme`, Radcliffe-like), тот же чекпоинт 6M, без переобучения.
Сырые JSON — в `final/`, `n2000/`. `vbon_<база>_n<N>` = verifier-bon над базой с N сэмплов;
`ens_*` = смешанная база; `*_beam<k>` = ширина beam у searchdiff.

Пересчитано 2026-07-27 из текущих JSON (после починки бага в beam-поиске searchdiff, см. README).
Предыдущая версия этого файла (до 2026-07-23) содержала числа ДО починки бага и была неверна
для всех searchdiff-производных строк.

## Все 9 методов на лучших настройках, N=10000 (финальный прогон)

| # | конфиг | точность % | время (с) |
|---:|---|---:|---:|
| 1 | vbon_searchdiff_beam8_n16 | 99.84 | 9042 |
| 2 | searchdiff_beam16 | 93.86 | 1041 |
| 3 | vbon_adaptive_n16 | 88.84 | 3792 |
| 4 | searchdiff_beam8 | 87.76 | 533 |
| 5 | remdm_324 | 44.94 | 1126 |
| 6 | margin_ds81 | 43.64 | 292 |
| 7 | guided | 43.52 | 375 |
| 8 | adaptive | 42.06 | 299 |
| 9 | stochastic | 4.89 | 90 |

## Grid гиперпараметров и комбинаций, N=48 конфигов, N=2000 паззлов (по убыванию точности)

| # | конфиг | точность % | время (с) |
|---:|---|---:|---:|
| 1 | vbon_sd_beam8_n16 | 99.75 | 1811 |
| 2 | vbon_searchdiff_n32 | 99.65 | 1893 |
| 3 | vbon_sd_beam8_n8 | 99.45 | 909 |
| 4 | vbon_searchdiff_n16 | 99.15 | 945 |
| 5 | vbon_searchdiff_n8 | 97.95 | 472 |
| 6 | ens_sdadre_n32 | 96.80 | 5064 |
| 7 | ens_all5_n32 | 95.85 | 3444 |
| 8 | vbon_remdm_n32 | 95.10 | 2998 |
| 9 | vbon_adaptive_n32 | 94.20 | 2147 |
| 10 | vbon_searchdiff_n4 | 94.10 | 236 |
| 11 | ens_sdadre_n16 | 93.85 | 2632 |
| 12 | vbon_sd_beam2_n8 | 92.70 | 258 |
| 13 | vbon_remdm_n16 | 91.25 | 1729 |
| 14 | vbon_guided_n16 | 89.35 | 988 |
| 15 | ens_sdadre_n8 | 89.20 | 1274 |
| 16 | vbon_adaptive_n16 | 88.75 | 1079 |
| 17 | vbon_margin_n32 | 88.00 | 575 |
| 18 | searchdiff_b8_c3 | 87.30 | 107 |
| 19 | vbon_remdm_n8 | 83.60 | 1034 |
| 20 | vbon_adaptive_n8 | 81.15 | 515 |
| 21 | vbon_guided_n8 | 81.10 | 722 |
| 22 | searchdiff_b8_c2 | 80.15 | 107 |
| 23 | vbon_margin_n16 | 79.85 | 275 |
| 24 | vbon_remdm_n4 | 73.20 | 520 |
| 25 | searchdiff_b4_c3 | 72.90 | 56 |
| 26 | searchdiff_b4_c2 | 71.30 | 56 |
| 27 | vbon_adaptive_n4 | 69.80 | 233 |
| 28 | vbon_margin_n8 | 68.20 | 130 |
| 29 | vbon_stochastic_n32 | 65.95 | 527 |
| 30 | searchdiff_b2_c2 | 55.30 | 31 |
| 31 | searchdiff_b2_c3 | 55.30 | 31 |
| 32 | vbon_margin_n4 | 53.65 | 69 |
| 33 | vbon_stochastic_n16 | 47.75 | 276 |
| 34 | remdm_324 | 45.45 | 200 |
| 35 | remdm_162 | 44.45 | 94 |
| 36 | margin_ds81 | 44.25 | 44 |
| 37 | remdm_81 | 43.90 | 53 |
| 38 | guided_lr05 | 43.20 | 69 |
| 39 | adaptive_rps1 | 41.80 | 49 |
| 40 | margin_ds64 | 41.60 | 37 |
| 41 | guided_lr10 | 40.80 | 66 |
| 42 | adaptive_rps2 | 35.85 | 28 |
| 43 | guided_lr20 | 30.35 | 67 |
| 44 | margin_ds20 | 29.45 | 19 |
| 45 | vbon_stochastic_n8 | 29.05 | 145 |
| 46 | adaptive_rps4 | 28.75 | 16 |
| 47 | vbon_stochastic_n4 | 15.50 | 74 |
| 48 | stochastic | 5.20 | 18 |
