# Cook's membrane: mean u_y of the loaded face at full load

Rows: mesh size h (TETRA10 element count in parentheses).  Columns: element (see run.jl).  Each entry is the newest record of results.tsv for that cell.

## elastic: E = 240.565, ν = 0.4999, σ_y = 1.0e10, K = 0.0, q = 6.25

| h | tet10 | tet15 | tet15-p1 | tet15-p0 | lcm-tet10 | lcm-ct |
|---|---|---|---|---|---|---|
| 8.0 (327) | 8.1960 | 8.2235 | 8.4122 | 8.5720 | 8.1960 | 8.4596 |
| 4.0 (1860) | 8.3241 | 8.3490 | 8.4551 | 8.5184 | 8.3241 | 8.4643 |

## plastic: E = 206.9, ν = 0.29, σ_y = 0.45, K = 0.12924, q = 0.14

| h | tet10 | tet15 | tet15-p1 | tet15-p0 | lcm-tet10 | lcm-ct |
|---|---|---|---|---|---|---|
| 8.0 (327) | 0.2689 | 0.2700 | 0.2706 | 0.2752 | 0.2689 | 0.2716 |
| 4.0 (1860) | 0.2710 | 0.2717 | 0.2721 | 0.2744 | 0.2710 | 0.2725 |
| 2.0 (14475) | 0.2720 | 0.2723 | 0.2724 | 0.2734 | 0.2720 | 0.2728 |

