# OFDMA / MU-MIMO for multi-AMR indoor networks (26-2 lab internship)

Sionna RT based simulations of subcarrier allocation (OFDMA, MU-MIMO ZF, hybrid 2x2) and path planning for 4 AMRs.

| Folder | Contents |
|---|---|
| `code/`, `figures/`, `build_report.py` | Original final project (assignment model) |
| `sionna_project/` | Sionna version of the final project (Task 1-5) |
| `sionna_short_term/` | Gaussian + RT multipath short-term version |
| `sionna_map_python_algo/` | Sionna map + pure-Python algorithm version |
| `sionna_fullphysics_wip/` | Full-physics RT variant (work in progress, paused) |
| `sionna_4amr_hybrid/` | 4-AMR map A: OFDMA / MU-MIMO / hybrid comparison |
| `new_map_candidates/` | Map redesign candidates (A, B, C) with fast proxy |
| `sionna_mapB/`, `sionna_mapC/` | Maps B and C: 10 cm RT, CSI-error comparison |
| `ofdma_warmstart/` | Warm-start OFDMA band allocation |
| `output/docx/` | Reports |

`PROJECT_README.md` is the original top-level README.

Not included (see `.gitignore`): lecture and paper PDFs (copyright), `__pycache__`, and `.npy`/`.npz` simulation data (~660 MB; regenerate with each folder's code).
