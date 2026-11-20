# OFDMA / MU-MIMO for multi-AMR indoor networks (26-2 lab internship)

Sionna RT simulations, grouped into 4 parts.

## 01_final_project_sionna — final project re-simulated with Sionna
| Folder | Contents |
|---|---|
| `sionna_project/` | Final project (Task 1-5) rebuilt with Sionna RT (4 walls, handout gain formula) |
| `sionna_short_term/` | Short-term variation added (Gaussian path/amplitude jitter + diffuse Rician) |
| `sionna_map_python_algo/` | Full Sionna RT map (ITU materials) + original Python algorithm |
| `sionna_fullphysics_wip/` | Full-physics RT variant (work in progress, paused) |
| `report/` | Report for the Sionna-map version |

## 02_mapA — 4-AMR map A (16x8 m, BS at ceiling centre)
| Folder | Contents |
|---|---|
| `sionna_4amr_hybrid/` | OFDMA (fixed/variable/exact), MU-MIMO ZF, hybrid 2x2, collision filter, 10 cm fine map |
| `ofdma_warmstart/` | Warm-start OFDMA band allocation on map A (`mapA_old/`) |

## 03_mapB — map B (partition + gates, metal pillars)
| Folder | Contents |
|---|---|
| `sionna_mapB/` | 10 cm RT SNR map, candidate paths, 3 schemes with perfect CSI and NMSE -40 dB CSI error |
| `new_map_candidates/` | Map redesign candidates (A aisles, B partition, C wall BS) with fast proxy, B v2-v5 iterations |
| `ofdma_warmstart/` | Warm-start OFDMA band allocation on map B |

## 04_mapC — map C (BS on south wall, concrete islands, metal block)
| Folder | Contents |
|---|---|
| `sionna_mapC/` | Same analysis as map B |
| `ofdma_warmstart/` | Warm-start OFDMA band allocation on map C |

`ofdma_warm.py` is copied into each map's `ofdma_warmstart/` folder; it takes the map folder name as an argument.

Not included (see `.gitignore`): lecture and paper PDFs (copyright), `__pycache__`, and `.npy`/`.npz` simulation data (~660 MB; regenerate with each folder's code).
