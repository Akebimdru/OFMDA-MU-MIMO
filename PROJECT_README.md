# Indoor Multi-AMR Final Project

## Stage 0: project contract

- `code/config.py`: all scene, PHY, simulation, QoS, and reproducibility constants.
- `code/fingerprint.py`: Task 1.1 ray tracer and cache (`fingerprint.npy`).
- `code/main.py`: one command entry point; currently runs Stage 0 and Task 1.1 validation.
- `code/visualizer.py`: Task 1.2-1.3 SISO/MRC maps and numerical validation.
- `code/path_planner.py`: Task 2.1 A* and Task 2.2 reproducible 30-path candidate sets.
- `code/simulator.py`: Task 3.1 trajectory sampling, dB bilinear interpolation, and vectorised Opt-BW.
- `code/joint_planner.py`: Task 2.3 900-pair QoS-constrained SmartPath and JOINT selection.
- `code/ablation.py`: Task 3.3 complete eight-scheme Path/BW/Antenna ablation.
- `code/tradeoff_analysis.py`: Task 4.1 Pareto frontier and Task 4.2 specified QoS sweep.
- `code/task5.py`: Task 5 MP4 animations and fixed-schedule Rayleigh robustness replays.
- `figures/`: later PNGs and animations.
- `tests/`: unit tests added alongside subsequent tasks.

Run from the project root:

```powershell
& 'C:\Users\jhrhee\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' code/main.py
```

To generate the 36 x 20 obstacle-buffered grid and 30 candidates for each AMR:

```powershell
& 'C:\Users\jhrhee\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' code/main.py --paths
```

To verify the required Equal-BW and prefix-sum Opt-BW implementation on the shortest paths:

```powershell
& 'C:\Users\jhrhee\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' code/main.py --bandwidth
```

To run the 900-pair QoS-constrained selection and save the scheme-path comparison:

```powershell
& 'C:\Users\jhrhee\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' code/main.py --joint-plan
```

To evaluate all eight schemes and generate the ablation bars:

```powershell
& 'C:\Users\jhrhee\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' code/main.py --ablation
```

To generate the 900-point Pareto analysis and the prescribed 11-point QoS sweep:

```powershell
& 'C:\Users\jhrhee\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' code/main.py --tradeoffs
```

To generate the two MP4 animations and the 30-realisation robustness analysis:

```powershell
& 'C:\Users\jhrhee\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' code/main.py --task5
```

The first run produces `code/fingerprint.npy`; later runs load the cache.  To rebuild it after a
scene-model change, pass `--force-fingerprint`.

To also generate the required AMR-agnostic SISO/MRC maps and validation record:

```powershell
& 'C:\Users\jhrhee\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' code/main.py --channel-maps
```

## Task 1 model fidelity checklist

- 18 x 10 x 3 m room; BS at (3, 5, 2.7) m; receiver height 0.5 m.
- Exact 12 obstacles, material loss in dB/m times intersection chord length.
- Four half-wavelength ULA positions are modelled separately.
- A channel entry is the coherent complex sum of LOS plus four first-order reflections from
  perimeter walls only.  Obstacles are blockers, never reflectors.
- Link offsets (AMR1 0 dB, AMR2 -7 dB) are intentionally deferred to the simulator.
