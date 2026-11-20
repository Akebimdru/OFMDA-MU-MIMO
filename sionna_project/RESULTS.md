# 결과 요약 (Sionna 버전, `python main.py` 출력 기준)

모든 수치는 `figures/results_summary.json` 에서 그대로 옮겼습니다. 같은 장면을 순수 NumPy로 구현한 기존 `code/` 결과와 8개 scheme의 R_min, 평균, T, 선택 경로 쌍이 소수 둘째 자리까지 일치합니다 (독립 구현 간 교차 검증).

## Task 1 — 채널 fingerprint
| 점검 항목 | 목표 | 결과 |
|---|---|---|
| SISO SNR (3, 3) | ≈ 50 dB ± 3 | **50.16 dB** |
| SISO SNR (13, 5) | ≲ 25 dB | **13.62 dB** |
| 평균 MIMO − SISO | ≈ 6 dB | **6.43 dB** (std 2.44, 범위 1.5 ~ 21.3 dB) |
| Sionna 경로 수 / 링크 | 5 | 모든 2880개 링크에서 5 |
| Sionna 지연 vs image-source | 0 | 최대 오차 9.8e-6 m |
| Sionna LOS 진폭 vs Friis | 0 dB | 최대 7.6e-6 dB |

- **Q1 가장 강한 음영**: 가장 밝은 셀 (3.25, 4.25) 54.2 dB, 가장 어두운 셀 (13.75, 4.75) −7.9 dB, 대비 **62.1 dB**. 동쪽 절반 평균 31.7 dB vs 서쪽 45.96 dB. 동쪽은 LOS가 두께 0.3 m 콘크리트 중앙 벽을 지나야 해서 15 dB(진폭) = 30 dB(전력) 손실을 받습니다. 가장 어두운 y ≈ 5 띠는 통로를 지나는 LOS가 Pillar BS(지름 0.6 m 콘크리트, 진폭 손실 최대 30 dB)와 Pillar E에 가로막히는 영역입니다.
- **Q2 경로 위 SNR** (AMR1 최단 경로, 6점): 42.3 → 50.2 → 47.2 → **16.2 dB (10.4, 5.25)** → 30.9 → 36.0 dB. 최악 지점은 끝점이 아니라 **경로 중간**(통로 직후)이므로 우회 경로가 도움이 될 수 있습니다.
- **Q3 MRC 이득의 공간 변화**: 평균 6.43 dB, 표준편차 2.44 dB. 안테나마다 5개 경로의 위상합이 다르게 간섭해 안테나 0이 깊은 fade에 있는 셀에서는 이득이 크고(최대 21 dB), 한 경로가 지배하는 셀에서는 10log10(4) = 6 dB 근처가 됩니다.

## Task 2 — 경로 계획
- 최단 경로 길이: AMR1 = AMR2 = **18.49 m**. 두 경로 모두 중앙 통로(y ∈ [5.0, 5.5])를 지나고, 방 가장자리(y < 1, y > 9) 우회는 없습니다. 장애물 버퍼로 막힌 셀은 78개입니다.
- 후보: AMR당 30개 (seed 42, 1.5배 길이 제한, `path[::3]` 서명 중복 제거).
- JOINT(70 Mbps): AMR1은 최단 길이(18.49 m)를 유지하고 AMR2가 북쪽으로 우회합니다(22.49 m, T = 45.0 s). handout의 예상과 같습니다.

## Task 3 — 8-scheme ablation
| Scheme | Path | BW | Ant | R_min [Mbps] | mean [Mbps] | T [s] |
|---|---|---|---|---|---|---|
| NAIVE | Shortest | Equal | SISO | 12.85 | 80.05 | 37.0 |
| +MRC | Shortest | Equal | MRC | 41.44 | 103.63 | 37.0 |
| +OPTBW | Shortest | Opt | SISO | 27.73 | 98.76 | 37.0 |
| +SMARTPATH | CommAware | Equal | SISO | 14.56 | 85.43 | 44.0 |
| MRC OPTBW | Shortest | Opt | MRC | 64.30 | 123.59 | 37.0 |
| MRC SMART | CommAware | Equal | MRC | 63.12 | 114.67 | 47.6 |
| OPT SMART | CommAware | Opt | SISO | 31.74 | 97.74 | 42.6 |
| **JOINT** | CommAware | Opt | MRC | **81.61** | **127.77** | 45.0 |

- Δadd: MRC +28.59, OptBW +14.88, SmartPath +1.71 → Σ = 45.18 Mbps
- Δdrop: MRC +49.87, OptBW +18.49, SmartPath +17.31
- JOINT − NAIVE = **68.76 Mbps** > Σadd (1.52배) → **초가법적(superadditive)**. 모든 기법에서 drop > add입니다.
- 순위: MRC > OptBW > SmartPath. MRC는 log 안의 SNR 자체를 키우고(전 구간 +6 dB), OptBW는 k(대역)를 병목 AMR로 옮기며, SmartPath는 병목 구간의 위치를 바꿉니다. R = Δf·k·log2(1 + G·SNR) 에서 세 기법이 서로 다른 인자를 곱으로 키우므로 함께 쓸 때 효과가 커집니다.
- NAIVE의 R_min은 t = 17.9 s (두 AMR이 통로 근처: AMR1 (8.4, 4.9), AMR2 (9.25, 5.25))에서 생깁니다. 병목은 **궤적 중간**입니다.
- NAIVE와 +OPTBW의 차이는 t = 8.1 s에서 가장 큽니다(30.9 Mbps). 이때 AMR1은 BS 근처라 SNR이 높고 AMR2는 동쪽 음영 + −7 dB라, Opt-BW가 대부분의 부반송파를 AMR2에 줍니다.
- +SMARTPATH(SISO, Equal)는 R_min을 1.7 Mbps만 올리고 T를 7 s 늘립니다. 이 기술 조합으로는 70 Mbps를 만족하는 쌍이 없어 max-R_min fallback이 선택됩니다. 반면 평균은 80.0 → 85.4 Mbps로 오릅니다. 최악 지표와 평균 지표가 다른 결론을 낼 수 있다는 예입니다.

## Task 4 — Pareto, QoS sweep
- feasible 쌍 수: 50 Mbps **622**, 70 Mbps **72**, 80 Mbps **22** (900쌍 중).
- JOINT 점(T = 44.97 s, 81.61 Mbps)은 feasible Pareto frontier 위에 있습니다 (frontier 3점: 44.97/81.6, 46.28/83.6, 50.87/88.1). 후보가 30개뿐이라 frontier는 계단 모양입니다.
- QoS sweep (−7 dB): 10–60 Mbps에서는 최단 경로(T = 36.97 s, R_min 64.3)로 충분하고, 65 Mbps에서 AMR1이 0.3 m 늘어난 경로(T = 37.56 s)로 바뀌며, 68–80 Mbps에서는 AMR2가 22.49 m로 우회합니다(T = 44.97 s). **이 장면에서는 80 Mbps까지 infeasible 구간이 나타나지 않습니다** (900쌍 최대 R_min이 80 Mbps보다 큼). 보고서에서는 "easy → detour" 두 구간만 관찰됐다고 쓰는 것이 정확합니다.
- AMR2 오프셋 −3 dB: 최단 경로의 R_min이 79.25 Mbps라 75 Mbps까지 우회가 필요 없고, 80 Mbps에서만 AMR1이 짧게 우회합니다. 링크가 덜 약하면 우회 시작점이 높은 QoS로 밀립니다.

## Task 5 — 온라인 Rayleigh (오프라인 k* 재사용, 30회)
| Scheme | 오프라인 R_min | 온라인 E[R_min] | 비율 |
|---|---|---|---|
| NAIVE | 12.85 | 11.20 | 0.871 |
| +MRC | 41.44 | 34.80 | 0.840 |
| +OPTBW | 27.73 | 23.45 | 0.846 |
| +SMARTPATH | 14.56 | 12.10 | 0.831 |
| JOINT | 81.61 | 70.46 | 0.863 |

모든 scheme에서 비율이 1보다 작습니다. log2(1 + ρ|g|²)는 오목 함수라 Jensen 부등식으로 E[log2(1+ρ|g|²)] ≤ log2(1+ρ) 이고, R_min은 여기에 시간에 대한 min까지 더해져 손실이 커집니다. 이번 실행에서 JOINT(0.863)는 NAIVE(0.871)보다 약간 낮아 조금 더 취약했지만 차이는 작습니다. 높은 SNR에서는 ρ가 커져도 log 손실(약 0.83 bit, 오일러 상수 효과)이 거의 일정하므로 상대 손실이 줄어드는 효과와, 최적화된 계획일수록 여유가 적은 효과가 서로 상쇄된 결과로 보입니다(추론).
