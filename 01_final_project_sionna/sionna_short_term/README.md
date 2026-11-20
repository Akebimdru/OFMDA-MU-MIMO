# Long-term + Short-term 채널 시뮬레이션 (Sionna RT, CPU)

`sionna_project`(과제 모델, Python 버전과 같은 숫자)에 **short-term 요소**를 더한 버전입니다.
기존 `sionna_project` 폴더는 건드리지 않았고, 코드를 복사해 이 폴더에서 확장했습니다.

교수님 지적 사항:
1. 과제 문서의 채널은 경로 손실·장애물·벽 반사 같은 **long-term** 요소뿐이다.
2. 작은 기류 변화, 벌레 등에 의한 **short-term 요동을 가우시안으로** 넣어 볼 것.
3. long-term과 short-term을 **Sionna ray tracing으로 CPU에서** 시뮬레이션할 것.

## 1. 실행

설치는 `sionna_project/README.md`와 같습니다 (Python 3.11+, `pip install -r requirements.txt`, Windows는 LLVM 설치 후 `DRJIT_LIBLLVM_PATH` 설정).

```
cd code
python st_main.py            # 전체 (30회 실현, CPU 약 3~4분)
python st_main.py --quick    # 실현 수를 줄인 빠른 확인
```

Visual Studio에서는 `code/st_main.py`를 시작 파일로 지정하고 작업 디렉터리를 `code`로 두면 됩니다.
결과 그림과 숫자는 `figures/`에 저장됩니다 (`st_results.json`에 모든 수치).
기존 과제 결과(Task 1~5)는 `python main.py`로 그대로 재현됩니다.

## 2. 채널 모델

| 성분 | 무엇을 나타내나 | 어떻게 계산하나 |
|---|---|---|
| **Long-term** | 경로 손실, 장애물 차폐, 벽 반사 (수 m 규모) | 0.5 m 격자 fingerprint (과제 모델 그대로). 비교용으로 경로 전력의 비동기 합 Σ\|a_i\|² (국소 평균)도 계산 |
| **Short-term (결정적)** | 다중경로 간섭에 의한 λ(5.2 cm) 규모 fading | AMR이 실제로 지나는 **정확한 위치마다 Sionna RT(PathSolver)를 다시 실행**하고 LOS + 벽 반사 4개를 위상까지 coherent하게 합성 |
| **Short-term (가우시안) (a)** | 기류·온도 변화로 생기는 경로 길이·진폭의 미세 요동 | 경로마다 길이 요동 Δd ~ N(0, 2 mm²), 진폭 요동 1+N(0, 0.05²). Gauss-Markov 시간 상관 (T_c = 0.5 s). 같은 물리 경로는 4개 안테나에서 같은 요동 |
| **Short-term (가우시안) (b)** | 벌레, 작은 물체 등 작은 산란체 | 복소 가우시안 산란체 20개 (확산 성분). Rician K = 10 dB, 지연 Exp(15 ns), 무작위 출발각 → ULA 위상 e^{jπn cosφ}, Gauss-Markov (T_c = 0.2 s) |

최종 채널 (안테나 n, 부반송파 k, 시각 t):

```
h_{n,k}(t) = Σ_i a_i (1 + σ_A x_i(t)) e^{-j2π f_k (τ_i + Δd_i(t)/c)}             ← RT 경로 + 가우시안 요동
           + Σ_s g_s(t) √(P_RT / (K·N_s)) e^{jπ n cos φ_s} e^{-j2π f_k τ_s}      ← 가우시안 확산 성분
```

주파수 응답은 Sionna PHY의 `cir_to_ofdm_channel`로 계산합니다.
경로 이득은 long-term 모델과 똑같이 handout 식(ρ=0.7, α·관통 길이)을 씁니다.

**경로 계획은 그대로 long-term fingerprint로 하고**, 실행할 때의 채널만 long-term + short-term으로 바꿔서
"계획은 대규모 지도로, 실제로는 작은 요동까지 겪는" 상황을 평가합니다.
대역 분할은 두 가지로 봅니다.
- **오프라인 k\*** : long-term으로 미리 정한 k\*(t)를 그대로 사용.
- **적응 k\*** : 순간 채널로 k\*를 다시 계산 (Opt-BW scheme만 차이가 남).

### 가우시안 파라미터는 가정입니다
위 숫자들(2 mm, 0.05, K=10 dB, T_c 등)은 실내 5.8 GHz에서 흔히 쓰는 범위로 제가 정한 가정입니다.
`code/config.py` 맨 아래 `ST_*` 상수를 바꾸면 됩니다. K가 결과에 가장 큰 영향을 주므로 K를 바꿔 가며 본 결과(§3.4)도 함께 넣었습니다.

## 3. 결과 (QoS = 70 Mbps, 30회 실현)

### 3.1 8개 scheme: long-term만 vs long-term + short-term

R_min [Mbps] (괄호는 실현 중 하위 10%), outage = R(t) < 70 Mbps인 시간 비율

| Scheme | Long-term만 (과제 결과) | + RT 다중경로 | + RT + 가우시안 (오프라인 k\*) | + 가우시안 (적응 k\*) | Outage |
|---|---|---|---|---|---|
| NAIVE | 12.85 | 0.51 | 2.29 (0.69) | 2.29 | 30.8% |
| +MRC | 41.44 | 2.77 | 9.74 | 9.74 | 13.3% |
| +OPTBW | 27.73 | 1.44 | 4.96 | 6.85 | 23.6% |
| +SMARTPATH | 14.56 | 1.94 | 4.96 | 4.96 | 27.9% |
| MRC OPTBW | 64.30 | 5.37 | 17.75 | 22.00 | 5.4% |
| MRC SMART | 63.12 | 53.56 | 54.46 | 54.46 | 1.3% |
| OPT SMART | 31.74 | 2.41 | 5.66 | 6.83 | 18.1% |
| **JOINT** | **81.61** | 40.55 | **69.62 (65.35)** | **75.14** | **0.2%** |

- **평균** 처리율은 거의 그대로이고, short-term 요소는 주로 **최악값(R_min)**을 깎습니다.
- 차단 구역(틈 동쪽 그늘)을 지나는 scheme은 경로 5개의 세기가 비슷해서 정확한 위치에서 깊은 null이 생깁니다. 0.5 m 격자를 dB로 보간하면 이 null이 가려집니다.
- 가우시안 성분을 넣으면 RT만 쓸 때보다 R_min이 오히려 올라갑니다. 완벽한 상쇄로 생긴 null이 작은 요동에 쉽게 메워지기 때문입니다.
- **JOINT는 그늘을 피하는 경로를 고르기 때문에 short-term을 넣어도 outage 0.2%로 가장 강건합니다.** 적응 k\*를 쓰면 75 Mbps로 QoS를 넘습니다.

### 3.2 Short-term 크기
- 격자 전체에서 coherent − long-term: 평균 −0.9 dB, 표준편차 3.4 dB, 최소 −23 dB.
- JOINT AMR1 경로를 따라 다중경로 성분 표준편차 3.9 dB. 격자 보간값과 정확한 RT 값의 RMS 차이 3.9 dB.

### 3.3 미세 구간 (2 m를 2 mm 간격으로)
- 상관 거리 (ACF = 0.5): 12 mm ≈ 0.23 λ.
- Short-term dB 표준편차: RT만 2.4 dB, RT + 가우시안 3.3 dB.
- P(−10 dB 미만): RT만 0.3%, RT + 가우시안 1.6%. Rayleigh보다 완만한 Rician형 분포입니다.

### 3.4 K(확산 성분 세기) 바꿔 보기, E[R_min] [Mbps]
K = 40 dB(가우시안 거의 없음) → 0 dB(확산 성분이 RT 경로만큼 큼)
- JOINT: 56.1 → 68.3 (K=5 dB에서 최대 70.5), outage ≤ 0.9%
- +MRC: 4.6 → 27.8
- NAIVE: 0.7 → 5.0

## 4. 그림 (`figures/`)

| 파일 | 내용 |
|---|---|
| st_01_long_vs_short_map.png | 격자 SNR: long-term(국소 평균) / coherent / 둘의 차이(short-term) |
| st_02_decomposition_along_path.png | JOINT AMR1 경로를 따라 long-term, RT 다중경로, + 가우시안 분해 |
| st_03_fine_fading_statistics.png | 2 mm 간격 미세 구간의 fading, CDF, 자기상관 |
| st_04_gaussian_only_at_rest.png | AMR이 멈춰 있을 때 가우시안 성분만으로 생기는 시간 요동 |
| st_05_schemes_long_vs_short.png | 8개 scheme의 R_min, outage 비교 |
| st_06_throughput_traces.png | 시간에 따른 R(t): long-term vs long+short |
| st_07_k_sweep.png | K에 따른 R_min, outage |

## 5. 코드에서 바뀐 곳

| 파일 | 변경 |
|---|---|
| `short_term.py` (새 파일) | RT 위치별 경로 추적, Gauss-Markov 가우시안 요동, 확산 성분, 채널 합성, scheme 평가 |
| `st_main.py` (새 파일) | 위 7개 실험 실행 및 그림 저장 |
| `fingerprint.py` | `rt_paths_at()` 추가: 임의 위치 목록에 Sionna RT를 나눠 실행 (기존 함수는 그대로) |
| `config.py` | 맨 아래 `ST_*` 파라미터 추가 |

나머지 파일은 `sionna_project`와 같습니다.
