# Sionna RT 맵 + 기존 Python 알고리즘

기존 순수 Python 프로젝트(`26-2 랩인턴/code`)를 그대로 복사했고, **첫 단계인 map(fingerprint) 생성만** Sionna RT로 바꿨습니다.
경로 계획(A*, 후보 30개), Equal/Opt-BW, SmartPath/JOINT 900쌍 선택, 8가지 ablation, Pareto와 QoS sweep, Task 5 애니메이션과 Rayleigh 강건성 분석은 원본 코드와 한 줄도 다르지 않습니다.

## 바뀐 파일

| 파일 | 내용 |
|---|---|
| `code/fingerprint.py` | `build_fingerprint()`를 Sionna RT와 가우시안 노이즈 버전으로 교체했습니다. 출력 형식(`H` (20,36,4,64), `x_m`, `y_m`, `occupied`, .npy 캐시)은 원본과 같습니다. 원본 해석식 추적기는 비교 그림용으로 `build_handout_fingerprint()`라는 이름으로 남겨 뒀습니다. |
| `code/sionna_scene.py` (새 파일) | handout 방 구조(둘레 벽, 바닥, 천장, 장애물 12개)를 Mitsuba XML과 PLY 메시로 만듭니다. |
| `code/sionna_figures.py` (새 파일) | Sionna 맵 관련 그림(00_*)을 그립니다. |
| `code/config.py` | 맨 아래에 `RT_*`(레이트레이싱)와 `NOISE_*`(가우시안 노이즈) 설정을 추가했습니다. 기존 상수는 그대로입니다. |
| `code/main.py` | `--sionna-figures` 옵션을 추가했고, `--tradeoffs` 단계에서 AMR2 오프셋 −3 dB QoS sweep(T4.2 질문 3용)을 하나 더 그립니다. |

나머지 `path_planner.py`, `simulator.py`, `joint_planner.py`, `ablation.py`, `tradeoff_analysis.py`, `task5.py`, `visualizer.py`는 원본과 동일합니다.

## Map 모델

1. **Sionna RT (CPU, LLVM)**: 4개 ULA 소자를 각각 송신기로, 720개 셀 중심(높이 0.5 m)을 수신기로 둡니다.
   - 모든 면에 ITU-R P.2040 재질을 씁니다. 벽, 바닥, 천장, 기둥은 콘크리트이고 캐비닛은 나무입니다.
   - 경로 하나에 상호작용을 최대 **6번**까지 허용합니다. 상호작용은 경면 반사, 굴절(투과), 회절, 확산 산란(Lambertian, S=0.2)이 섞일 수 있습니다.
   - 안테나당 레이 100만 개를 쏘고, 확산 산란은 따로 30만 개, 깊이 4로 추가 계산합니다.
   - 결과는 경로 **4,078만 개**, 링크당 평균 약 1.4만 개, 최대 3.2만 개입니다. handout의 LOS와 1차 벽 반사 4개를 합한 5개와 비교됩니다.
2. **가우시안 환경 노이즈** (기류, 벌레, 작은 움직이는 물체):
   - 경로마다 경로 길이 지터 σ = 2 mm(위상 지터)와 상대 진폭 지터 σ = 0.05를 줍니다.
   - 약한 원형 복소 가우시안 확산장을 더합니다(K = 20 dB, 안테나와 부반송파마다 독립).
   - 시드를 고정해서 매번 같은 결과가 나옵니다.
3. 모든 경로를 부반송파별로 coherent하게 더합니다.
   H[k] = Σ a_p·g_p·exp(−j2π f_k(τ_p+δτ_p))·λ_k/λ_c
   Sionna `Paths.a`에는 반송파 위상이 빠져 있어서 exp(−j2π f_k τ)로 넣었습니다.
4. 단단한 콘크리트 기둥(Pillar E) 안의 셀 1개에는 레이가 하나도 닿지 않습니다. 그래서 가장 가까운 셀 값을 복사했습니다. 이 셀은 경로 계획에서 막힌 칸이라 결과에는 영향이 없습니다.

노이즈 크기와 확산 계수는 제가 잡은 가정값입니다. `config.py`의 `NOISE_*`, `RT_SCATTERING_COEFFICIENT`에서 바꿀 수 있습니다.

## 결과 요약 (원본 → Sionna 맵)

| 방식 | 원본 Rmin [Mbps] | Sionna Rmin [Mbps] | Sionna 평균 [Mbps] |
|---|---|---|---|
| NAIVE | 12.9 | 9.3 | 54.0 |
| +MRC | 41.4 | 21.3 | 77.4 |
| +OPTBW | 27.7 | 13.0 | 72.2 |
| +SMARTPATH | 14.6 | 16.7 | 50.9 |
| MRC OPTBW | 64.3 | 47.3 | 100.6 |
| MRC SMART | 63.1 | 45.6 | 74.5 |
| OPT SMART | 31.7 | 32.3 | 72.8 |
| **JOINT** | **81.6** | **71.8** | **103.8** |

- **서쪽**(BS 쪽) 평균 SNR은 거의 같습니다(46.0 → 45.6 dB).
- **동쪽**은 31.7 → 17.1 dB로 크게 낮아졌습니다. 실제 ITU 콘크리트는 5.8 GHz에서 손실이 약 130 dB/m라서, 0.3 m 중앙 벽을 지나면 약 42 dB가 줄어듭니다. handout 값은 50 dB/m × 0.3 m = 15 dB였습니다. 그래서 동쪽은 0.5 m 틈, 회절, 다중 반사에 주로 의존합니다.
- 그래도 알고리즘의 결론은 같습니다. 70 Mbps QoS를 만족하는 방식은 JOINT뿐이고(가능한 쌍 186/900), AMR1은 최단 경로를, AMR2는 북쪽으로 우회하는 경로를 고릅니다.
- Task 5 offline→online 비율은 JOINT 기준 0.86 → 0.80입니다(57.2 Mbps).
- Task 1~5의 모든 검증 assert를 통과했습니다.

## 그림 (`figures/`)

- `00_sionna_scene_render.png`: Sionna 3D 렌더와 동쪽 셀까지 가는 경로(그림이 너무 복잡해지지 않게 깊이 2로 그림)
- `00b_map_handout_vs_sionna.png`: 원본 맵, Sionna 맵, Sionna 맵+노이즈, 차이 맵
- `00c_sionna_path_statistics.png`: 링크별 경로 수, 상호작용 종류별 전력 비중, 경로 깊이 분포, 전력 지연 프로파일(PDP)
- `00d_gaussian_noise.png`: 가우시안 노이즈에 따른 SNR 변화(중앙값 |ΔSNR| ≈ 1.1 dB)
- `01`~`15`: 원본과 같은 그림을 Sionna 맵으로 다시 만든 것입니다.
- `12_qos_sweep_amr2_minus3db.png`: AMR2 오프셋을 −7 dB 대신 −3 dB로 둔 QoS sweep입니다(T4.2 질문 3). 값은 `task4_qos_sweep_amr2_minus3db.json`에 있습니다.
- Sionna 맵에서는 QoS sweep의 세 구간 중 "우회" 구간이 나타나지 않습니다. 최단 길이(18.49 m) 경로쌍 [0, 22]가 이미 Rmin 71.8 Mbps로 거의 최대값(71.9 Mbps)이라, 70 Mbps까지는 우회 없이 T = 37.0 s로 만족하고 72 Mbps부터는 바로 불가능(fallback)이 됩니다. −3 dB에서는 11개 값 모두 같은 쌍(Rmin 86.8 Mbps)으로 만족합니다.

## 실행 (Windows, Visual Studio, CPU)

```powershell
pip install -r requirements.txt
# LLVM을 설치한 뒤 (예: C:\Program Files\LLVM)
$env:DRJIT_LIBLLVM_PATH = "C:\Program Files\LLVM\bin\LLVM-C.dll"
cd sionna_map_python_algo
python code/main.py --channel-maps --sionna-figures --paths --bandwidth --joint-plan --ablation --tradeoffs --task5
```

- 처음 실행할 때 Sionna 레이트레이싱을 합니다. 이 컨테이너(4코어)에서 약 23분 걸렸습니다.
- 원시 경로는 `code/sionna_paths.npz`(약 460 MB)에, 맵은 `code/fingerprint.npy`에 캐시됩니다.
- 노이즈 설정만 바꿨다면 `--force-fingerprint`로 맵만 다시 만들면 됩니다. 이때 레이트레이싱은 다시 하지 않습니다. 레이트레이싱 설정을 바꾸면 자동으로 다시 계산합니다.
- 용량 때문에 `sionna_paths.npz`는 폴더에 넣지 않았습니다. `fingerprint.npy`는 들어 있어서 다른 단계는 바로 실행됩니다.
- 메모리가 부족하면 `config.RT_RX_BATCH`(기본 6)나 `RT_MAX_PATHS_PER_SRC`(기본 400만, 이때 최대 약 5.5 GB)를 줄이세요. 배치가 너무 크면 Sionna 경로 버퍼가 넘쳐서 경로가 조용히 빠집니다.
