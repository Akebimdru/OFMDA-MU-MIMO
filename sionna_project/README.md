# 기말 프로젝트 — Sionna 구현 (Indoor Multi-AMR, Joint Path-and-Resource Optimization)

Lecture 7(Sionna 소개)에서 다룬 **Sionna 2.x (PyTorch 백엔드) + Sionna RT** 로 기말 프로젝트 Task 1~5를 구현한 코드입니다.
GPU 없이 CPU에서 동작하도록 만들었고, 테스트 환경(4코어 CPU)에서 `python main.py` 전체 실행이 약 100초 걸렸습니다.

## Sionna가 쓰이는 곳

| 단계 | Sionna 기능 | 파일 |
|---|---|---|
| T1.1 장면 | Mitsuba XML + PLY 메시로 18×10×3 m 방 생성 → `sionna.rt.load_scene` | `scene_builder.py` |
| T1.1 ray tracing | ULA 4소자 = `Transmitter` 4개(정확한 p_ant 위치), 720개 격자점 = `Receiver`, `PathSolver(max_depth=1, los, specular_reflection)` 로 LOS + 벽 1차 반사 5경로 탐색 | `fingerprint.py` |
| T1.1 CIR → OFDM | `sionna.phy.channel.subcarrier_frequencies`, `cir_to_ofdm_channel` 로 64개 부반송파 H_k 계산 | `fingerprint.py` |
| T5.3 Rayleigh | `sionna.phy.utils.complex_normal` (전역 seed `sionna.phy.config.seed = 42`) | `schemes.py` |
| (선택) 3D 렌더 | `scene.render_to_file(..., paths=...)` → `figures/00_sionna_scene_render.png` | `visualizer.py` |

### 채널 모델을 handout 식과 정확히 맞춘 방법
Sionna RT는 기하(경로 지연 τ, 반사점)를 계산하고, 경로 이득은 handout 식으로 다시 만듭니다.
handout은 콘크리트 벽 반사계수를 상수 ρ = 0.7로, 장애물은 반사 없이 α[dB/m]×chord 투과 손실로 정의하는데,
Sionna RT 기본 재질 모델(ITU Fresnel 계수, slab 굴절)은 이와 다르기 때문입니다.

- RT 장면에는 반사체인 **둘레 벽 4개만** 넣습니다 (바닥·천장 반사 없음, 장애물은 반사하지 않음).
- `h_path = ρ · 10^(−L_pen/20) · λ/(4πd) · e^(−j2πd/λ_k)`, d = c·τ (Sionna가 준 τ)
- `L_pen = Σ α_m · chord_m`: Sionna가 준 반사점과 수신점을 잇는 선분과 12개 장애물의 교차 길이(slab / 원기둥 2차식)
- 위상은 `cir_to_ofdm_channel` 이 e^(−j2π f_k τ)로 정확히 계산하고, 진폭의 λ_k/λ_c 비율은 모든 경로에 공통이라 마지막에 곱합니다.

실행할 때마다 Sionna 결과를 자동으로 검증합니다 (`results_summary.json` → `task1.sionna_rt_check`).
- 모든 (안테나, 셀) 링크에서 경로가 정확히 5개
- Sionna 경로 길이 vs image-source 해석해 오차 최대 1e-5 m
- Sionna LOS 진폭 vs Friis λ/(4πd) 차이 1e-5 dB 이하

`config.RT_GAIN_MODEL = "sionna"` 로 바꾸면 반사 계수로 Sionna의 ITU 콘크리트 Fresnel 계수를 그대로 쓰는 비교 실험도 할 수 있습니다.

## 설치 (Windows, CPU, Visual Studio)

1. **Python 3.11 또는 3.12 (64-bit)** 를 준비합니다. Sionna 2.x는 Python 3.11 이상이 필요합니다.
2. **LLVM 설치** (CPU에서 Sionna RT를 쓰려면 필요합니다. Lecture 7 p.11 "CPU only: LLVM 추가 설치 필요")
   - https://github.com/llvm/llvm-project/releases 에서 `LLVM-<버전>-win64.exe` 를 받아 설치합니다.
   - 환경 변수 `DRJIT_LIBLLVM_PATH` = `C:\Program Files\LLVM\bin\LLVM-C.dll` 를 추가합니다.
     (설치 경로가 다르면 맞춰서 바꾸세요. 환경 변수를 추가한 뒤 Visual Studio를 다시 시작해야 적용됩니다.)
3. 이 폴더(`sionna_project`)에서 가상환경을 만들고 패키지를 설치합니다.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU 전용 PyTorch (작음)
pip install -r requirements.txt
python -c "import sionna, sionna.rt, mitsuba as mi; print(sionna.__version__, mi.variant())"
```
마지막 줄에서 `2.2.0 llvm_ad_mono_polarized` 처럼 나오면 정상입니다 (Lecture 7 p.15와 같습니다).

### Visual Studio에서 실행
1. Visual Studio Installer에서 **Python 개발** 워크로드를 설치합니다.
2. `파일 > 열기 > 폴더` 로 `sionna_project` 폴더를 엽니다.
3. 솔루션 탐색기의 `Python 환경` 에서 위에서 만든 `.venv` 를 선택합니다 (`환경 추가 > 기존 환경`).
4. `code\main.py` 를 오른쪽 클릭 → **시작 항목으로 설정** 후 F5 (또는 Ctrl+F5).
   - 인수를 주려면 `디버그 > 디버그 및 시작 설정`에서 `"args": ["--no-video"]` 처럼 넣습니다.

명령줄에서 실행해도 됩니다.
```powershell
cd code
python main.py              # Task 1~5 전체 (fingerprint는 처음 한 번만 ray tracing, 이후 캐시 사용)
python main.py --force-rt   # Sionna RT fingerprint 다시 계산
python main.py --no-video   # MP4 생략 (빠른 확인)
python main.py --render     # Sionna RT 3D 장면 렌더 이미지 추가
```

## 파일 구성
```
sionna_project/
  requirements.txt
  code/
    config.py          장면·PHY·QoS·seed 상수 (handout p.5-12)
    scene_builder.py   Mitsuba XML/PLY 장면 생성 (RT용: 벽 4개, 렌더용: 벽+장애물+바닥)
    fingerprint.py     T1.1 Sionna RT → H (20,36,4,64), 캐시 fingerprint_sionna.npy
    path_planner.py    T2.1 A* (8-연결, 유클리드 휴리스틱), T2.2 후보 30개 (np.random.seed(42))
    simulator.py       T1.2 SNR 지도, bilinear 보간, T3.1 Equal-BW / prefix-sum Opt-BW
    joint_planner.py   T2.3 900쌍 전수 탐색 + QoS 필터 + fallback
    schemes.py         T3.3 8-scheme ablation, T4 Pareto·QoS sweep, T5.3 Rayleigh 재생
    visualizer.py      모든 그림·MP4
    main.py            전체 실행
  scene/               생성된 Mitsuba 장면
  figures/             결과 그림, 동영상, results_summary.json
  RESULTS.md           결과 수치와 분석 질문 답변 요약
```

## 구현 메모
- 격자는 0.5 m 셀 **중심**(x = 0.25…17.75, y = 0.25…9.75)입니다. 격자점을 0, 0.5, … 로 두면 y = 5.0, 5.5 점이 중앙 벽 0.2 m 버퍼 안에 들어가 0.5 m 통로가 막힙니다.
- 반사 경로의 장애물 검사는 handout 대로 image-BS → AMR 직선(방 안에서는 반사점 → AMR 구간)에 합니다. `fingerprint.REFLECTION_BLOCKING = "both_legs"` 로 BS → 반사점 구간도 넣을 수 있습니다.
- handout T2.3은 계획 시간축을 2·SIM_DT 로 적었지만, 기본값은 계획과 평가 모두 SIM_DT = 0.1 s 입니다 (`config.PLAN_TIME_STRIDE = 2` 로 바꿀 수 있음). 이렇게 해야 선택된 쌍의 R_min이 평가에서 그대로 재현됩니다.
- T5.3은 오프라인 k*(t)를 그대로 재사용하고(재계획 없음), 페이딩은 |H|²(SNR)에 곱합니다.
- ffmpeg이 없으면 애니메이션을 GIF로 저장합니다 (`imageio-ffmpeg` 설치 시 MP4).
