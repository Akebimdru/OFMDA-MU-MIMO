"""검증: OFDMA 가변 대역의 경계 이동 방식 vs 정확한 최적 (ofdma_exact.py 그대로), 완벽한 CSI와 NMSE −40 dB.

ofdma_exact.py 코드를 그대로 실행하되 두 가지만 바꾼다.
* 부반송파 rate 표: ofdma_var_csi.sc_prefix_table_csi (NMSE 0 또는 −40 dB)
* 출력 파일 이름 (두 경우를 동시에 돌려도 겹치지 않게)
사용: python ofdma_exact_csi.py perfect   또는   python ofdma_exact_csi.py csi40
출력: figures/13_ofdma_exact_{perfect|csi40}.png, _summary.json, code/ofdma_exact_{perfect|csi40}_results.npy
"""
import sys
import types
from pathlib import Path

from ofdma_var_csi import NMSE_DB, sc_prefix_table_csi

level = sys.argv[1]
lin = 0.0 if level == "perfect" else 10 ** (NMSE_DB / 10)
src = (Path(__file__).with_name("ofdma_exact.py")).read_text(encoding="utf-8")
src = src.replace("10_ofdma_exact_vs_shift_10cm", f"13_ofdma_exact_{level}")
src = src.replace('"ofdma_exact_results.npy"', f'"ofdma_exact_{level}_results.npy"')
src = src.replace('if __name__ == "__main__":', "if False:")
src = src.replace("all 24 orders) — 10 cm map\")", f"all 24 orders) — 10 cm map, {level} CSI\")")
mod = types.ModuleType("ofdma_exact_run")
mod.__file__ = __file__
exec(compile(src, "ofdma_exact.py", "exec"), mod.__dict__)
mod.sc_prefix_table = lambda h: sc_prefix_table_csi(h, lin)
mod.main()
