from __future__ import annotations

import json
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
OUT = ROOT / "output" / "docx" / "indoor_multi_amr_final_project_report.docx"


def set_cell_fill(cell, color: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shading = OxmlElement("w:shd")
    shading.set(qn("w:fill"), color)
    tc_pr.append(shading)


def set_cell_border(cell, color: str = "D9D9D9") -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = tc_pr.first_child_found_in("w:tcBorders")
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tc_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = qn(f"w:{edge}")
        element = borders.find(tag)
        if element is None:
            element = OxmlElement(f"w:{edge}")
            borders.append(element)
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), "4")
        element.set(qn("w:color"), color)


def set_repeat_table_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    header = OxmlElement("w:tblHeader")
    header.set(qn("w:val"), "true")
    tr_pr.append(header)


def set_font(run, size=10.5, bold=False, color=None):
    run.font.name = "Aptos"
    run._element.rPr.rFonts.set(qn("w:ascii"), "Aptos")
    run._element.rPr.rFonts.set(qn("w:hAnsi"), "Aptos")
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Malgun Gothic")
    run.font.size = Pt(size)
    run.bold = bold
    if color:
        run.font.color.rgb = RGBColor.from_string(color)


def add_text(doc, text: str, style=None, align=None, before=0, after=6):
    p = doc.add_paragraph(style=style)
    if align is not None:
        p.alignment = align
    p.paragraph_format.space_before = Pt(before)
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.line_spacing = 1.18
    r = p.add_run(text)
    set_font(r)
    return p


def add_heading(doc, text: str, level=1):
    p = doc.add_paragraph(style=f"Heading {level}")
    p.paragraph_format.space_before = Pt(10 if level == 1 else 7)
    p.paragraph_format.space_after = Pt(5)
    r = p.add_run(text)
    set_font(r, size=15 if level == 1 else 12, bold=True)
    return p


def add_caption(doc, text: str):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(8)
    r = p.add_run(text)
    set_font(r, size=8.5, color="555555")


def add_figure(doc, filename: str, width: float, caption: str):
    path = FIG / filename
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(0)
    p.add_run().add_picture(str(path), width=Inches(width))
    add_caption(doc, caption)


def add_table(doc, headers, rows, widths=None):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.autofit = False
    header = table.rows[0]
    set_repeat_table_header(header)
    for i, value in enumerate(headers):
        cell = header.cells[i]
        cell.text = ""
        set_cell_fill(cell, "1F4E78")
        set_cell_border(cell)
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(str(value))
        set_font(r, size=8.5, bold=True, color="FFFFFF")
        if widths:
            cell.width = Inches(widths[i])
    for row_values in rows:
        row = table.add_row()
        for i, value in enumerate(row_values):
            cell = row.cells[i]
            cell.text = ""
            set_cell_border(cell)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            if len(table.rows) % 2 == 1:
                set_cell_fill(cell, "F5F8FB")
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(2)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if i > 0 else WD_ALIGN_PARAGRAPH.LEFT
            r = p.add_run(str(value))
            set_font(r, size=8.5)
            if widths:
                cell.width = Inches(widths[i])
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return table


def page_break(doc):
    doc.add_page_break()


def main():
    ablation = json.loads((FIG / "task3_3_ablation_summary.json").read_text(encoding="utf-8"))
    task4 = json.loads((FIG / "task4_tradeoff_summary.json").read_text(encoding="utf-8"))
    task5 = json.loads((FIG / "task5_robustness_summary.json").read_text(encoding="utf-8"))
    validation2 = json.loads((FIG / "task2_3_validation.json").read_text(encoding="utf-8"))

    doc = Document()
    sec = doc.sections[0]
    sec.top_margin = Inches(0.72); sec.bottom_margin = Inches(0.67)
    sec.left_margin = Inches(0.78); sec.right_margin = Inches(0.78)
    styles = doc.styles
    styles["Normal"].font.name = "Aptos"; styles["Normal"].font.size = Pt(10.5)
    for style_name in ("Title", "Heading 1", "Heading 2"):
        styles[style_name].font.color.rgb = RGBColor(0, 0, 0)
    footer = sec.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer_run = footer.add_run("통신시스템 기말프로젝트  |  실내 다중 AMR 네트워크")
    set_font(footer_run, size=8, color="666666")

    # Cover page
    p = doc.add_paragraph(style="Title")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(115)
    p.paragraph_format.space_after = Pt(16)
    r = p.add_run("실내 다중 AMR 네트워크\n기말프로젝트 보고서")
    set_font(r, size=25, bold=True)
    add_text(doc, "실내 다중 AMR의 공동 경로 및 무선 자원 최적화", align=WD_ALIGN_PARAGRAPH.CENTER, after=28)
    add_text(doc, "통신시스템", align=WD_ALIGN_PARAGRAPH.CENTER, after=5)
    add_text(doc, "기말프로젝트 제출 보고서", align=WD_ALIGN_PARAGRAPH.CENTER, after=48)
    add_text(doc, "학번 및 이름  ______________________________", align=WD_ALIGN_PARAGRAPH.CENTER, after=6)
    add_text(doc, "제출일  ______________________________", align=WD_ALIGN_PARAGRAPH.CENTER, after=0)
    page_break(doc)

    # Page 2
    add_heading(doc, "1. 요약")
    add_text(doc, "본 보고서는 실내 창고를 이동하는 두 대의 자율이동로봇(AMR)이 20 MHz OFDMA 하향링크를 공유하는 상황을 위한 종단 간 시뮬레이터를 제시한다. 기하학적 레이 트레이싱으로 결정론적 채널 지문 지도를 만들고, 충돌 없는 격자 경로를 계획하며, 매 시간 단계마다 연속 대역폭 할당을 최적화해 통신 성능과 임무 시간의 상충 관계를 평가했다.")
    add_text(doc, "핵심 결과는 통신 인지 경로, 4 안테나 최대비 결합(MRC), 순간 최대-최소 대역폭 할당을 함께 사용하는 JOINT 방식이 최상의 최저 처리량을 달성한다는 것이다. 평가한 후보 집합에서 JOINT의 최소 처리량은 81.6 Mbps로 NAIVE 기준 방식의 12.9 Mbps보다 높았다. 대신 AMR2가 더 긴 경로를 선택하므로 완료 시간은 약 37.0초에서 45.0초로 늘어났다.")
    add_heading(doc, "프로젝트 목표", 2)
    add_text(doc, "최적화 목표는 최소 처리량 70 Mbps 제약을 만족하면서 두 로봇 중 늦게 도착하는 로봇의 완료 시간을 최소화하는 것이다. 즉 우회 경로가 추가하는 이동 시간을 보상할 만큼 무선 상태를 개선할 때만 그 경로는 유용하다. AMR별 30개 후보 경로를 평가하여 총 900개 경로 쌍을 비교했다.")
    add_heading(doc, "시스템 파라미터", 2)
    add_table(doc, ["항목", "값", "역할"], [
        ["공간", "18 m x 10 m x 3 m", "실내 창고 형상"],
        ["기지국", "(3, 5, 2.7) m, 4소자 ULA", "천장 수신 배열"],
        ["OFDMA", "20 MHz, 부반송파 64개", "연속 대역폭 분할"],
        ["AMR 속도", "각 0.5 m/s", "임무 시간 계산"],
        ["격자·안전", "0.5 m, 버퍼 0.2 m", "A* 점유 격자 계획"],
        ["QoS 목표", "70 Mbps", "실현 가능성 기준"],
    ], [1.35, 1.55, 3.4])
    add_text(doc, "난수 기반 후보 경로 생성에는 시드 42를 사용했다. 소스 코드는 하나의 실행 순서로 채널 지문, 그림, 수치 요약을 재현한다.", after=0)
    page_break(doc)

    # Page 3
    add_heading(doc, "2. 채널 지문 지도와 검증")
    add_text(doc, "0.5 m 간격의 각 격자에서 채널 지문은 복소수 4 x 64 행렬을 저장한다. 기지국 안테나와 부반송파마다 채널은 가시경로와 각 외곽 벽에서 발생하는 1차 반사경로의 코히어런트 합으로 계산된다. 레이 트레이서는 장애물을 관통하는 현 길이를 이용해 재질별 투과 손실을 적용한다. 중앙 벽과 기둥은 콘크리트, 캐비닛은 목재로 모델링했다.")
    add_figure(doc, "02_snr_heatmaps.png", 6.55, "그림 1. 장애물과 단말 위치를 중첩한 평균 SISO 및 MRC SNR 지도.")
    add_table(doc, ["검증 항목", "측정값", "기대 조건", "결과"], [
        ["기지국 근처 SISO", "52.5 dB", "약 50 dB", "통과"],
        ["동쪽 음영 SISO", "15.4 dB", "25 dB 이하", "통과"],
        ["평균 MRC 이득", "6.43 dB", "약 6 dB", "통과"],
        ["지문 행렬 크기", "20 x 36 x 4 x 64", "격자·안테나·부반송파", "통과"],
    ], [1.45, 1.35, 2.55, 0.85])
    add_text(doc, "가장 큰 감쇠는 x = 9 m의 두 콘크리트 벽 구간이 지배적인 직접파를 약화시키는 실내 동쪽에서 나타난다. MRC는 주파수 선택적 다중경로에 따른 공간 변동을 유지하면서도 4 안테나 수신 다이버시티에 해당하는 약 6 dB 이득을 회복했다.", after=0)
    page_break(doc)

    # Page 4
    add_heading(doc, "3. 경로 계획과 대역폭 할당")
    add_text(doc, "A* 알고리즘은 36 x 20 점유 격자에서 8방향 이동으로 구현했다. 상하좌우 이동의 비용은 1 격자 단위이고 대각 이동 비용은 루트 2이며, 유클리드 휴리스틱은 허용적이다. 장애물 버퍼는 실제 장애물 외곽에 적용했다. 두 AMR의 최단 경로는 모두 중앙 통로를 통과하고 가장자리 우회를 피하며, 길이는 각각 18.49 m이다.")
    add_figure(doc, "04_candidate_paths.png", 6.35, "그림 2. MRC 지도 위에 중첩한 AMR별 재현 가능한 후보 경로 30개. 굵은 선은 최단 경로.")
    add_text(doc, "시뮬레이터의 0.1초 시간 단계마다 부반송파 SNR은 dB 영역에서 이중선형 보간한다. 균등 할당은 AMR당 32개 부반송파를 배정한다. Opt BW는 AMR1에 대해 순방향, AMR2에 대해 역방향 누적 스펙트럼 효율을 계산한 뒤 순간 최소 전송률을 최대화하는 분할을 선택한다. 벡터화된 누적합 구현은 요구 범위인 k* = 1~63을 모두 확인했으며 균등 할당보다 낮은 최소 전송률을 만들지 않았다.")
    page_break(doc)

    # Page 5
    add_heading(doc, "4. Task 3.1 대역폭 할당")
    add_figure(doc, "06_bw_allocation_shortest_paths.png", 4.65, "그림 3. 5개 주요 방식의 연속 대역폭 점유율. 균등 할당 방식은 0.5로 유지되고, Opt BW 방식은 시간에 따라 AMR1의 몫을 조정한다.")
    add_text(doc, "최적 분할은 현재 전송률이 제한되는 AMR에 더 많은 부반송파를 동적으로 배정한다. 이는 OFDMA 모델이 요구하는 연속 블록 구조를 보존하면서 두 순간 전송률 중 작은 값을 최대화한다.")
    page_break(doc)

    # Page 6
    add_heading(doc, "5. Task 3.2 시간별 처리량")
    add_figure(doc, "01_throughput.png", 6.15, "그림 4. 5개 주요 방식의 시간별 Rmin. 빨간 점선은 70 Mbps QoS 제약, 색상별 세로 점선은 각 방식의 완료 시간이다.")
    add_text(doc, "JOINT는 전체 구간에서 가장 높은 최저 처리량을 보이며, SmartPath 단독은 평균 처리량은 개선하지만 중간 음영 구간의 급락을 완전히 제거하지 못한다. 그림 안의 요약 상자는 각 방식의 Rmin, 평균 처리량, 완료 시간을 함께 제시한다.")
    page_break(doc)

    # Page 7
    add_heading(doc, "6. QoS 제약 공동 계획")
    add_text(doc, "900개 경로 쌍은 공통 시간 격자에서 평가한다. 각 쌍에 대해 시뮬레이터는 AMR1과 AMR2에 각각 0 dB와 -7 dB 링크 오프셋을 적용하고, 경로 쌍의 최소 처리량과 완료 시간을 계산한다. 70 Mbps를 만족하는 쌍 중 완료 시간이 최소인 쌍을 선택하며, 동률이면 최소 처리량이 큰 쌍을 선택한다. 실현 가능한 쌍이 없을 경우에는 최소 처리량이 최대인 쌍을 선택한다.")
    add_figure(doc, "05_paths_by_scheme.png", 6.35, "그림 5. MRC 열지도 위의 최단 경로, SmartPath, JOINT 경로. 실선은 AMR1, 점선은 AMR2.")
    joint = validation2["JOINT"]
    smart = validation2["SMARTPATH"]
    add_table(doc, ["계획", "가능 쌍", "Rmin", "완료 시간", "선택 기준"], [
        ["SMARTPATH SISO 균등", str(smart["feasible_pair_count"]), f"{smart['selected_rmin_mbps']:.1f} Mbps", f"{smart['selected_completion_time_s']:.1f} s", "최대 Rmin 대체 선택"],
        ["JOINT MRC Opt BW", str(joint["feasible_pair_count"]), f"{joint['selected_rmin_mbps']:.1f} Mbps", f"{joint['selected_completion_time_s']:.1f} s", "가능 해 중 최소 T"],
    ], [1.55, 1.2, 1.25, 1.45, 1.5])
    add_text(doc, "JOINT 계획은 70 Mbps 목표에서 72개의 실현 가능한 쌍을 보였다. AMR1은 A*가 기하적으로 다른 동률 최단 경로를 선택할 수 있으나 길이는 최단 경로와 같다. AMR2는 통신 인지 북쪽 우회 경로를 선택하여 경로 길이가 4.0 m 증가한다.", after=0)
    page_break(doc)

    # Page 7
    add_heading(doc, "7. 8개 방식 전체 소거 분석")
    add_text(doc, "소거 분석은 세 가지 이진 축을 독립적으로 전환한다. 경로는 최단 또는 통신 인지 경로, 대역폭은 균등 또는 최적 할당, 수신은 SISO 또는 MRC이다. 모든 방식은 일관된 시뮬레이터와 해당 방식의 경로 선택 규칙으로 평가했다.")
    add_figure(doc, "10_ablation_bars.png", 6.5, "그림 6. 2의 3승 전체 소거 분석. 8개 방식의 Rmin, 평균 처리량, 완료 시간을 표시한다.")
    rows = []
    for name, data in ablation["schemes"].items():
        rows.append([name, f"{data['rmin_mbps']:.1f}", f"{data['mean_mbps']:.1f}", f"{data['completion_time_s']:.1f}"])
    add_table(doc, ["방식", "Rmin Mbps", "평균 Mbps", "T s"], rows, [2.05, 1.25, 1.35, 1.05])
    add_text(doc, "MRC는 NAIVE 대비 Rmin을 28.6 Mbps 높이는 가장 큰 단독 기여 요소였다. JOINT의 개선량은 각 요소의 고립된 기여를 합한 값보다 크며, 이는 초가산성과 일치한다. MRC는 유효 SNR을 높이고, Opt BW는 부반송파 비율을 바꾸며, SmartPath는 임무 중 표본화되는 채널값을 바꾼다. 따라서 이들의 상호작용은 단순 가산이 아니라 전송률 식 안에서 곱셈적으로 나타난다.", after=0)
    page_break(doc)

    # Page 8
    add_heading(doc, "8. 파레토 경계")
    add_text(doc, "파레토 산점도는 JOINT 기술 조합으로 900개 후보 경로 쌍 전체를 평가한다. 실현 가능한 점은 70 Mbps 요구조건을 만족한다. 선택된 JOINT 점은 다른 어떤 실현 가능한 후보도 더 빨리 끝나지 않으므로 실현 가능 경계 위에 위치한다. AMR별 후보가 30개인 이산 경로 집합이므로 경계는 계단형 형태를 보인다.")
    add_figure(doc, "11_pareto.png", 5.85, "그림 7. 선택된 JOINT 점과 파레토 경계를 포함한 MRC Opt BW 후보 쌍의 실현 가능·불가능 분포.")
    counts = task4["feasibility_counts"]
    add_text(doc, f"실현 가능한 경로 쌍 수는 50 Mbps에서 {counts['50']}개, 70 Mbps에서 {counts['70']}개, 80 Mbps에서 {counts['80']}개다. QoS 문턱이 엄격해질수록 이 수가 감소하며, 품질이 높은 경로 쌍이 제한적임을 보여 준다.")
    page_break(doc)

    # Page 9
    add_heading(doc, "9. QoS 스윕")
    add_figure(doc, "12_qos_sweep.png", 4.55, "그림 8. 지정된 11개 QoS 값에 대한 스윕. 아래 패널은 선택된 AMR1·AMR2 경로 길이를 나타낸다.")
    add_text(doc, "스윕 결과 60 Mbps까지는 최단 경로가 실현 가능하고 완료 시간도 약 37초로 유지되는 완화 구간이다. 65 Mbps에서 선택 해는 우회를 시작한다. 68 Mbps부터 AMR2는 더 긴 22.49 m 경로를 택하고 완료 시간은 약 45초로 증가한다. 명시된 80 Mbps까지의 값에서 평가 후보 집합은 실현 가능했다.", after=0)
    page_break(doc)

    # Page 10
    add_heading(doc, "10. 오프라인·온라인 강건성 및 애니메이션")
    add_text(doc, "오프라인 계획은 소규모 레일리 페이딩 하에서 독립적인 30회 실현으로 재생했다. 페이딩은 원형 복소 가우시안 이득의 제곱 크기로 선형 SNR을 곱한다. 매 실현에서 오프라인 k* 일정표를 그대로 재사용하고 온라인 대역폭 재최적화는 수행하지 않았다. 이를 통해 오프라인 결정론적 지문 지도에 의존할 때의 실질적 비용을 분리해 확인했다.")
    add_figure(doc, "13_offline_vs_online.png", 4.75, "그림 9. 오프라인 결정론적 Rmin 궤적, 온라인 재생 평균, 그리고 표준편차 1개 범위.")
    robust_rows = []
    for name, values in task5.items():
        robust_rows.append([name, f"{values['offline_rmin_mbps']:.1f}", f"{values['online_rmin_mean_mbps']:.1f}", f"{values['offline_to_online_ratio']:.3f}"])
    add_table(doc, ["방식", "오프라인 Rmin", "온라인 평균 Rmin", "온라인/오프라인 비"], robust_rows, [1.35, 1.45, 1.75, 1.6])
    add_heading(doc, "11. 결론")
    add_text(doc, "본 프로젝트는 무선 자원 결정과 물리적 이동 궤적을 함께 최적화해야 함을 보인다. MRC가 가장 큰 단독 이득을 제공했지만, 최상의 결과는 수신 다이버시티·적응형 연속 대역폭 할당·AMR2의 통신 인지 우회를 결합했을 때 얻어졌다. 이 우회는 QoS 목표가 추가 임무 시간을 정당화할 만큼 높을 때만 가치가 있다.")
    add_text(doc, "온라인 재생 결과는 고정된 오프라인 일정표에서 페이딩에 의해 예상되는 처리량 저하를 보인다. JOINT는 높은 평균 최소 처리량을 유지하지만 NAIVE보다 오프라인/온라인 비가 약간 낮다. 향후에는 온라인 채널 추정, 제한된 피드백 기반의 대역폭 갱신 정책, 이산 파레토 경계를 부드럽게 할 더 큰 후보 경로 집합을 포함할 수 있다.", after=0)
    add_heading(doc, "참고문헌", 2)
    add_text(doc, "[1] H. J. Yang, 실내 다중 AMR 네트워크를 위한 공동 경로 및 자원 최적화 기말프로젝트 안내서, 통신시스템, 2026.", after=0)

    # Appendix: direct answers to the handout analysis questions.
    page_break(doc)
    add_heading(doc, "부록 A. Task 1~3 분석 질문 답변")
    add_heading(doc, "A.1 Task 1: 채널 지도 해석", 2)
    add_text(doc, "강한 음영은 중앙 콘크리트 벽 뒤의 실내 동쪽에서 발생한다. 매뉴얼 정의대로 부반송파 전력을 선형 영역에서 평균한 뒤 dB로 변환하면, 기지국 인근의 밝은 격자와 가장 어두운 동쪽 음영 격자의 SISO SNR 차이는 62.08 dB이다. 이는 x = 9 m 부근의 콘크리트 벽 두 구간이 직접파를 크게 감쇠시키기 때문이다.")
    add_text(doc, "같은 정의의 SISO 지도에서 AMR1 최단 경로 6개 균등 시간 표본은 각각 42.31, 50.17, 47.13, 16.17, 30.98, 35.95 dB였다. 최저점은 출발·도착점이 아니라 22.2초의 중간 궤적(x=10.40 m, y=5.25 m)에서 나타났다. 따라서 통신 인지 우회는 종점 보정이 아니라 중앙 벽을 지나는 중간 구간의 음영을 회피할 때 효과가 있다.")
    add_text(doc, "같은 선형 평균 후 dB 변환 정의에서 격자 전체의 MRC-SISO 이득은 평균 6.43 dB, 표준편차 2.44 dB이다. 이상적인 독립 4안테나의 평균 이득은 약 6 dB이지만, 각 위치에서 직접파·반사파의 위상 및 장애물 관통 경로가 달라 안테나별 상관도가 변한다. 이 때문에 실제 이득은 공간적으로 균일하지 않다.")
    add_heading(doc, "A.2 Task 2: 경로 계획 검증", 2)
    add_text(doc, "두 AMR의 최단 경로 길이는 모두 18.49 m이며, y=5.0~5.5 m의 중앙 통로를 통과한다. 어느 경로도 y<1 또는 y>9의 실내 가장자리 우회를 사용하지 않는다. 후보 경로는 시드 42로 생성한 30개 경로이며, JOINT에서는 AMR1의 길이는 18.49 m로 유지되고 AMR2만 북쪽으로 우회해 22.49 m가 된다.")
    add_heading(doc, "A.3 Task 3.1~3.2: Opt BW와 병목", 2)
    add_text(doc, "Opt BW에서 AMR1의 누적 전송률 R1(k)는 k가 커질수록 증가하고, AMR2의 R2(k)는 감소한다. 따라서 min(R1,R2)는 두 곡선이 교차하는 부근까지 증가한 뒤 감소하는 단봉 형태이며, 부반송파 1~63을 한 번의 누적합 스캔으로 탐색할 수 있다. 실제 선택된 k*는 8~46 범위였고, 모든 시간 단계에서 Opt BW의 최소 전송률은 균등 할당보다 낮지 않았다.")
    add_text(doc, "NAIVE의 전체 최저점은 17.9초에 AMR2가 병목일 때 발생했다. 시간 전체에서는 AMR1이 병목인 비율이 45.3%, AMR2가 병목인 비율이 54.7%여서, 단일 종점이 아니라 이동 중 채널 상태가 병목을 번갈아 만든다. NAIVE와 +OPTBW의 순간 차이는 8.1초에 가장 크며 30.85 Mbps이다. 이때 균등 할당 전송률은 AMR1 171.42 Mbps, AMR2 76.45 Mbps로 비대칭적이었고, Opt BW는 k*=20을 선택해 두 전송률을 약 107.30/107.66 Mbps로 맞춘다.")
    add_text(doc, "+SMARTPATH 단독의 Rmin은 14.56 Mbps로 NAIVE의 12.85 Mbps보다 1.71 Mbps만 높고, 완료 시간은 36.97초에서 44.04초로 증가한다. 평균 처리량은 80.05 Mbps에서 85.43 Mbps로 개선된다. 즉 일부 구간의 채널은 좋아지지만 중간 궤적 병목을 충분히 없애지는 못하며, 평균 성능만 보면 유리해 보여도 최저 처리량과 임무 시간이라는 서비스 지표에서는 불리할 수 있다.")

    page_break(doc)
    add_heading(doc, "부록 B. Task 3~5 분석 질문 답변")
    add_heading(doc, "B.1 Task 3.3: 소거 분석", 2)
    add_text(doc, "단독 Rmin 개선량 Delta_add는 MRC 28.59 Mbps, Opt BW 14.88 Mbps, SmartPath 1.71 Mbps 순이다. 따라서 가장 큰 기여 요소는 수신 SNR 자체를 높이는 MRC이다. Delta_add의 합은 45.18 Mbps인 반면 JOINT-NAIVE는 68.76 Mbps이므로, 결합 이득은 합보다 23.58 Mbps 크다. MRC는 SNR 항, Opt BW는 할당 부반송파 수 항, SmartPath는 경로를 따라 표본화되는 채널 항을 바꾸므로 log2(1+SNR) 전송률 식에서 상호 증폭된다.")
    add_text(doc, "JOINT에서 각 요소를 제거했을 때의 Delta_drop은 MRC 49.87 Mbps, Opt BW 18.49 Mbps, SmartPath 17.31 Mbps로 모든 요소에서 Delta_add보다 크다. 이는 다른 두 요소가 작동하는 상태에서 한 요소를 제거하면 결합 이득까지 함께 잃기 때문이다. SmartPath는 평균 처리량을 5.38 Mbps 높였지만 최저 처리량의 개선은 작고 완료 시간을 늘렸으므로, 실제 응용에서는 평균 처리량만으로 경로 품질을 판단하면 안 된다.")
    add_heading(doc, "B.2 Task 4: 파레토와 QoS 스윕", 2)
    add_text(doc, "900개 후보 중 50, 70, 80 Mbps를 만족하는 경로 쌍은 각각 646개, 72개, 22개다. JOINT는 실현 가능한 쌍 가운데 완료 시간을 최소화하도록 선택되므로 파레토 경계에 존재한다. 경계가 계단형인 이유는 AMR별 후보가 30개뿐인 이산 집합이기 때문이며, 후보를 300개로 늘리면 더 촘촘하고 부드러운 근사 경계가 기대된다.")
    add_text(doc, "QoS 스윕의 완화 구간은 60 Mbps 이하(T=36.97초)이고, 65 Mbps에서 AMR1의 작은 우회가 시작된다. 68 Mbps부터는 약 45초의 큰 우회 구간으로 전환되며, AMR2가 22.49 m 북쪽 경로를 택한다. 지정된 최대값 80 Mbps까지는 비실현 가능 구간이 나타나지 않았으므로, 그 임계값은 본 후보 집합에서 80 Mbps보다 높다. AMR2는 -7 dB 링크 오프셋 때문에 더 자주 병목이 되어 큰 우회를 먼저 수행한다.")
    add_text(doc, "반사실 비교로 AMR2 오프셋을 -3 dB로 완화하면 70 Mbps에서 596개 쌍이 가능해지고, 최단 경로만으로 T=36.97초 및 Rmin=79.25 Mbps를 달성한다. 80 Mbps에서만 작은 우회(T=37.56초)가 필요했다. 즉 약한 링크를 개선해도 완화-우회-비실현 가능이라는 세 구간 구조 자체는 남지만, 우회와 비실현 가능의 문턱은 더 높은 QoS 쪽으로 이동한다.")
    add_heading(doc, "B.3 Task 5: 오프라인-온라인 강건성", 2)
    add_text(doc, "온라인/오프라인 Rmin 비율은 NAIVE 0.867, +MRC 0.822, +OPTBW 0.861, +SMARTPATH 0.809, JOINT 0.859이다. JOINT는 NAIVE보다 약간 낮은 비율로, 고품질 경로와 적응형 할당으로 얻은 오프라인 이득이 고정된 일정표 아래 레일리 페이딩에 일부 민감함을 보인다. 이는 log2(1+rho|g|^2)가 오목함수이므로 Jensen 부등식에 따라 E[log2(1+rho|g|^2)]가 log2(1+rho)보다 작아지는 현상으로 설명된다. 본 재생은 매 실현에서 오프라인 k*를 그대로 사용했으며, 온라인 재계획을 하지 않았다.")

    doc.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
