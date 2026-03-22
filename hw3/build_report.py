"""
生成原油期货课程作业 PDF 报告。

依赖：
1. term_structure_analysis.py
2. crude_term_structure.csv
3. crude_term_structure.png
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MPLCONFIGDIR = BASE_DIR / ".mplconfig"
MPLCONFIGDIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR.resolve()))

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from term_structure_analysis import CrudeTermStructureAnalyzer, configure_font


OUTPUT_DIR = BASE_DIR
CSV_PATH = OUTPUT_DIR / "crude_term_structure.csv"
PNG_PATH = OUTPUT_DIR / "crude_term_structure.png"
PDF_PATH = OUTPUT_DIR / "hw3_report.pdf"


def load_or_build_term_data() -> tuple[pd.DataFrame, object]:
    analyzer = CrudeTermStructureAnalyzer(output_dir=OUTPUT_DIR)

    if CSV_PATH.exists() and PNG_PATH.exists():
        term_df = pd.read_csv(CSV_PATH)
        term_df["contract_month"] = pd.to_datetime(term_df["contract_label"] + "-01")
        term_df["close_date"] = pd.to_datetime(term_df["close_date"])
        term_df = term_df.sort_values("contract_month").reset_index(drop=True)
        term_df["month_index"] = range(len(term_df))
        analysis = analyzer.analyze_structure(term_df)
        return term_df, analysis

    term_df = analyzer.build_term_structure()
    analysis = analyzer.analyze_structure(term_df)
    analyzer.save_term_table(term_df)
    analyzer.plot_term_structure(term_df, analysis)
    return term_df, analysis


def wrap_paragraph(text: str, width: int = 36) -> str:
    lines = []
    for paragraph in text.splitlines():
        if not paragraph.strip():
            lines.append("")
            continue
        lines.extend(textwrap.wrap(paragraph, width=width))
    return "\n".join(lines)


def add_page_title(fig: plt.Figure, title: str, subtitle: str | None = None) -> None:
    fig.text(0.5, 0.95, title, ha="center", va="top", fontsize=22, fontweight="bold")
    if subtitle:
        fig.text(0.5, 0.915, subtitle, ha="center", va="top", fontsize=11, color="#444444")


def create_cover_page(pdf: PdfPages, term_df: pd.DataFrame, analysis: object) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    add_page_title(fig, "原油期货期限结构分析报告", "程序设计课程作业")

    latest_date = term_df["close_date"].max().date()
    near = term_df.iloc[0]
    far = term_df.iloc[-1]

    intro = (
        "本次作业围绕上海国际能源交易中心原油期货(INE 原油, SC)展开。"
        "程序使用 AkShare 抓取当前可交易的不同月份合约，并提取最近一个交易日的收盘价，"
        "据此绘制期限结构图，判断市场处于升水还是贴水状态，最后结合期限结构给出可能的跨期交易思路。"
    )
    tasks = (
        "1. 自动识别原油各月份合约。\n"
        "2. 获取各合约最近一个交易日的收盘价。\n"
        "3. 绘制期限结构图并判断结构形态。\n"
        "4. 对结果进行经济含义解释。\n"
        "5. 给出基于期限结构的交易策略与风险提示。"
    )
    snapshot = (
        f"本次报告使用的数据最新收盘日期为 {latest_date}。\n"
        f"近月合约为 {near['symbol']}，收盘价 {near['close_price']:.2f} 元/桶；"
        f"远月合约为 {far['symbol']}，收盘价 {far['close_price']:.2f} 元/桶。\n"
        f"根据首末价差与整体斜率判断，当前市场属于 {analysis.structure_name_cn}。"
    )

    fig.text(0.08, 0.82, "作业目标", fontsize=15, fontweight="bold")
    fig.text(0.08, 0.77, wrap_paragraph(intro, width=39), fontsize=12, va="top", linespacing=1.7)

    fig.text(0.08, 0.59, "完成内容", fontsize=15, fontweight="bold")
    fig.text(0.08, 0.55, tasks, fontsize=12, va="top", linespacing=1.8)

    fig.text(0.08, 0.35, "结果摘要", fontsize=15, fontweight="bold")
    fig.text(0.08, 0.31, wrap_paragraph(snapshot, width=39), fontsize=12, va="top", linespacing=1.7)

    fig.text(0.08, 0.12, "数据来源：AkShare + 新浪期货接口", fontsize=11, color="#444444")
    fig.text(0.08, 0.09, "输出文件：crude_term_structure.csv / crude_term_structure.png / crude_oil_assignment_report.pdf", fontsize=11, color="#444444")

    plt.axis("off")
    pdf.savefig(fig)
    plt.close(fig)


def create_method_page(pdf: PdfPages, term_df: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    add_page_title(fig, "数据与方法")

    left_text = (
        "1. 数据对象\n"
        "选择上海国际能源交易中心原油期货 SC 合约作为研究对象，"
        "按当前市场上可交易的不同月份合约构造期限结构。\n\n"
        "2. 数据抓取方法\n"
        "先调用 ak.futures_zh_realtime('原油') 获取所有原油月份合约列表，"
        "再对每个具体合约调用 ak.futures_zh_daily_sina(symbol=合约代码)，"
        "提取最近一个交易日的收盘价。\n\n"
        "3. 期限结构判断规则\n"
        "若远月价格整体高于近月价格，则判定为升水结构(Contango)；"
        "若近月价格整体高于远月价格，则判定为贴水结构(Backwardation)。"
    )
    right_text = (
        "4. 关键指标\n"
        "1) 首末价差 = 远月收盘价 - 近月收盘价\n"
        "2) 首末价差幅度 = 首末价差 / 近月收盘价\n"
        "3) 整体斜率 = 对期限结构曲线做线性拟合后的斜率\n\n"
        "5. 分析意义\n"
        "期限结构反映市场对近远期供需、库存、仓储成本和资金成本的定价。"
        "因此，它不仅能描述市场状态，也能为跨期价差交易和展期收益分析提供依据。"
    )

    fig.text(0.08, 0.84, wrap_paragraph(left_text, width=28), fontsize=12, va="top", linespacing=1.7)
    fig.text(0.54, 0.84, wrap_paragraph(right_text, width=28), fontsize=12, va="top", linespacing=1.7)

    ax = fig.add_axes([0.08, 0.08, 0.84, 0.34])
    ax.axis("off")
    preview = term_df[["symbol", "contract_label", "close_price", "volume", "position"]].head(10).copy()
    preview.columns = ["合约", "月份", "收盘价", "成交量", "持仓量"]
    preview["收盘价"] = preview["收盘价"].map(lambda x: f"{x:.1f}")
    preview["成交量"] = preview["成交量"].map(lambda x: f"{int(x)}")
    preview["持仓量"] = preview["持仓量"].map(lambda x: f"{int(x)}")
    table = ax.table(
        cellText=preview.values,
        colLabels=preview.columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    ax.set_title("前 10 个合约样本", fontsize=13, pad=12)

    pdf.savefig(fig)
    plt.close(fig)


def create_result_page(pdf: PdfPages, term_df: pd.DataFrame, analysis: object) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    add_page_title(fig, "结果与判断")

    ax_img = fig.add_axes([0.08, 0.47, 0.84, 0.36])
    ax_img.axis("off")
    ax_img.imshow(plt.imread(PNG_PATH))
    ax_img.set_title("原油期货期限结构图", fontsize=13, pad=10)

    result_text = (
        f"1. 当前结构判断：{analysis.structure_name_cn}\n"
        f"2. 近月合约：{analysis.near_contract}，收盘价 {analysis.near_close:.2f} 元/桶\n"
        f"3. 远月合约：{analysis.far_contract}，收盘价 {analysis.far_close:.2f} 元/桶\n"
        f"4. 首末价差：{analysis.spread:+.2f} 元/桶，幅度 {analysis.spread_pct:+.2f}%\n"
        f"5. 整体斜率：{analysis.slope:+.2f} 元/桶/合约\n\n"
        f"{analysis.analysis_text}\n\n"
        "从图形上看，合约月份越远，价格整体越低，曲线向下倾斜较明显，"
        "说明市场更愿意为近期可交割原油支付更高价格。"
        "这一结果通常对应短期供给偏紧、库存偏低或近端需求更强的市场环境。"
    )
    fig.text(0.08, 0.40, wrap_paragraph(result_text, width=40), fontsize=12, va="top", linespacing=1.8)

    pdf.savefig(fig)
    plt.close(fig)


def create_strategy_page(pdf: PdfPages, analysis: object) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    add_page_title(fig, "交易思路与风险提示")

    strategy_lines = "\n".join([f"{idx}. {item}" for idx, item in enumerate(analysis.strategy_notes, start=1)])
    risk_text = (
        "风险提示：\n"
        "1. 期限结构会随地缘政治、OPEC+ 产量决策、库存变化和汇率波动快速变化。\n"
        "2. 跨期策略虽然方向风险相对单边更小，但仍然存在价差扩大而非收敛的风险。\n"
        "3. 远月合约成交量和持仓量通常较低，交易时应关注流动性与滑点。\n"
        "4. 本报告仅用于课程作业中的市场结构分析，不构成任何真实投资建议。"
    )
    conclusion_text = (
        "结论：\n"
        "本次程序设计作业完成了从数据抓取、结构识别、图形展示到策略讨论的完整流程。"
        "基于最近一次样本结果，上海原油期货呈现明显贴水结构，"
        "因此在分析上可重点讨论近端偏紧、远端折价以及跨期价差交易的逻辑。"
    )

    fig.text(0.08, 0.82, "可讨论的策略思路", fontsize=15, fontweight="bold")
    fig.text(0.08, 0.77, wrap_paragraph(strategy_lines, width=40), fontsize=12, va="top", linespacing=1.9)

    fig.text(0.08, 0.47, "风险提示", fontsize=15, fontweight="bold")
    fig.text(0.08, 0.42, wrap_paragraph(risk_text, width=40), fontsize=12, va="top", linespacing=1.8)

    fig.text(0.08, 0.20, "总结", fontsize=15, fontweight="bold")
    fig.text(0.08, 0.15, wrap_paragraph(conclusion_text, width=40), fontsize=12, va="top", linespacing=1.8)

    pdf.savefig(fig)
    plt.close(fig)


def main() -> None:
    configure_font()
    term_df, analysis = load_or_build_term_data()

    with PdfPages(PDF_PATH) as pdf:
        create_cover_page(pdf, term_df, analysis)
        create_method_page(pdf, term_df)
        create_result_page(pdf, term_df, analysis)
        create_strategy_page(pdf, analysis)

    print(f"PDF 报告已生成：{PDF_PATH}")


if __name__ == "__main__":
    main()
