"""
生成 PJ1 PDF 报告。
"""

from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MPLCONFIGDIR = BASE_DIR / ".mplconfig"
MPLCONFIGDIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR.resolve()))

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


SUMMARY_JSON = BASE_DIR / "performance_summary.json"
TRADE_LOG_CSV = BASE_DIR / "trade_log.csv"
PLOT_PATH = BASE_DIR / "strategy_backtest.png"
PDF_PATH = BASE_DIR / "pj1_report.pdf"


def configure_font() -> None:
    candidates = [
        "Noto Sans CJK JP",
        "Noto Sans CJK SC",
        "PingFang SC",
        "Heiti SC",
        "Arial Unicode MS",
    ]
    installed = {font.name for font in fm.fontManager.ttflist}
    for name in candidates:
        if name in installed:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False


def wrap_text(text: str, width: int = 38) -> str:
    parts: list[str] = []
    for block in text.splitlines():
        if not block.strip():
            parts.append("")
        else:
            parts.extend(textwrap.wrap(block, width=width))
    return "\n".join(parts)


def add_page_title(fig: plt.Figure, title: str, subtitle: str | None = None) -> None:
    fig.text(0.5, 0.95, title, ha="center", va="top", fontsize=22, fontweight="bold")
    if subtitle:
        fig.text(0.5, 0.915, subtitle, ha="center", va="top", fontsize=11, color="#444444")


def load_inputs() -> tuple[dict, pd.DataFrame]:
    with open(SUMMARY_JSON, "r", encoding="utf-8") as f:
        summary = json.load(f)
    trades = pd.read_csv(TRADE_LOG_CSV)
    return summary, trades


def cover_page(pdf: PdfPages, summary: dict) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    add_page_title(fig, "PJ1 原油期货双均线交易系统报告", "程序设计课程项目")

    intro = (
        "本项目基于上海国际能源交易中心原油主力连续合约 SC0 构建双均线趋势跟踪系统。"
        "系统包含开仓、平仓、资金管理、风险控制以及量化指标评估，"
        "目标是形成一套较完整、可运行、可回测的交易系统。"
    )
    tasks = (
        "1. 使用真实行情数据进行回测。\n"
        "2. 建立双均线开平仓规则。\n"
        "3. 引入 ATR 止损与跟踪止损。\n"
        "4. 使用风险预算控制仓位。\n"
        "5. 输出收益率、回撤、胜率、赔率、夏普比等指标。"
    )
    snapshot = (
        f"回测区间：{summary['start_date']} 至 {summary['end_date']}\n"
        f"标的：{summary['instrument']}\n"
        f"累计收益率：{summary['total_return']:.2%}\n"
        f"最大回撤：{summary['max_drawdown']:.2%}\n"
        f"夏普比率：{summary['sharpe_ratio']:.2f}"
    )

    fig.text(0.08, 0.82, "项目内容", fontsize=15, fontweight="bold")
    fig.text(0.08, 0.77, wrap_text(intro, 39), fontsize=12, va="top", linespacing=1.7)

    fig.text(0.08, 0.59, "完成任务", fontsize=15, fontweight="bold")
    fig.text(0.08, 0.55, tasks, fontsize=12, va="top", linespacing=1.8)

    fig.text(0.08, 0.33, "结果摘要", fontsize=15, fontweight="bold")
    fig.text(0.08, 0.29, snapshot, fontsize=12, va="top", linespacing=1.8)

    fig.text(0.08, 0.10, "数据来源：AkShare + 新浪期货主力连续数据", fontsize=11, color="#444444")
    plt.axis("off")
    pdf.savefig(fig)
    plt.close(fig)


def method_page(pdf: PdfPages) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    add_page_title(fig, "策略设计")

    left = (
        "开仓规则\n"
        "1. 当 MA10 上穿并持续高于 MA40 时建立多头仓位。\n"
        "2. 当 MA10 下穿并持续低于 MA40 时建立空头仓位。\n\n"
        "平仓规则\n"
        "1. 出现反向均线信号时平仓或反手。\n"
        "2. 价格触发 2 倍 ATR 硬止损时离场。\n"
        "3. 盈利后从阶段最优价格回撤超过 3 倍 ATR 时跟踪止盈。"
    )
    right = (
        "资金管理与风控\n"
        "1. 初始资金 1,000,000。\n"
        "2. 单笔风险预算为账户权益的 2%。\n"
        "3. 单边最大仓位上限 80%。\n"
        "4. 回测中考虑手续费 0.03% 与滑点 0.02%。\n\n"
        "策略逻辑\n"
        "该系统属于趋势跟踪范式，核心思想是让盈利头寸尽可能延续，"
        "同时通过 ATR 风控限制单笔亏损。"
    )

    fig.text(0.08, 0.83, wrap_text(left, 28), fontsize=12, va="top", linespacing=1.8)
    fig.text(0.54, 0.83, wrap_text(right, 28), fontsize=12, va="top", linespacing=1.8)

    pdf.savefig(fig)
    plt.close(fig)


def result_page(pdf: PdfPages, summary: dict, trades: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    add_page_title(fig, "回测结果")

    ax_img = fig.add_axes([0.08, 0.47, 0.84, 0.36])
    ax_img.axis("off")
    ax_img.imshow(plt.imread(PLOT_PATH))
    ax_img.set_title("价格走势与资金曲线", fontsize=13, pad=10)

    result_text = (
        f"期末权益：{summary['final_equity']:.2f}\n"
        f"累计收益率：{summary['total_return']:.2%}\n"
        f"年化收益率：{summary['annual_return']:.2%}\n"
        f"最大回撤：{summary['max_drawdown']:.2%}\n"
        f"夏普比率：{summary['sharpe_ratio']:.2f}\n"
        f"交易次数：{summary['trade_count']}\n"
        f"胜率：{summary['win_rate']:.2%}\n"
        f"赔率：{summary['payoff_ratio']:.2f}\n"
        f"Profit Factor：{summary['profit_factor']:.2f}\n\n"
        "结果显示该双均线系统在样本区间内取得了正收益。"
        "虽然胜率不高，但赔率大于 2，说明系统主要依靠趋势行情中的大盈亏比交易获利。"
    )
    fig.text(0.08, 0.40, result_text, fontsize=12, va="top", linespacing=1.8)

    ax_table = fig.add_axes([0.08, 0.08, 0.84, 0.18])
    ax_table.axis("off")
    preview = trades.head(8)[["entry_date", "exit_date", "direction", "entry_price", "exit_price", "pnl_amount", "exit_reason"]].copy()
    preview.columns = ["开仓日", "平仓日", "方向", "开仓价", "平仓价", "盈亏", "平仓原因"]
    table = ax_table.table(
        cellText=preview.values,
        colLabels=preview.columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    pdf.savefig(fig)
    plt.close(fig)


def conclusion_page(pdf: PdfPages, summary: dict) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    add_page_title(fig, "结论与改进方向")

    conclusion = (
        "结论\n"
        "本项目已经形成一套较完整的量化交易系统，包含信号生成、仓位控制、"
        "止损止盈、交易成本建模和绩效评估。"
        "在原油主力连续合约样本中，策略实现了正收益，说明双均线趋势跟踪方法具有一定可行性。"
    )
    risks = (
        "不足与风险\n"
        "1. 最大回撤仍达到 23% 左右，说明趋势失效阶段会对净值造成明显冲击。\n"
        "2. 夏普比率不高，收益质量仍有优化空间。\n"
        "3. 主力连续合约存在换月拼接问题，真实交易效果可能与回测不同。"
    )
    improve = (
        "后续可改进方向\n"
        "1. 增加波动率过滤或突破确认条件。\n"
        "2. 引入不同市场状态下的参数切换。\n"
        "3. 加入更精细的合约乘数、保证金和换月处理。\n"
        "4. 增加组合层面的风控指标，例如卡玛比率和分阶段绩效。"
    )

    fig.text(0.08, 0.82, wrap_text(conclusion, 40), fontsize=12, va="top", linespacing=1.8)
    fig.text(0.08, 0.58, wrap_text(risks, 40), fontsize=12, va="top", linespacing=1.8)
    fig.text(0.08, 0.34, wrap_text(improve, 40), fontsize=12, va="top", linespacing=1.8)

    fig.text(0.08, 0.12, f"本次回测累计收益率 {summary['total_return']:.2%}，最大回撤 {summary['max_drawdown']:.2%}。", fontsize=11, color="#444444")
    pdf.savefig(fig)
    plt.close(fig)


def main() -> None:
    configure_font()
    summary, trades = load_inputs()
    with PdfPages(PDF_PATH) as pdf:
        cover_page(pdf, summary)
        method_page(pdf)
        result_page(pdf, summary, trades)
        conclusion_page(pdf, summary)
    print(f"PDF 报告已生成：{PDF_PATH}")


if __name__ == "__main__":
    main()
