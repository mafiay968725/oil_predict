"""
国内原油期货期限结构分析
功能：
1. 获取上海国际能源交易中心原油期货(原油/SC)各月份合约
2. 提取最近一个交易日的收盘价
3. 绘制期限结构图
4. 判断升水(Contango)或贴水(Backwardation)
5. 输出适合作业展示的简要分析和策略思路
"""

from __future__ import annotations

import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MPLCONFIGDIR = BASE_DIR / ".mplconfig"
MPLCONFIGDIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR.resolve()))

import akshare as ak
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def configure_font() -> None:
    """优先使用可显示中文的字体。"""
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


@dataclass
class StructureAnalysis:
    structure_type: str
    structure_name_cn: str
    near_contract: str
    far_contract: str
    near_close: float
    far_close: float
    spread: float
    spread_pct: float
    slope: float
    analysis_text: str
    strategy_notes: list[str]


class CrudeTermStructureAnalyzer:
    def __init__(self, output_dir: str | Path = BASE_DIR) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.term_df: pd.DataFrame | None = None

    @staticmethod
    def _parse_contract_month(symbol: str) -> pd.Timestamp | pd.NaT:
        match = re.fullmatch(r"SC(\d{2})(\d{2})", str(symbol))
        if not match:
            return pd.NaT
        year = 2000 + int(match.group(1))
        month = int(match.group(2))
        return pd.Timestamp(year=year, month=month, day=1)

    @staticmethod
    def _format_contract_month(ts: pd.Timestamp) -> str:
        return ts.strftime("%Y-%m")

    def fetch_contract_snapshot(self) -> pd.DataFrame:
        """获取原油品种当前所有可交易合约的快照。"""
        realtime_df = ak.futures_zh_realtime(symbol="原油")
        realtime_df = realtime_df.copy()
        realtime_df = realtime_df[realtime_df["symbol"].astype(str).str.fullmatch(r"SC\d{4}")]
        realtime_df["contract_month"] = realtime_df["symbol"].map(self._parse_contract_month)
        realtime_df = realtime_df.dropna(subset=["contract_month"]).sort_values("contract_month").reset_index(drop=True)
        if realtime_df.empty:
            raise ValueError("未获取到有效的原油月份合约列表。")
        return realtime_df

    def fetch_latest_close(self, symbol: str) -> dict:
        """获取单个合约最近一个交易日的收盘价。"""
        daily_df = ak.futures_zh_daily_sina(symbol=symbol)
        daily_df = daily_df.copy()
        daily_df["date"] = pd.to_datetime(daily_df["date"])
        daily_df = daily_df.dropna(subset=["close"]).sort_values("date")
        if daily_df.empty:
            raise ValueError(f"{symbol} 未获取到有效日线收盘价。")

        latest = daily_df.iloc[-1]
        return {
            "symbol": symbol,
            "close_date": latest["date"].date(),
            "close_price": float(latest["close"]),
            "open_price": float(latest["open"]),
            "high_price": float(latest["high"]),
            "low_price": float(latest["low"]),
            "daily_volume": float(latest["volume"]),
            "daily_hold": float(latest["hold"]),
        }

    def build_term_structure(self) -> pd.DataFrame:
        """整合原油各月份合约收盘价。"""
        snapshot_df = self.fetch_contract_snapshot()
        records: list[dict] = []

        print("=" * 72)
        print("获取上海国际能源交易中心原油期货期限结构数据")
        print("=" * 72)
        print(f"共发现 {len(snapshot_df)} 个原油月份合约，开始抓取最近收盘价...\n")

        for idx, row in snapshot_df.iterrows():
            symbol = row["symbol"]
            print(f"[{idx + 1:02d}/{len(snapshot_df):02d}] 获取 {symbol} ...", end="")
            try:
                daily_info = self.fetch_latest_close(symbol)
                record = {
                    "symbol": symbol,
                    "contract_month": row["contract_month"],
                    "contract_label": self._format_contract_month(row["contract_month"]),
                    "close_date": daily_info["close_date"],
                    "close_price": daily_info["close_price"],
                    "trade_price": float(row["trade"]),
                    "pre_settlement": float(row["presettlement"]),
                    "volume": float(row["volume"]),
                    "position": float(row["position"]),
                    "tick_time": row["ticktime"],
                }
                records.append(record)
                print(f" 完成，最近收盘价={daily_info['close_price']:.2f}")
            except Exception as exc:
                print(f" 失败：{exc}")

        if not records:
            raise RuntimeError("未能获取任何原油合约的收盘价数据。")

        term_df = pd.DataFrame(records).sort_values("contract_month").reset_index(drop=True)
        term_df["month_index"] = np.arange(len(term_df))
        term_df["spread_vs_near"] = term_df["close_price"] - term_df.loc[0, "close_price"]
        self.term_df = term_df
        return term_df

    @staticmethod
    def analyze_structure(term_df: pd.DataFrame) -> StructureAnalysis:
        """判断期限结构形态并给出分析。"""
        near = term_df.iloc[0]
        far = term_df.iloc[-1]
        spread = float(far["close_price"] - near["close_price"])
        spread_pct = spread / float(near["close_price"]) * 100
        slope = float(np.polyfit(term_df["month_index"], term_df["close_price"], 1)[0])

        threshold = max(abs(float(near["close_price"])) * 0.005, 2.0)
        if spread > threshold:
            structure_type = "contango"
            structure_name_cn = "升水结构"
            analysis_text = (
                "远月合约价格整体高于近月合约，期限结构向上倾斜，"
                "说明市场更愿意为未来交割支付更高价格，常见于库存相对宽松、"
                "持仓成本和仓储成本较高的阶段。"
            )
            strategy_notes = [
                "若预期升水收敛，可考虑做多近月、做空远月的跨期价差策略。",
                "若具备库存与交割条件，可从持有现货并卖出远月合约的正向套利思路理解升水结构。",
                "滚动多头时需警惕负展期收益，因换月通常要从较低近月换到较高远月。",
            ]
        elif spread < -threshold:
            structure_type = "backwardation"
            structure_name_cn = "贴水结构"
            analysis_text = (
                "近月合约价格高于远月合约，期限结构向下倾斜，"
                "通常意味着短期供应偏紧或现货需求较强，市场为近期交割支付溢价。"
            )
            strategy_notes = [
                "若预期贴水回归正常，可考虑做空近月、做多远月的跨期价差策略。",
                "趋势型多头在贴水环境下往往能获得正展期收益，但需防止近月高位回落。",
                "若现货紧张缓解，近月相对远月的高溢价可能快速收敛。",
            ]
        else:
            structure_type = "flat"
            structure_name_cn = "平坦结构"
            analysis_text = (
                "近远月价格差异较小，期限结构较平，说明当前市场对近期与远期供需的定价较为均衡。"
            )
            strategy_notes = [
                "期限结构信号较弱，更适合结合库存、裂解价差、宏观变量再决定方向。",
                "跨期策略可等待价差扩大后再介入，避免在低波动区间频繁交易。",
            ]

        return StructureAnalysis(
            structure_type=structure_type,
            structure_name_cn=structure_name_cn,
            near_contract=str(near["symbol"]),
            far_contract=str(far["symbol"]),
            near_close=float(near["close_price"]),
            far_close=float(far["close_price"]),
            spread=spread,
            spread_pct=spread_pct,
            slope=slope,
            analysis_text=analysis_text,
            strategy_notes=strategy_notes,
        )

    def save_term_table(self, term_df: pd.DataFrame) -> Path:
        output_path = self.output_dir / "crude_term_structure.csv"
        export_cols = [
            "symbol",
            "contract_label",
            "close_date",
            "close_price",
            "trade_price",
            "pre_settlement",
            "volume",
            "position",
            "spread_vs_near",
        ]
        term_df[export_cols].to_csv(output_path, index=False, encoding="utf-8-sig")
        return output_path

    def plot_term_structure(self, term_df: pd.DataFrame, analysis: StructureAnalysis) -> Path:
        configure_font()

        fig, ax = plt.subplots(figsize=(12, 7))
        color = "#d95f02" if analysis.structure_type == "contango" else "#1b9e77"
        if analysis.structure_type == "flat":
            color = "#4c78a8"

        ax.plot(
            term_df["contract_label"],
            term_df["close_price"],
            marker="o",
            linewidth=2.5,
            markersize=7,
            color=color,
            label="最近交易日收盘价",
        )

        for _, row in term_df.iterrows():
            ax.annotate(
                f"{row['close_price']:.1f}",
                (row["contract_label"], row["close_price"]),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=9,
            )

        ax.set_title("上海原油期货期限结构图", fontsize=15, fontweight="bold")
        ax.set_xlabel("合约月份")
        ax.set_ylabel("收盘价 (元/桶)")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="best")
        plt.xticks(rotation=45)

        summary = (
            f"结构判断: {analysis.structure_name_cn}\n"
            f"近月 {analysis.near_contract}: {analysis.near_close:.2f}\n"
            f"远月 {analysis.far_contract}: {analysis.far_close:.2f}\n"
            f"首末价差: {analysis.spread:+.2f} ({analysis.spread_pct:+.2f}%)"
        )
        ax.text(
            0.02,
            0.98,
            summary,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.9},
        )

        plt.tight_layout()
        output_path = self.output_dir / "crude_term_structure.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return output_path

    @staticmethod
    def print_summary(term_df: pd.DataFrame, analysis: StructureAnalysis) -> None:
        data_date = term_df["close_date"].max()

        print("\n" + "=" * 72)
        print("原油期限结构分析结果")
        print("=" * 72)
        print(f"样本日期: {data_date}")
        print(f"合约数量: {len(term_df)}")
        print(f"结构判断: {analysis.structure_name_cn} ({analysis.structure_type})")
        print(f"近月合约: {analysis.near_contract}，收盘价 {analysis.near_close:.2f} 元/桶")
        print(f"远月合约: {analysis.far_contract}，收盘价 {analysis.far_close:.2f} 元/桶")
        print(f"首末价差: {analysis.spread:+.2f} 元/桶，幅度 {analysis.spread_pct:+.2f}%")
        print(f"整体斜率: {analysis.slope:+.2f} 元/桶/合约")

        print("\n【期限结构说明】")
        print(analysis.analysis_text)

        print("\n【适合作业中写的交易思路】")
        for idx, note in enumerate(analysis.strategy_notes, start=1):
            print(f"{idx}. {note}")

        print("\n【前10个合约样本】")
        preview_cols = ["symbol", "contract_label", "close_date", "close_price", "trade_price", "volume", "position"]
        print(term_df[preview_cols].head(10).to_string(index=False))


def main() -> None:
    analyzer = CrudeTermStructureAnalyzer(output_dir=BASE_DIR)
    term_df = analyzer.build_term_structure()
    analysis = analyzer.analyze_structure(term_df)
    csv_path = analyzer.save_term_table(term_df)
    plot_path = analyzer.plot_term_structure(term_df, analysis)
    analyzer.print_summary(term_df, analysis)

    print("\n输出文件：")
    print(f"- {csv_path}")
    print(f"- {plot_path}")


if __name__ == "__main__":
    main()
