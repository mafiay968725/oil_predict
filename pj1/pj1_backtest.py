"""
PJ1: 原油期货双均线交易系统回测

交易系统包含：
1. 开仓规则：20日均线上穿/下穿60日均线
2. 平仓规则：反向信号、ATR止损、ATR跟踪止损
3. 资金管理：单笔风险2%，最大仓位80%
4. 风控：手续费、滑点、硬止损、跟踪止损
5. 输出指标：胜率、赔率、最大回撤、收益率、夏普比等
"""

from __future__ import annotations

import json
import math
import os
import warnings
from dataclasses import dataclass
from datetime import datetime
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
class StrategyConfig:
    symbol: str = "SC0"
    start_date: str = "20190101"
    end_date: str = datetime.today().strftime("%Y%m%d")
    ma_fast: int = 10
    ma_slow: int = 40
    atr_window: int = 14
    stop_atr: float = 2.0
    trailing_atr: float = 3.0
    initial_capital: float = 1_000_000.0
    risk_per_trade: float = 0.02
    max_position_size: float = 0.80
    fee_rate: float = 0.0003
    slippage_rate: float = 0.0002


class MovingAverageTradingSystem:
    def __init__(self, config: StrategyConfig, output_dir: Path) -> None:
        self.config = config
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data: pd.DataFrame | None = None
        self.backtest_df: pd.DataFrame | None = None
        self.trade_log: pd.DataFrame | None = None
        self.metrics: dict[str, float | int | str] | None = None

    def fetch_data(self) -> pd.DataFrame:
        print("=" * 72)
        print("获取原油主力连续合约历史数据")
        print("=" * 72)
        raw_df = ak.futures_main_sina(
            symbol=self.config.symbol,
            start_date=self.config.start_date,
            end_date=self.config.end_date,
        )
        df = raw_df.copy()
        df = df.rename(
            columns={
                "日期": "date",
                "开盘价": "open",
                "最高价": "high",
                "最低价": "low",
                "收盘价": "close",
                "成交量": "volume",
                "持仓量": "open_interest",
            }
        )
        df["date"] = pd.to_datetime(df["date"])
        numeric_cols = ["open", "high", "low", "close", "volume", "open_interest"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["date", "open", "high", "low", "close"])
        df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
        self.data = df

        print(
            f"数据区间: {df['date'].min().date()} 至 {df['date'].max().date()}, "
            f"共 {len(df)} 条记录"
        )
        return df

    def prepare_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["ma_fast"] = df["close"].rolling(self.config.ma_fast).mean()
        df["ma_slow"] = df["close"].rolling(self.config.ma_slow).mean()

        prev_close = df["close"].shift(1)
        tr_components = pd.concat(
            [
                (df["high"] - df["low"]).abs(),
                (df["high"] - prev_close).abs(),
                (df["low"] - prev_close).abs(),
            ],
            axis=1,
        )
        df["tr"] = tr_components.max(axis=1)
        df["atr"] = df["tr"].rolling(self.config.atr_window).mean()

        df["signal"] = 0
        df.loc[df["ma_fast"] > df["ma_slow"], "signal"] = 1
        df.loc[df["ma_fast"] < df["ma_slow"], "signal"] = -1
        return df

    def _target_weight(self, row: pd.Series) -> float:
        atr = float(row["atr"])
        close = float(row["close"])
        if np.isnan(atr) or atr <= 0 or close <= 0:
            return 0.0
        stop_pct = max((self.config.stop_atr * atr) / close, 0.01)
        return min(self.config.max_position_size, self.config.risk_per_trade / stop_pct)

    def run_backtest(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        equity = self.config.initial_capital
        position = 0
        weight = 0.0
        entry_price: float | None = None
        entry_date: pd.Timestamp | None = None
        entry_equity: float | None = None
        entry_atr: float | None = None
        peak_price: float | None = None
        trough_price: float | None = None
        peak_equity = equity

        records: list[dict] = []
        trades: list[dict] = []

        for i, row in df.iterrows():
            date = row["date"]
            close = float(row["close"])
            high = float(row["high"])
            low = float(row["low"])
            signal = int(row["signal"])

            prev_close = close if i == 0 else float(df.iloc[i - 1]["close"])
            daily_return = 0.0 if i == 0 else (close / prev_close - 1.0)
            gross_pnl_rate = position * weight * daily_return
            equity_before_cost = equity * (1.0 + gross_pnl_rate)

            if position > 0:
                peak_price = close if peak_price is None else max(peak_price, close)
            elif position < 0:
                trough_price = close if trough_price is None else min(trough_price, close)

            planned_position = position
            planned_weight = weight
            exit_reason = ""
            action = "HOLD"

            stop_triggered = False
            if position > 0 and entry_price is not None and entry_atr is not None:
                hard_stop = entry_price - self.config.stop_atr * entry_atr
                trail_stop = (
                    peak_price - self.config.trailing_atr * float(row["atr"])
                    if peak_price is not None and not np.isnan(float(row["atr"]))
                    else -math.inf
                )
                if low <= hard_stop:
                    planned_position = 0
                    planned_weight = 0.0
                    exit_reason = "hard_stop"
                    stop_triggered = True
                elif low <= trail_stop:
                    planned_position = 0
                    planned_weight = 0.0
                    exit_reason = "trailing_stop"
                    stop_triggered = True
            elif position < 0 and entry_price is not None and entry_atr is not None:
                hard_stop = entry_price + self.config.stop_atr * entry_atr
                trail_stop = (
                    trough_price + self.config.trailing_atr * float(row["atr"])
                    if trough_price is not None and not np.isnan(float(row["atr"]))
                    else math.inf
                )
                if high >= hard_stop:
                    planned_position = 0
                    planned_weight = 0.0
                    exit_reason = "hard_stop"
                    stop_triggered = True
                elif high >= trail_stop:
                    planned_position = 0
                    planned_weight = 0.0
                    exit_reason = "trailing_stop"
                    stop_triggered = True

            if not stop_triggered:
                if signal != position:
                    planned_position = signal
                    planned_weight = self._target_weight(row) if signal != 0 else 0.0
                    if position == 0 and signal != 0:
                        action = "OPEN"
                    elif position != 0 and signal == 0:
                        action = "CLOSE"
                        exit_reason = "trend_exit"
                    elif position != 0 and signal == -position:
                        action = "REVERSE"
                        exit_reason = "reverse_signal"
                else:
                    planned_weight = weight
            else:
                action = "STOP_EXIT"

            turnover = abs(planned_position * planned_weight - position * weight)
            trading_cost_rate = turnover * (self.config.fee_rate + self.config.slippage_rate)
            equity_after_cost = equity_before_cost * (1.0 - trading_cost_rate)

            if position != 0 and planned_position != position:
                assert entry_date is not None
                assert entry_price is not None
                assert entry_equity is not None
                direction = "long" if position > 0 else "short"
                trades.append(
                    {
                        "entry_date": entry_date.date(),
                        "exit_date": date.date(),
                        "direction": direction,
                        "entry_price": round(entry_price, 4),
                        "exit_price": round(close, 4),
                        "position_size": round(weight, 4),
                        "holding_bars": i - df.index[df["date"] == entry_date][0],
                        "entry_equity": round(entry_equity, 2),
                        "exit_equity": round(equity_after_cost, 2),
                        "pnl_amount": round(equity_after_cost - entry_equity, 2),
                        "return_pct": round(equity_after_cost / entry_equity - 1.0, 6),
                        "exit_reason": exit_reason or "signal_change",
                    }
                )

            if planned_position != position:
                if planned_position != 0:
                    entry_date = date
                    entry_price = close
                    entry_equity = equity_after_cost
                    entry_atr = float(row["atr"]) if not np.isnan(float(row["atr"])) else None
                    peak_price = close
                    trough_price = close
                else:
                    entry_date = None
                    entry_price = None
                    entry_equity = None
                    entry_atr = None
                    peak_price = None
                    trough_price = None

            equity = equity_after_cost
            position = planned_position
            weight = planned_weight
            peak_equity = max(peak_equity, equity)
            drawdown = equity / peak_equity - 1.0

            records.append(
                {
                    "date": date,
                    "close": close,
                    "ma_fast": row["ma_fast"],
                    "ma_slow": row["ma_slow"],
                    "atr": row["atr"],
                    "signal": signal,
                    "position": position,
                    "position_size": weight,
                    "daily_return": daily_return,
                    "strategy_return": gross_pnl_rate - trading_cost_rate,
                    "equity": equity,
                    "drawdown": drawdown,
                    "action": action,
                }
            )

        backtest_df = pd.DataFrame(records)
        trade_log = pd.DataFrame(trades)
        self.backtest_df = backtest_df
        self.trade_log = trade_log
        return backtest_df, trade_log

    def calculate_metrics(
        self, backtest_df: pd.DataFrame, trade_log: pd.DataFrame
    ) -> dict[str, float | int | str]:
        daily_returns = backtest_df["equity"].pct_change().dropna()
        total_return = backtest_df["equity"].iloc[-1] / self.config.initial_capital - 1.0
        annual_return = (1.0 + total_return) ** (252 / max(len(backtest_df), 1)) - 1.0
        max_drawdown = float(backtest_df["drawdown"].min())
        sharpe = 0.0
        if daily_returns.std(ddof=0) > 0:
            sharpe = float(np.sqrt(252) * daily_returns.mean() / daily_returns.std(ddof=0))

        winning_trades = trade_log[trade_log["pnl_amount"] > 0]
        losing_trades = trade_log[trade_log["pnl_amount"] <= 0]
        win_rate = len(winning_trades) / len(trade_log) if len(trade_log) > 0 else 0.0
        avg_win = float(winning_trades["pnl_amount"].mean()) if not winning_trades.empty else 0.0
        avg_loss = float(losing_trades["pnl_amount"].mean()) if not losing_trades.empty else 0.0
        payoff_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0

        metrics: dict[str, float | int | str] = {
            "instrument": self.config.symbol,
            "start_date": str(backtest_df["date"].min().date()),
            "end_date": str(backtest_df["date"].max().date()),
            "initial_capital": round(self.config.initial_capital, 2),
            "final_equity": round(float(backtest_df["equity"].iloc[-1]), 2),
            "total_return": round(float(total_return), 6),
            "annual_return": round(float(annual_return), 6),
            "max_drawdown": round(max_drawdown, 6),
            "sharpe_ratio": round(sharpe, 6),
            "trade_count": int(len(trade_log)),
            "win_rate": round(float(win_rate), 6),
            "payoff_ratio": round(float(payoff_ratio), 6),
            "avg_trade_return": round(float(trade_log["return_pct"].mean()), 6) if len(trade_log) > 0 else 0.0,
            "profit_factor": round(
                float(winning_trades["pnl_amount"].sum() / abs(losing_trades["pnl_amount"].sum())),
                6,
            )
            if not winning_trades.empty and not losing_trades.empty and losing_trades["pnl_amount"].sum() != 0
            else 0.0,
        }
        self.metrics = metrics
        return metrics

    def save_outputs(
        self, backtest_df: pd.DataFrame, trade_log: pd.DataFrame, metrics: dict[str, float | int | str]
    ) -> None:
        backtest_df.to_csv(self.output_dir / "backtest_daily.csv", index=False, encoding="utf-8-sig")
        trade_log.to_csv(self.output_dir / "trade_log.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame([metrics]).to_csv(
            self.output_dir / "performance_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
        with open(self.output_dir / "performance_summary.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

    def plot_results(self, df: pd.DataFrame, backtest_df: pd.DataFrame, trade_log: pd.DataFrame) -> None:
        configure_font()
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True, height_ratios=[3, 2])

        axes[0].plot(df["date"], df["close"], label="收盘价", color="#1f77b4", linewidth=1.4)
        axes[0].plot(df["date"], df["ma_fast"], label=f"MA{self.config.ma_fast}", color="#ff7f0e", linewidth=1.2)
        axes[0].plot(df["date"], df["ma_slow"], label=f"MA{self.config.ma_slow}", color="#2ca02c", linewidth=1.2)

        if not trade_log.empty:
            long_entries = trade_log[trade_log["direction"] == "long"]
            short_entries = trade_log[trade_log["direction"] == "short"]
            axes[0].scatter(
                pd.to_datetime(long_entries["entry_date"]),
                long_entries["entry_price"],
                marker="^",
                color="#d62728",
                s=40,
                label="多头开仓",
            )
            axes[0].scatter(
                pd.to_datetime(short_entries["entry_date"]),
                short_entries["entry_price"],
                marker="v",
                color="#9467bd",
                s=40,
                label="空头开仓",
            )

        axes[0].set_title("原油主力连续合约双均线交易系统", fontsize=14, fontweight="bold")
        axes[0].set_ylabel("价格")
        axes[0].grid(True, alpha=0.25)
        axes[0].legend(loc="best")

        axes[1].plot(backtest_df["date"], backtest_df["equity"], color="#111111", linewidth=1.6, label="资金曲线")
        axes[1].fill_between(
            backtest_df["date"],
            backtest_df["equity"],
            backtest_df["equity"].cummax(),
            color="#ff9896",
            alpha=0.25,
            label="回撤区间",
        )
        axes[1].set_ylabel("账户权益")
        axes[1].set_xlabel("日期")
        axes[1].grid(True, alpha=0.25)
        axes[1].legend(loc="best")

        plt.tight_layout()
        plt.savefig(self.output_dir / "strategy_backtest.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    def print_summary(self, metrics: dict[str, float | int | str]) -> None:
        print("\n" + "=" * 72)
        print("双均线交易系统回测结果")
        print("=" * 72)
        print(f"品种: {metrics['instrument']}")
        print(f"区间: {metrics['start_date']} 至 {metrics['end_date']}")
        print(f"初始资金: {metrics['initial_capital']:.2f}")
        print(f"期末权益: {metrics['final_equity']:.2f}")
        print(f"累计收益率: {metrics['total_return']:.2%}")
        print(f"年化收益率: {metrics['annual_return']:.2%}")
        print(f"最大回撤: {metrics['max_drawdown']:.2%}")
        print(f"夏普比率: {metrics['sharpe_ratio']:.2f}")
        print(f"交易次数: {metrics['trade_count']}")
        print(f"胜率: {metrics['win_rate']:.2%}")
        print(f"赔率: {metrics['payoff_ratio']:.2f}")
        print(f"平均单笔收益率: {metrics['avg_trade_return']:.2%}")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")


def main() -> None:
    config = StrategyConfig()
    system = MovingAverageTradingSystem(config=config, output_dir=BASE_DIR)
    raw_df = system.fetch_data()
    df = system.prepare_indicators(raw_df)
    backtest_df, trade_log = system.run_backtest(df)
    metrics = system.calculate_metrics(backtest_df, trade_log)
    system.save_outputs(backtest_df, trade_log, metrics)
    system.plot_results(df, backtest_df, trade_log)
    system.print_summary(metrics)


if __name__ == "__main__":
    main()
