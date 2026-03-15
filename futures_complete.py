"""
外盘期货走势展示与预测系统 - 完整版本（单文件）
功能：获取原油期货数据、展示合约信息、技术指标分析、价格预测
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

import matplotlib.font_manager as fm
for _f in fm.fontManager.ttflist:
    if 'Noto Sans CJK JP' in _f.name:
        plt.rcParams['font.family'] = _f.name
        break

class FuturesAnalyzer:
    def __init__(self):
        self.data = {}

    def get_sample_data(self):
        """使用akshare获取真实期货数据"""
        import akshare as ak

        print("=" * 70)
        print("Fetching real futures data via akshare...")
        print("=" * 70)

        try:
            print("\n[WTI Crude Oil (CL)]")
            wti_data = ak.futures_foreign_hist(symbol="CL")
            wti_data['date'] = pd.to_datetime(wti_data['date'])
            wti_data = wti_data.sort_values('date').tail(252).reset_index(drop=True)
            self.data['WTI'] = wti_data
            print(f"✓ WTI data fetched: {len(wti_data)} rows, latest: ${wti_data.iloc[-1]['close']:.2f}")
        except Exception as e:
            print(f"✗ WTI fetch failed: {e}")

        try:
            print("\n[Brent Crude Oil (OIL)]")
            brent_data = ak.futures_foreign_hist(symbol="OIL")
            brent_data['date'] = pd.to_datetime(brent_data['date'])
            brent_data = brent_data.sort_values('date').tail(252).reset_index(drop=True)
            self.data['Brent'] = brent_data
            print(f"✓ Brent data fetched: {len(brent_data)} rows, latest: ${brent_data.iloc[-1]['close']:.2f}")
        except Exception as e:
            print(f"✗ Brent fetch failed: {e}")

        return len(self.data) > 0

    def calculate_indicators(self, symbol):
        """计算技术指标"""
        if symbol not in self.data:
            return None

        data = self.data[symbol].copy()

        # 移动平均线
        data['MA5'] = data['close'].rolling(window=5).mean()
        data['MA10'] = data['close'].rolling(window=10).mean()
        data['MA20'] = data['close'].rolling(window=20).mean()

        # RSI指标
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))

        # MACD指标
        exp1 = data['close'].ewm(span=12, adjust=False).mean()
        exp2 = data['close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = exp1 - exp2
        data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
        data['Histogram'] = data['MACD'] - data['Signal']

        # 布林带
        data['BB_Middle'] = data['close'].rolling(window=20).mean()
        data['BB_Std'] = data['close'].rolling(window=20).std()
        data['BB_Upper'] = data['BB_Middle'] + (data['BB_Std'] * 2)
        data['BB_Lower'] = data['BB_Middle'] - (data['BB_Std'] * 2)

        self.data[symbol] = data
        return data

    def display_contract_info(self):
        """展示合约信息"""
        print("\n" + "=" * 70)
        print("主力合约详细信息")
        print("=" * 70)

        contracts_info = {
            'WTI': {
                'name': 'WTI原油',
                'exchange': 'NYMEX',
                'multiplier': 100,
                'unit': 'BBL',
                'tick_size': 0.01
            },
            'Brent': {
                'name': 'Brent原油',
                'exchange': 'ICE',
                'multiplier': 100,
                'unit': 'BBL',
                'tick_size': 0.01
            }
        }

        for symbol, config in contracts_info.items():
            if symbol not in self.data:
                continue

            data = self.data[symbol]
            latest = data.iloc[-1]

            print(f"\n【{config['name']}】")
            print("-" * 70)
            print(f"交易所: {config['exchange']}")
            print(f"合约乘数: {config['multiplier']} USD/点")
            print(f"最小变动单位: {config['tick_size']}")
            print(f"计价单位: {config['unit']}")

            print(f"\n【价格信息】")
            print(f"  最新价格: ${latest['close']:.2f}")
            print(f"  开盘价:   ${latest['open']:.2f}")
            print(f"  最高价:   ${latest['high']:.2f}")
            print(f"  最低价:   ${latest['low']:.2f}")

            if len(data) > 1:
                prev_close = data.iloc[-2]['close']
                change = latest['close'] - prev_close
                change_pct = (change / prev_close) * 100
                print(f"  涨跌:     {change:+.2f} ({change_pct:+.2f}%)")

            print(f"\n【成交持仓】")
            print(f"  成交量:   {latest['volume']:.0f}")

            if len(data) > 1:
                vol_change = latest['volume'] - data.iloc[-2]['volume']
                print(f"  成交量变化: {vol_change:+.0f}")

            print(f"\n【期限结构】")
            print(f"  数据周期: {data['date'].min().date()} 至 {data['date'].max().date()}")
            print(f"  平均价格: ${data['close'].mean():.2f}")
            print(f"  价格范围: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
            print(f"  波动率:   {data['close'].std():.2f}")

            print(f"\n【技术指标】")
            print(f"  MA5:      ${latest['MA5']:.2f}")
            print(f"  MA10:     ${latest['MA10']:.2f}")
            print(f"  MA20:     ${latest['MA20']:.2f}")
            print(f"  RSI(14):  {latest['RSI']:.2f}")
            print(f"  MACD:     {latest['MACD']:.4f}")

    def forecast_arima(self, symbol, periods=30):
        """ARIMA预测"""
        print(f"\n正在进行{symbol}原油ARIMA预测...")

        try:
            from statsmodels.tsa.arima.model import ARIMA
        except ImportError:
            print("需要安装statsmodels")
            return None

        if symbol not in self.data:
            return None

        data = self.data[symbol].copy()
        ts_data = data['close'].reset_index(drop=True)

        try:
            model = ARIMA(ts_data, order=(1, 1, 1))
            fitted_model = model.fit()

            forecast = fitted_model.get_forecast(steps=periods)
            forecast_df = forecast.conf_int(alpha=0.05)
            forecast_df['forecast'] = forecast.predicted_mean.values

            print(f"✓ ARIMA模型拟合完成")
            print(f"  模型AIC: {fitted_model.aic:.2f}")

            return {
                'model': fitted_model,
                'forecast': forecast_df,
                'historical': data,
                'symbol': symbol
            }
        except Exception as e:
            print(f"✗ ARIMA预测失败: {e}")
            return None

    def plot_analysis(self, forecast_result):
        """绘制分析图表"""
        if forecast_result is None:
            return

        symbol = forecast_result['symbol']
        historical = forecast_result['historical']
        forecast_df = forecast_result['forecast']

        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        # 图1: 价格走势和预测
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(historical['date'], historical['close'], 'b-', label='Historical Price', linewidth=2)

        forecast_dates = pd.date_range(start=historical['date'].iloc[-1], periods=len(forecast_df)+1, freq='D')[1:]
        ax1.plot(forecast_dates, forecast_df['forecast'], 'r--', label='Forecast', linewidth=2)
        ax1.fill_between(forecast_dates,
                         forecast_df.iloc[:, 0],
                         forecast_df.iloc[:, 1],
                         alpha=0.2, color='red', label='95% Confidence Interval')

        symbol_cn = 'WTI' if symbol == 'WTI' else 'Brent'
        ax1.set_title(f'{symbol_cn}原油期货价格走势与预测', fontsize=14, fontweight='bold')
        ax1.set_ylabel('价格 (USD/BBL)')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)

        # 图2: 移动平均线
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(historical['date'], historical['close'], 'b-', label='收盘价', linewidth=1.5)
        ax2.plot(historical['date'], historical['MA5'], 'r--', label='MA5', alpha=0.7)
        ax2.plot(historical['date'], historical['MA10'], 'g--', label='MA10', alpha=0.7)
        ax2.plot(historical['date'], historical['MA20'], 'orange', linestyle='--', label='MA20', alpha=0.7)
        ax2.set_title('移动平均线', fontsize=12, fontweight='bold')
        ax2.set_ylabel('价格 (USD/BBL)')
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(True, alpha=0.3)

        # 图3: RSI
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.plot(historical['date'], historical['RSI'], 'purple', linewidth=2, label='RSI(14)')
        ax3.axhline(y=70, color='r', linestyle='--', alpha=0.5, label='超买 (70)')
        ax3.axhline(y=30, color='g', linestyle='--', alpha=0.5, label='超卖 (30)')
        ax3.fill_between(historical['date'], 30, 70, alpha=0.1, color='gray')
        ax3.set_title('相对强弱指数 (RSI)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('RSI')
        ax3.set_ylim([0, 100])
        ax3.legend(loc='best', fontsize=9)
        ax3.grid(True, alpha=0.3)

        # 图4: MACD
        ax4 = fig.add_subplot(gs[2, 0])
        ax4.plot(historical['date'], historical['MACD'], 'b-', label='MACD', linewidth=1.5)
        ax4.plot(historical['date'], historical['Signal'], 'r-', label='信号线', linewidth=1.5)
        ax4.bar(historical['date'], historical['Histogram'], label='柱状图', alpha=0.3)
        ax4.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax4.set_title('MACD指标', fontsize=12, fontweight='bold')
        ax4.set_ylabel('MACD值')
        ax4.legend(loc='best', fontsize=9)
        ax4.grid(True, alpha=0.3)

        # 图5: 价格分布
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.hist(historical['close'], bins=30, alpha=0.7, color='blue', edgecolor='black')
        ax5.axvline(historical['close'].mean(), color='red', linestyle='--',
                   linewidth=2, label=f'均值: ${historical["close"].mean():.2f}')
        ax5.set_title('历史价格分布', fontsize=12, fontweight='bold')
        ax5.set_xlabel('价格 (USD/BBL)')
        ax5.set_ylabel('频数')
        ax5.legend(fontsize=9)
        ax5.grid(True, alpha=0.3, axis='y')

        plt.savefig(f'/home/runheyang/DataDisk/temp/{symbol}_analysis.png', dpi=300, bbox_inches='tight')
        print(f"✓ 图表已保存: {symbol}_analysis.png")
        plt.show()

    def generate_report(self, forecast_result):
        """生成预测报告"""
        if forecast_result is None:
            return

        symbol = forecast_result['symbol']
        forecast_df = forecast_result['forecast']
        historical = forecast_result['historical']

        print(f"\n{'='*70}")
        print(f"{symbol}原油期货预测报告")
        print(f"{'='*70}")

        print(f"\n【历史数据统计】")
        print(f"  最新价格: ${historical['close'].iloc[-1]:.2f}")
        print(f"  平均价格: ${historical['close'].mean():.2f}")
        print(f"  标准差: ${historical['close'].std():.2f}")
        print(f"  最高价: ${historical['close'].max():.2f}")
        print(f"  最低价: ${historical['close'].min():.2f}")

        print(f"\n【未来30天预测】")
        print(f"  预测均值: ${forecast_df['forecast'].mean():.2f}")
        print(f"  预测最高: ${forecast_df['forecast'].max():.2f}")
        print(f"  预测最低: ${forecast_df['forecast'].min():.2f}")

        print(f"\n【前10天详细预测】")
        for i in range(min(10, len(forecast_df))):
            print(f"  第{i+1}天: ${forecast_df['forecast'].iloc[i]:.2f} "
                  f"(置信区间: ${forecast_df.iloc[i, 0]:.2f} - ${forecast_df.iloc[i, 1]:.2f})")


    def generate_pdf(self, forecast_results):
        """生成PDF报告"""
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.units import cm
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        import matplotlib.font_manager as fm

        # 注册中文字体
        pdfmetrics.registerFont(TTFont('CJK', '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf'))

        output_path = '/home/runheyang/DataDisk/temp/期货分析报告.pdf'
        doc = SimpleDocTemplate(output_path, pagesize=A4,
                                leftMargin=2*cm, rightMargin=2*cm,
                                topMargin=2*cm, bottomMargin=2*cm)

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle('title', fontName='CJK', fontSize=18, alignment=1, spaceAfter=6)
        h2_style = ParagraphStyle('h2', fontName='CJK', fontSize=13, spaceAfter=4, spaceBefore=12, textColor=colors.HexColor('#1a5276'))
        normal_style = ParagraphStyle('normal', fontName='CJK', fontSize=10, spaceAfter=3)
        small_style = ParagraphStyle('small', fontName='CJK', fontSize=9, textColor=colors.gray)

        contracts_info = {
            'WTI':   {'name': 'WTI原油', 'exchange': 'NYMEX', 'multiplier': 100, 'unit': 'BBL'},
            'Brent': {'name': 'Brent原油', 'exchange': 'ICE',   'multiplier': 100, 'unit': 'BBL'},
        }

        story = []

        # 标题
        story.append(Paragraph('外盘原油期货走势分析报告', title_style))
        story.append(Paragraph(f'生成时间：{datetime.now().strftime("%Y-%m-%d %H:%M")}', small_style))
        story.append(Spacer(1, 0.4*cm))

        for result in forecast_results:
            symbol = result['symbol']
            historical = result['historical']
            forecast_df = result['forecast']
            info = contracts_info[symbol]
            latest = historical.iloc[-1]
            prev = historical.iloc[-2]
            change = latest['close'] - prev['close']
            change_pct = change / prev['close'] * 100

            # 品种标题
            story.append(Paragraph(f'{info["name"]} ({symbol})', h2_style))

            # 主力合约信息表格
            vol_change = latest['volume'] - prev['volume']
            pos_change = latest['position'] - prev['position'] if 'position' in historical.columns else 0

            table_data = [
                ['项目', '数值', '项目', '数值'],
                ['交易所', info['exchange'], '合约乘数', f'{info["multiplier"]} USD/点'],
                ['最新价', f'${latest["close"]:.2f}', '涨跌幅', f'{change:+.2f} ({change_pct:+.2f}%)'],
                ['开盘价', f'${latest["open"]:.2f}', '最高价', f'${latest["high"]:.2f}'],
                ['最低价', f'${latest["low"]:.2f}', '成交量', f'{latest["volume"]:.0f}'],
                ['增减仓', f'{pos_change:+.0f}' if pos_change != 0 else '暂无数据', '计价单位', info['unit']],
            ]

            t = Table(table_data, colWidths=[3.5*cm, 4*cm, 3.5*cm, 4*cm])
            t.setStyle(TableStyle([
                ('FONTNAME', (0,0), (-1,-1), 'CJK'),
                ('FONTSIZE', (0,0), (-1,-1), 9),
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1a5276')),
                ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#eaf4fb')),
                ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#eaf4fb'), colors.white]),
                ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#aed6f1')),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                ('FONTNAME', (0,0), (-1,0), 'CJK'),
                ('FONTSIZE', (0,0), (-1,0), 10),
                ('TOPPADDING', (0,0), (-1,-1), 5),
                ('BOTTOMPADDING', (0,0), (-1,-1), 5),
            ]))
            story.append(t)
            story.append(Spacer(1, 0.3*cm))

            # 期限结构
            story.append(Paragraph('期限结构', h2_style))
            term_data = [
                ['数据周期', '平均价格', '价格区间', '历史波动率'],
                [
                    f'{historical["date"].min().strftime("%Y-%m-%d")} 至 {historical["date"].max().strftime("%Y-%m-%d")}',
                    f'${historical["close"].mean():.2f}',
                    f'${historical["close"].min():.2f} - ${historical["close"].max():.2f}',
                    f'{historical["close"].std():.2f}',
                ]
            ]
            t2 = Table(term_data, colWidths=[4.5*cm, 3.5*cm, 4.5*cm, 2.5*cm])
            t2.setStyle(TableStyle([
                ('FONTNAME', (0,0), (-1,-1), 'CJK'),
                ('FONTSIZE', (0,0), (-1,-1), 9),
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1a5276')),
                ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#eaf4fb')),
                ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#aed6f1')),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                ('TOPPADDING', (0,0), (-1,-1), 5),
                ('BOTTOMPADDING', (0,0), (-1,-1), 5),
            ]))
            story.append(t2)
            story.append(Spacer(1, 0.3*cm))

            # ARIMA预测摘要
            story.append(Paragraph('未来30天价格预测（ARIMA模型）', h2_style))
            pred_data = [
                ['预测均值', '预测最高', '预测最低', '第1天预测', '第5天预测', '第10天预测'],
                [
                    f'${forecast_df["forecast"].mean():.2f}',
                    f'${forecast_df["forecast"].max():.2f}',
                    f'${forecast_df["forecast"].min():.2f}',
                    f'${forecast_df["forecast"].iloc[0]:.2f}',
                    f'${forecast_df["forecast"].iloc[4]:.2f}',
                    f'${forecast_df["forecast"].iloc[9]:.2f}',
                ]
            ]
            t3 = Table(pred_data, colWidths=[2.5*cm]*6)
            t3.setStyle(TableStyle([
                ('FONTNAME', (0,0), (-1,-1), 'CJK'),
                ('FONTSIZE', (0,0), (-1,-1), 9),
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1a5276')),
                ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#eaf4fb')),
                ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#aed6f1')),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                ('TOPPADDING', (0,0), (-1,-1), 5),
                ('BOTTOMPADDING', (0,0), (-1,-1), 5),
            ]))
            story.append(t3)
            story.append(Spacer(1, 0.3*cm))

            # 插入图表
            story.append(Paragraph('综合分析图表', h2_style))
            img_path = f'/home/runheyang/DataDisk/temp/{symbol}_analysis.png'
            story.append(Image(img_path, width=15*cm, height=11.25*cm))
            story.append(Spacer(1, 0.5*cm))

        doc.build(story)
        print(f"✓ PDF报告已生成: 期货分析报告.pdf")


def main():
    analyzer = FuturesAnalyzer()

    # 1. 获取数据
    if not analyzer.get_sample_data():
        print("数据获取失败，程序退出")
        return

    # 2. 计算技术指标
    print("\n正在计算技术指标...")
    for symbol in analyzer.data.keys():
        analyzer.calculate_indicators(symbol)
        print(f"✓ {symbol}技术指标计算完成")

    # 3. 展示合约信息
    analyzer.display_contract_info()

    # 4. 进行预测和分析
    forecast_results = []
    for symbol in analyzer.data.keys():
        forecast_result = analyzer.forecast_arima(symbol, periods=30)
        if forecast_result:
            analyzer.plot_analysis(forecast_result)
            analyzer.generate_report(forecast_result)
            forecast_results.append(forecast_result)

    # 5. 生成PDF报告
    if forecast_results:
        analyzer.generate_pdf(forecast_results)

    print("\n" + "="*70)
    print("✓ 分析完成！")
    print("="*70)


if __name__ == "__main__":
    main()
