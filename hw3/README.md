# HW3 原油期货期限结构分析

## 作业要求

本次 `hw3` 只完成以下任务：

1. 选取一类期货合约，这里选择上海国际能源交易中心原油期货 `SC`。
2. 获取不同月份合约的最近一个交易日收盘价。
3. 绘制期限结构图。
4. 判断当前市场属于升水结构还是贴水结构。
5. 结合期限结构讨论可能的交易策略。

本目录不包含 WTI、Brent 历史走势分析，也不包含 ARIMA 预测内容。

## 目录说明

```text
term_structure_analysis.py   获取原油各月份合约收盘价并绘制期限结构图
build_report.py              生成 PDF 报告
crude_term_structure.csv     原油不同月份合约收盘价数据表
crude_term_structure.png     原油期限结构图
hw3_report.pdf               本次作业 PDF 报告
README.md                    本说明文件
```

## 数据来源

- 数据接口：`akshare`
- 合约品种：上海国际能源交易中心原油期货 `SC`
- 数据内容：不同月份合约最近一个交易日的真实收盘价

说明：
- 这里使用的是实际读取的期货行情数据，不是随机生成的数据。
- 程序会先获取原油品种当前可交易的月份合约列表，再逐个读取对应合约的最近收盘价。

## 运行方法

在项目根目录执行：

```bash
source /Users/runheyang/miniconda3/etc/profile.d/conda.sh
conda activate oil_predict_py312
python /Users/runheyang/Desktop/oil_predict/hw3/term_structure_analysis.py
python /Users/runheyang/Desktop/oil_predict/hw3/build_report.py
```

运行后会在 `hw3` 目录下生成：

```text
crude_term_structure.csv
crude_term_structure.png
hw3_report.pdf
```

## 程序实现思路

1. 使用 `ak.futures_zh_realtime("原油")` 获取当前所有原油月份合约。
2. 过滤出具体月份合约，例如 `SC2604`、`SC2605`。
3. 对每个合约调用 `ak.futures_zh_daily_sina(symbol=...)`。
4. 提取最近一个交易日的收盘价，形成期限结构数据表。
5. 将近月到远月价格按月份顺序绘图。
6. 比较近月与远月价格，判断是升水还是贴水，并给出交易思路。

## 结果解释

- 如果远月价格普遍高于近月价格，则为升水结构 `Contango`。
- 如果近月价格普遍高于远月价格，则为贴水结构 `Backwardation`。

交易思路示例：

- 升水结构下，可以关注做多近月、做空远月的跨期价差收敛思路。
- 贴水结构下，可以关注做空近月、做多远月的跨期价差回归思路。
- 实际交易还需要考虑库存、地缘政治、流动性和保证金风险。
