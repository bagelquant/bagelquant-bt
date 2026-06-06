# 架构与设计

`bagelquant-bt` 是 DataFrame 优先的评估包。

```text
prices + weights/factor scores
    |
    v
输入校验
    |
    v
收益、换手、成本、IC、分位数
    |
    v
结果 dataclass
    |
    v
可视化辅助函数
```

## 设计哲学

- 回测包不生成研究信号。
- 公共边界保持为 DataFrame。
- 交易成本必须显式且可复现。
- 返回结构化结果对象，而不是只打印报告。
- 可视化层只消费结果对象。

## 结构

- `inputs`：frame 校验、对齐和数值检查。
- `returns`：资产收益和累计收益工具。
- `costs`：换手和交易成本计算。
- `engine`：权重回测编排和公开 dispatch。
- `factor`：IC、分位数和 top-N 评估。
- `performance`：汇总指标。
- `results`：供下游检查的 dataclass。
- `visualization`：绘图辅助函数。

## 数据边界

输入行是日期，列是资产。权重 frame 表示目标权重。因子 frame 表示截面分数。价格 frame 表示用于计算收益的数值价格。

包不导入 `bagelquant-data` 或 `bagelquant-core`；调用方先把这些包的输出适配成 DataFrame。

