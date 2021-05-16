# zhaohangContest
招商银行2021年Fintech数据赛道思路及代码

## 赛题

岗位业务量预测

一、赛题背景

近年来，以A（artificial intelligence）B（big data）C（cloud）为代表的数据智能技术飞速发展。为适应新时期银行科技转型的发展战略，招商银行提出“轻运营”理念，通过对未来业务量进行精准预测，可以合理安排人力，提升关键工作节点的精确化和自动化能力，向着以卓越、高效、低成本为特点的精益运营更进一步。

二、课题研究要求

本次大赛为参赛选手提供了两个数据集（训练数据集和评分数据集），包含日期、节假日信息、时间段、岗位（含2种岗位A、B）、业务类型和业务量数据。希望参赛选手基于训练数据集，通过有效的特征提取，构建业务量预测模型，并将模型应用在评分数据集上，输出未来的业务量预测。

任务1：预测未来31天各岗位每天的业务量总量。

任务2：预测未来31天各岗位每天每半小时粒度的业务总量。

三、评价标准

1.评价公式：所有岗位MAPE的均值


2.评价规则：
A、B榜得分计算方式相同，具体如下，其中ScoreT1为任务1的最优MAPE值，ScoreT2为任务2的最优MAPE值。

最终得分以B榜得分为准。最终得分越小，成绩排名越前。

3.评价范围：
任务一：A岗位统计每天的业务量预测MAPE误差，B岗位统计工作日（WN+WS）每天的业务量预测MAPE误差。
任务二：A岗位统计每天8:30-18:30（共20个时间段）半小时粒度业务量预测的MAPE误差；B岗位统计工作日（WN+WS）8:30-18:30（共20个时间段）半小时粒度业务量预测的MAPE误差。

## 思路
整体思路借鉴[https://github.com/DLLXW/data-science-competition/tree/main/else/%E6%8B%9B%E5%95%86%E9%93%B6%E8%A1%8C2021FinTech](https://github.com/DLLXW/data-science-competition/tree/main/else/%E6%8B%9B%E5%95%86%E9%93%B6%E8%A1%8C2021FinTech)，采用了LightGBM库中的GBDT作为baseline，在此基础上融合了以下元素

1. 根据各个业务种类进行细粒度的模型预测，利用细粒度的预测结果最终求取每日业务量总量以及每小时总量

2. 对于部分数据进行筛选，如日期特点为WN和WS但整天业务量为0的数据进行了剔除，保证预测结果准确性

3. 融合规律，如非工作时间的业务量预测值规定为0

## 结果

B榜

任务1：0.13

任务2：0.17

排名：200


