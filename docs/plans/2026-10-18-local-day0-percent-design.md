# Local day0 percentage clustering implementation plan

用户要求：在项目中复刻 day0 支撑阻力识别算法、使用 merge_percent、515450 图中以不同颜色绘出区域并从最早成员拐点开始、说明 ZigZag 参数。

- [x] 将 day0 Ward 聚类及簇中位数聚合内置到 day0_backend.py，运行不导入克隆仓库；保留作者、许可、固定来源信息。sklearn/scipy 是本地数值依赖。
- [x] 聚类入口使用 close、merge_percent；内部严格按样本收盘均价×merge_percent/100 设置 Ward distance_threshold。同步保存簇成员的最早拐点、确认日期与最低/最高价格。
- [x] 移除 ZigZag 最小间隔过滤，保留 ATR 自适应阈值；水平线默认使用全部已确认历史拐点，不额外删除单点簇。逐日聚类保持因果，版本号更新，旧回测不视为当前结果。
- [x] 绘图使用当前快照的簇，以成员最早拐点为起点，独立颜色显示上下界、中位线及区域；明确快照回溯显示与实际可交易时点不同。API 同步提供区域和起点信息。
- [x] 更新参数、使用说明和回测来源记录，生成并查看515450图。本次不新增或运行测试，不重跑未请求的300标的回测。
