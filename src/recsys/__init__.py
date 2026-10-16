"""Amazon Beauty 离线推荐系统 —— 工程化源码包。

分层结构：
    data/     数据读取与清洗
    eda/      数据探查（L0~L3）
    features/ 特征工程（时序切分 / 物品特征 / 用户 PIT 特征）
    recall/   召回（ItemCF + 热门 + 多路合并）
    rank/     精排（负采样 / PIT 挂载 / LightGBM）
    rerank/   重排（多样性 + 热门惩罚）
    pipeline/ 流水线编排（清洗 / EDA / 训练）
"""
