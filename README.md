# Amazon Beauty 离线推荐系统（工业工程化版）

把你原来 Jupyter 里的「EDA → 清洗 → 时序切分 → 特征 → 召回 → 精排 → 重排」单脚本流水线，
重构成一个**可配置、可分步运行、可断点续跑、有日志、有测试**的标准 Python 工程。

> 原始 Jupyter 脚本已备份在 `legacy/`，本工程是它们的工程化迁移（逻辑等价 + 修复若干 bug）。

---

## 1. 目录结构

```
amazon-beauty-recsys/
├── main.py                     # 唯一入口：python main.py
├── config/
│   └── config.yaml             # 全部配置：路径 / 超参 / 开关（改配置不用改代码）
├── experiments/                # 实验管理（见 3.5）：实验配置 + 产物 + 对比表
├── src/recsys/                 # 项目代码包
│   ├── config.py               # yaml → 类型化配置对象（启动即校验）+ 实验配置覆盖合并
│   ├── experiment.py           # 实验管理器：快照 / meta / 运行报告 / 对比表
│   ├── logging_utils.py        # 日志：控制台 + 滚动文件
│   ├── spark_utils.py          # SparkSession 单例工厂
│   ├── io_utils.py             # JSON 读取（统一字段别名）+ checkpoint 读写原语
│   ├── eda/                    # EDA L0~L3（日志统计 + 图表落盘）
│   ├── data/
│   │   ├── clean.py            # 数据清洗
│   │   └── split.py            # 时序切分 80/10/10
│   ├── features/
│   │   ├── item_features.py    # 物品静态特征（仅 train 统计，防泄露）
│   │   ├── user_features.py    # 用户 PIT 动态特征（窗口函数快照）
│   │   ├── context.py          # 年/月上下文特征
│   │   └── feature_pipeline.py # 特征阶段编排
│   ├── recall/
│   │   ├── engine.py           # 热门召回 + ItemCF + 多路合并
│   │   └── evaluate.py         # Recall@K
│   ├── rank/
│   │   ├── sampling.py         # 正样本 + hard/随机负样本构造
│   │   ├── pit.py              # PIT 特征挂载（取 <= 样本时间的最近快照）
│   │   ├── features.py         # 特征转换（复用召回 StringIndexer 防漂移）
│   │   ├── train.py            # LightGBM 训练 + 测试集打分
│   │   └── pipeline.py         # 精排阶段编排
│   ├── evaluate/metrics.py     # 共享评估：Recall@K / NDCG@K / 多样性 / AUC
│   ├── rerank/reranker.py      # 规则重排（多样性 + 热门惩罚）
│   └── pipeline/runner.py      # 编排器 + checkpoint（断点续跑）
├── tests/                      # 单元测试（配置 / 指标 / 重排 / 实验管理）
├── legacy/                     # 你的原始 Jupyter 脚本（只读参考）
└── outputs/                    # 运行时自动生成（模型 / 结果 / 日志 / 图表 / 中间产物）
```

**数据流（一个 stage 一个纯函数，产物显式传递，不再依赖全局变量）：**

```
Beauty_5.json ─┐
               ├─→ [eda]      探查结论（日志+图）
meta_Beauty.json ─┘
               ├─→ [clean]    → inter(宽表) + meta(商品表)      → checkpoint
               ├─→ [split]    → train/val/test（用户过滤）      → checkpoint
               ├─→ [feature]  → item_feat + user_ts_feat(PIT)   → checkpoint
               ├─→ [recall]   → 3 份候选 + StringIndexer 编码器 → checkpoint
               ├─→ [rank]     → LightGBM 模型 + test_result      → checkpoint
               └─→ [rerank]   → reorder_final_result.csv + 指标对比
```

---

## 2. 环境准备（保姆级）

> 📌 **完整保姆级环境搭建教程（含国内镜像、低配调参、FAQ）见 `docs/ENV_SETUP.md`**，
> 从零到跑通约 30~60 分钟。下面只是要点速览。

### 2.1 需要的软件
| 软件 | 版本要求 | 说明 |
|---|---|---|
| Python | **3.10**（recsys 环境） | 本项目用 conda 虚拟环境，见 2.2 |
| Java | **11**（已装好，无需操作） | ⚠️ 与 pyspark 3.5.1 严格匹配，见下方红线 |
| 数据 | `Beauty_5.json` / `meta_Beauty.json` | 路径配置在 `config.yaml` |

> ⚠️ **版本红线**：本项目固定 **Java 11 + PySpark 3.5.1**。pyspark 4.x 需要 Java 17，
> 与你的 Java 11 不兼容（报 `UnsupportedClassVersionError`），**不要升级 pyspark**。
> 本机环境已按 `docs/ENV_SETUP.md` 搭好：conda 环境名 **recsys**，Python 3.10。

### 2.2 激活环境

在项目根目录打开终端（VS Code：`Ctrl+Shift+` ` 打开终端），激活 `recsys` 环境：

```powershell
conda activate recsys
pip install -r requirements.txt   # 只在首次 / 换机器时执行
```

> 本机已经装好，不需要再创建环境。重新建环境的完整步骤见 `docs/ENV_SETUP.md`。

### 2.3 检查 Java

```powershell
java -version
```

你的机器已装 openjdk 11（无需任何操作）。**不要配置 `JAVA_HOME` 环境变量**——
工程在进程内自动发现 Java，配系统环境变量反而可能引入新问题。

### 2.4 Windows 读写本地文件（winutils）

PySpark 在 Windows 上写 parquet/checkpoint 需要 `winutils.exe`（否则报
`HADOOP_HOME and hadoop.home.dir are unset`）。**工程已自带**：
`tools/winutils/bin/` 下有 winutils.exe + hadoop.dll，`spark_utils.py` 启动时
自动探测并注入（含 HADOOP_HOME / PATH / hadoop.home.dir），**全程不碰系统环境变量**。

---

## 3. 运行（VS Code）

1. VS Code 打开项目文件夹：`文件 → 打开文件夹 → 选中 amazon-beauty-recsys`
2. 打开终端（`Ctrl+Shift+` ` ），激活环境（见 2.2）
3. 确认 `config/config.yaml` 里 `paths.inter_path / meta_path` 指向你的数据
4. 执行：

```powershell
# 全流程（EDA → 清洗 → 切分 → 特征 → 召回 → 精排 → 重排）
python main.py

# 只跑 EDA（先看数据结论）
python main.py --steps eda

# 分步跑
python main.py --steps clean,split

# 精排 + 重排，复用之前已落盘的 checkpoint（不用重算特征/召回）
python main.py --steps rank,rerank --resume

# 关闭 checkpoint
python main.py --no-checkpoint
```

运行日志在 `outputs/logs/recsys.log`，进度同时在终端输出。

---

## 3.5 实验管理（改参对比的正确姿势）

> **要解决的问题**：以前改参（比如把 `hard_neg_ratio` 从 3 改成 5）是直接改 `config/config.yaml`，
> 重跑后**旧结果被覆盖、再也回不去**，也没法回答「这次和上次比到底改了哪几项、差了多少」。
> 实验管理把「改配置 → 跑 → 结果被覆盖」变成「写实验配置 → 跑 → 自动留痕 → 可对比可复现」。

### 3.5.1 目录约定（全部在项目根 `experiments/` 下）

```
experiments/
├── exp01.yaml                  # 实验配置：只写要覆盖基线的参数（你手动创建）
├── exp01/                      # 运行后自动生成，本实验一切产物都隔离在这里
│   ├── config_snapshot.yaml    # 合并后的完整配置快照 = 这个实验的「复现钥匙」
│   ├── meta.json               # 运行时间 / 命令行 / Python 版本 / 基线配置 hash
│   ├── run_report.json         # 每阶段耗时 + 数据量（行数/用户数）+ 可序列化指标
│   ├── checkpoints/            # 本实验独立的中间产物（不污染 outputs/ 也互不污染）
│   ├── model/  results/  eda/
└── overview.csv                # 全局对比表：每个实验一行，指标横排可比较
```

> `run_report.json` 里每个 stage 除了耗时，还记录它的**数据量**（如 clean 的交互/用户/商品数、
> split 三份规模、recall 候选行数、rank 的正/负样本构成、rerank 输出行数）。
> 这是你理解「每个 stage 在干什么、数据长什么样」最直接的窗口，也是后续做
> 特征/样本实验时最自然的基准线（样本构成一变，这里一眼能看出来）。

### 3.5.2 用法（三步）

```powershell
# ① 建实验配置：只写要改的参数，其余自动继承 config.yaml
#    新建 experiments/exp01.yaml，例如：
#    rank:
#      hard_neg_ratio: 5

# ② 跑实验（自动合并基线 + 覆盖，产物全进 experiments/exp01/）
python main.py --exp exp01

# 实验内断点续跑（复用本实验已算好的上游，不碰全局 checkpoint）
python main.py --exp exp01 --steps rank,rerank --resume

# ③ 随时查询
python main.py --list-exp     # 有哪些实验
python main.py --compare      # 指标横向对比表（overview.csv）
```

### 3.5.3 覆盖规则（deep_merge）

- **标量 / 列表**：直接整体替换基线同名键（如 `hard_neg_ratio: 5` 替换基线 3）；
- **嵌套 dict**：递归合并，只覆盖你写了的那一层；
- **没写的键**：全部继承基线，**不需要复制整个 config.yaml**。

### 3.5.4 实验隔离三原则（为什么这样设计）

1. **配置隔离**：实验用合并后的快照跑，基线 `config.yaml` 永远不被改动；
2. **产物隔离**：checkpoint / 模型 / 结果 / EDA 图全在 `experiments/<name>/` 下，
   实验之间互不污染，删实验只删一个目录；
3. **可追溯**：每个实验都有「完整配置快照 + git commit + 命令 + 耗时」，随时可复现、可对比。

> **注意**：`meta.json` 的 `git_commit` 目前显示 `unknown`，因为项目还没初始化 git。
> 想启用代码版本追溯：在项目根执行 `git init && git add -A && git commit -m "init"`，
> 之后每个实验都会记录当时的 commit，与代码版本一一对应。
> 另外，实验模式的日志仍统一写在 `outputs/logs/recsys.log`，方便排查。

> **普通模式不受影响**：不带 `--exp` 时行为与以前完全一致（结果写 `outputs/`），
> 实验管理是纯增量能力。

---

## 4. 输出物

| 路径 | 内容 |
|---|---|
| `outputs/model/lgb_rank_model.txt` | LightGBM 精排模型 |
| `outputs/results/reorder_final_result.csv` | 每个用户最终推荐（重排后）|
| `outputs/results/feature_importance.csv` | 特征重要性（gain）|
| `outputs/checkpoints/<stage>/` | 各阶段中间产物（parquet/pkl/json），`--resume` 用 |
| `outputs/eda/plots/*.png` | EDA 图表 |
| `outputs/logs/recsys.log` | 全流程日志 |
| `experiments/<name>/` | 实验产物（配置快照 / meta / 运行报告 / 隔离的 checkpoint 等，见 3.5） |
| `experiments/overview.csv` | 实验对比表（每个实验一行） |

---

## 4.5 产物治理（盘点与清理）

> 产物分两类：**中间产物（checkpoint，可重建）** 与 **最终产物（results / model / logs）**。
> 中间产物占磁盘大头（实测 651 MB / 660 MB），随时可删、删了 `--resume` 会自动重算；
> 最终产物是结果与留痕，**永不自动删**。

### 盘点：我的东西都在哪、占多少

```powershell
python main.py --inspect
```

输出：outputs 总大小、每阶段 checkpoint 大小 / 完整状态（✔ 完整 / ✘ 不完整）/ 修改时间、
results / model / logs 文件清单、experiments 实验记录数。

### 清理：安全删中间产物

```powershell
python main.py --clean all --dry-run            # 查看将删除内容（默认行为，不真删）
python main.py --clean split,feature --yes      # 删除指定阶段 checkpoint
python main.py --clean all --yes                # 清空全部 checkpoint（释放 ~651 MB）
python main.py --clean experiments --yes        # 清理实验中间产物（保留配置快照/指标）
```

**安全规则**（设计保证）：
| 规则 | 说明 |
|---|---|
| 默认 dry-run | 只打印"将删除什么 + 释放多少空间"，加 `--yes` 才真删 |
| 只删中间产物 | checkpoint 目录整体删除；results / model / logs 永不自动删 |
| 实验留痕保留 | `--clean experiments` 删 checkpoints/model/results/eda，保留 `*.yaml`、`config_snapshot.yaml`、`meta.json`、`run_report.json`、`overview.csv` |
| 删了能重建 | checkpoint 由上游 stage 重算，`--resume` 会自动补算缺失阶段 |

### 当前磁盘占用（2026-10 实测）

| 产物 | 大小 | 可清理 |
|---|---|---|
| checkpoints/（7 阶段中间产物） | 651.2 MB | ✔（--clean all）|
| results/（最终推荐 + 特征重要性） | 8.7 MB | ✘ 永不自动删 |
| model/（精排模型） | 0.5 MB | ✘ 永不自动删 |
| eda/（图表） | 0.3 MB | ✘ 永不自动删 |
| logs/（运行日志） | 0.1 MB | ✘ 永不自动删 |
| **outputs/ 合计** | **660.8 MB** | — |

---

## 5. 配置说明（config.yaml 关键项）

| 配置 | 含义 | 建议 |
|---|---|---|
| `paths.inter_path / meta_path` | 原始数据路径 | 改成你的实际路径 |
| `data_clean.dedup_user_item` | (user,item) 重复交互去重 | True |
| `data_clean.keep_missing_meta` | meta 缺失商品交互保留 | True |
| `split.*` | 时序切分比例 | 0.8/0.1/0.1，和必须=1 |
| `recall.*` | 召回超参（TopK/共现下限） | 按实验结果调 |
| `rank.*` | 负采样比例 / LGB 参数 | `hard_neg_ratio`、`num_boost_round` 常调 |
| `rerank.*` | 热门惩罚 / 类目占比 | `hot_penalty_coeff` 越大越压制爆款 |
| `pipeline.seed` | 全局随机种子 | 固定后结果可复现 |

改配置 → 重跑即可，**不需要改代码**。这是与 Jupyter 最大的工程差异之一。

> 💡 **做参数对比实验时**：不要直接改 `config.yaml`（会覆盖基线），
> 请用实验模式（见 3.5）：`experiments/<name>.yaml` 只写要改的参数，
> 跑 `python main.py --exp <name>`，基线保留、结果可对比。

---

## 6. 测试

```powershell
pytest tests/ -v
```

覆盖：配置加载与校验、Recall@K/NDCG@K 数值正确性、重排的类目限流/热门惩罚/兜底逻辑、
实验管理（配置覆盖合并 / 快照落盘 / 对比表列对齐等）、运行报告（数据量采集 / 耗时 / 样本统计）。

### 6.1 两类测试

| 类型 | 文件 | 是否需要 Spark | 说明 |
|---|---|---|---|
| 纯逻辑测试 | test_config / test_metrics / test_reranker / test_experiment / test_run_report | 否 | 秒级跑完，不依赖真实数据与 Spark |
| 数据守卫测试 | test_guards.py | 是（session 级一次，约 20~40s） | 用构造小数据守护 4 条最致命的业务规则，见下 |

### 6.2 数据守卫（P1 测试守卫）守护的规则

> 守卫的意义：这 4 条规则一旦被后续改动悄悄破坏，模型结果会「看起来正常、实际上错」，
> 且很难被发现。守卫用小数据 + 真实业务函数断言，保证这些规则永远成立。

| 守卫 | 规则 | 对应测试 |
|---|---|---|
| G1 clean 无重复对 | `(user,item)` 交互去重（保留最新），非法评分被过滤 | TestCleanGuard（3 个） |
| G2 split 时间严格递增 | train < val < test 时间不重叠；冷用户（只出现在 val/test）被过滤；val/test 用户 ⊆ train 用户 | TestSplitGuard（4 个） |
| G3 PIT 不泄露 | train 窗口只看严格早于自己的历史（排除当前行）；val/test/负样本取 ≤ 样本时间的最近快照，绝不取未来 | TestPitGuard（3 个） |
| G4 类别编码复用 | StringIndexer 只在 train 拟合、val/test 复用；未见过类别进 unknown 桶；`apply_cat_codes` 以 train 的 cat_map 编码 test（未见 → -1） | TestEncodingGuard（2 个） |

> 说明：守卫测试全部使用构造小数据（几十到千行），不触碰真实数据与 checkpoint，
> 结果确定、可复现；需要 Spark 的用例通过 `tests/conftest.py` 的 session 级 fixture
> 只启动一次 Spark（本地 local[1]，用完自动释放）。

---

## 7. 与原 Jupyter 代码的差异与修复清单

### 7.1 结构性差异（工程化）
| 原 Jupyter | 现在 |
|---|---|
| 6 个脚本，靠全局变量（`df_final`/`meta_clean`）串数据 | 7 个 stage 纯函数，产物显式返回传递 |
| 硬编码路径 `D:\Amazon_beauty_project\...` | 全部收敛到 `config.yaml` |
| 每个脚本各自建 SparkSession | 全局单例，一次创建一次销毁 |
| `print` 输出 | 统一日志（时间戳 + 模块名 + 落文件）|
| 一次性从头跑到尾 | checkpoint 落盘 + `--resume` 断点续跑 |
| 无测试 | `tests/` 单元测试 |

### 7.2 修复的 Bug / 隐患（完整清单）

**A. 原代码自带的致命 bug（不修跑不起来）**
1. **`0.0R5` 语法错误**（主流水线 `calc_ranking_metrics` 的 return）→ 修正为 `0.0`。
   原文件**无法通过 Python 解析**（你在 Jupyter 里可能改过一版才跑得动）。
2. **`N_OUTPUT` 未定义 NameError**（重排模块打印语句）→ 统一为 `rerank.top_n_output`。
3. **CSV 编码名带连字符 `utf-8`**（pandas 读不了）→ 改为标准 `utf-8`。

**B. 迁移时修的工程问题**
4. **`display()` 依赖 Jupyter**（EDA_L0）→ 改为日志输出。
5. **废弃的 Spark 配置键** `spark.sql.autoBroadcastJoinThreshold` → 仅保留
   `spark.sql.adaptive.autoBroadcastJoinThreshold`。
6. **重排/精排重复的指标函数** → 合并为 `evaluate/metrics.py` 一份（行为等价，
   重排场景用 `sort=False` 保留「最终排序顺序」语义）。
7. **`DCG` 函数在循环内重复定义** → 提到模块级一次定义。
8. **`val_new_user` 校验实际恒为 0**（inner join 构造保证）→ 保留说明并移除冗余计算。

**C. 本机实跑时修的环境/平台问题**
9. **meta 商品表是 Python repr 格式**（键用单引号，Spark 全读成损坏行）→
   新增 `tools/fix_meta_format.py` 转标准 JSON（产出 `meta_Beauty_std.json`），
   `clean.py` / `io_utils.py` 加 `_corrupt_record` 列存在性判断（标准 JSON 无该列时跳过过滤）。
10. **parquet 落盘报 `HADOOP_HOME unset`**（Windows 写文件必需 winutils）→
    工程自包含方案：winutils.exe + hadoop.dll 放 `tools/winutils/bin/`，
    `spark_utils.py` 启动时自动注入 HADOOP_HOME / PATH / hadoop.home.dir，不碰系统环境变量。
11. **`--resume` 一用就报「缺少上游产物」**（依赖预检查在空 ctx 上必然失败）→
    依赖检查移到逐 stage 执行前，resume 模式按 STAGE_ORDER 预加载已落盘 checkpoint。
12. **EDA L2/L3 聚合遇空集返回 NULL** → `or 0` 兜底，同类聚合加固。
13. **TaskResultLost / Connection refused**（本机在校园网段，driver 绑定外部 IP 被
    DHCP/防火墙干扰）→ `spark.driver.host` + `spark.driver.bindAddress` 强制 127.0.0.1，
    另加 `driver.maxResultSize 4g`、`memory.fraction 0.8`、`storageFraction 0.3`。

**D. 本机实跑时修的算法/数据问题**
14. **随机负采样 OOM**：原实现「用户×全部商品」全笛卡尔积（2.7 亿行只为每用户取 1 负样本）
    → 改为按用户采样：物品池编号 + 每用户随机编号 join 取物 + anti-join 剔除已交互
    （数据量 2.7 亿 → 2.2 万）。
15. **union 列数不匹配（4 vs 5）**：随机负采样辅助列 `item_idx/k` 残留 →
    合并前 `.drop("k","item_idx")`。
16. **PIT 负样本折叠**：`attach_features_pit` 窗口按 `(user_id, pit_time)` 分区，
    负样本 pit_time 统一为用户 last_ts，同用户负样本互相挤掉只剩 1 条 →
    分区加 `item_id`（`features/user_features.py` 同步修复）。
17. **LightGBM 训练卡死（本项目最难的坑）**：pandas category dtype × 高基数列
    （brand 3885 类）× LightGBM 4.3.0 × Windows 的 categorical 训练路径死锁 →
    `align_features` 以 train 类别集统一编码后转 **int32 codes 数值特征**入模
    （`apply_cat_codes` 返回 cat_map 供 test 复用），彻底绕开 categorical 路径。
    修复后 16 万行全特征 5 秒跑完 103 轮早停，val AUC 0.8891。
18. **Part 6 KeyError 'label'**：test_pdf 自带 `label=0` 与正样本 label merge 冲突 →
    merge 前先 drop label 列。
19. **中文路径下 LightGBM 模型读写乱码**（路径含中文时 C++ 层用 GBK 解析成乱码）→
    `io_utils.save/load_lgb_model` 先写 ASCII 临时路径再复制；
    **项目迁移到 D 盘纯英文路径后该绕行已移除，代码恢复直接读写**。
20. **rerank merge 同名列冲突**：test_result 已含全部特征列（含 `item_train_cnt`），
    与物品表 merge 时同名列变 `_x/_y` 导致 KeyError → merge 前只保留
    `user_id/item_id/score/label` 核心列。
21. **假日志**：原 `main()` 只要 ctx 有 test_result 就打印「重排结果已写入」（误导）→
    改为仅当重排结果真正产出才打印，否则提示先跑 rerank。

### 7.3 刻意保留的业务逻辑（请放心，一行都没改）
- 时序切分：`approxQuantile` 分位阈值、val/test 只保留 train 用户；
- PIT 特征：`rowsBetween(unboundedPreceding, -1)` 窗口、`hist_ts` 最近快照匹配；
- 召回：ItemCF 共现 `cooc/sqrt(pop_i*pop_j)` 相似度、热门 Top50、多路合并取最高分；
- 精排：hard 负 3 : 随机负 1、`rating>=4` 为正、`log1p` 处理计数特征、
  StringIndexer 只在召回 fit 一次、LightGBM 参数与早停；
- 重排：`score − hot_penalty×log1p(热度)`、类目占比 0.35、不足补齐。

> **已知保留行为**：`brand / main_category` 等类别特征在 `align_features` 里以
> **train 类别集统一编码后转 int32 codes 数值特征**入模（`apply_cat_codes` 返回
> cat_map 供 test 复用同一编码），这既保留了类别信息，又绕开了 Windows 上
> LightGBM categorical 训练路径的死锁。若后续想在 Linux/新版本上重开 categorical
> 路径，只需把 `apply_cat_codes` 的输出改回 pandas category dtype 并传
> `categorical_feature` 参数即可。

---

## 8. 如何扩展（给你的学习路径）

- **新增一个召回通道**：在 `recall/engine.py` 加一路（如 ItemKNN / Swing / 向量召回），
  在 `run_recall_for_users` 里 `unionByName` 合并，评估自动生效；
- **新增特征**：在 `features/` 下加一个模块，stage 里把它 `join` 进样本表，
  记得只统计 train（PIT 防泄露是红线）；
- **实验管理**：已内置（见 3.5）——`experiments/<name>.yaml` 覆盖基线 →
  `python main.py --exp <name>` 自动留痕 → `--list-exp` / `--compare` 对比；
  后续可学「每个实验再附一段中文结论/备注」或「把实验产物打包成可分享的 zip」；
- **模型升级**：把 `rank/train.py` 的 LightGBM 换成 DeepFM / 双塔，接口不变；
- **上线**：`outputs/model/lgb_rank_model.txt` + 召回词典就是可发布的离线产物，
  线上用同一套特征工程 + 模型 predict 即可。

---

## 9. 常见问题（FAQ）

**Q: 运行报 `ModuleNotFoundError: pyspark`**
没有安装依赖，或没激活虚拟环境。见 2.2。

**Q: Spark 报 `java.io.IOException: Cannot run program "java"`**
Java 未安装或 `JAVA_HOME` 未配置。见 2.3。

**Q: 报本地文件权限 / HADOOP_HOME 相关错误**
确认 `config.yaml` 的 `spark.hadoop_local_fs_fix: true`（默认开启）。

**Q: `--resume` 生效但加载的 checkpoint 是旧逻辑算的**
改过特征/采样逻辑后，请删掉 `outputs/checkpoints/` 重新全量跑一次。

**Q: 数据量大、内存不足**
调低 `spark.driver_memory` 与 `spark.shuffle_partitions`（config.yaml），
或把 `rank.sampling` 的随机负采样改为「按用户采样」（代码里已留注释）。

---

## 10. 依赖数据字段契约（速查）

**inter（清洗后宽表）**：`user_id, item_id, rating(float), timestamp(long),
helpful_useful, helpful_total, reviewText, summary, reviewText_empty,
brand, main_category, price, sales_rank_beauty, also_bought_items, also_viewed_items, meta_missing`

**meta（清洗后商品表）**：`item_id, brand, price, sales_rank_beauty,
also_bought_items, also_viewed_items, main_category`

## 11. 日常重跑操作卡（如何再次运行）

> 
> 所有命令都在 **cmd（命令提示符）** 中执行，不要用 PowerShell。

### 11.1 固定开场（每次必做）

```
conda activate recsys
cd /d D:\amazon-beauty-recsys
```

> 
> 若 `conda activate` 报错，改用：`D:\Anaconda3\Scripts\activate && conda activate recsys`
> 确认终端前缀出现 `(recsys)` 再继续。

### 11.2 按场景选择命令

表格

| 场景 | 命令 | 耗时 |
| --- | --- | --- |
| A. 只想重新生成结果（日常最常用） | `python main.py --steps rerank --resume` | 约 1 分钟 |
| B. 完整重跑全流程 | `rmdir /s /q outputs\checkpoints` 然后 `python main.py` | 1.5~2.5 小时 |
| C. 只重跑某几个阶段（例：重训精排） | `rmdir /s /q outputs\checkpoints\rank outputs\checkpoints\rerank` 然后 `python main.py --steps rank,rerank --resume` | 视阶段而定 |
| D. 中断后续跑 | `python main.py --resume` | 自动跳过已完成 |
| E. 参数对比实验（推荐） | 建 `experiments/exp01.yaml`（只写要改的），然后 `python main.py --exp exp01` | 视阶段而定 |
| F. 查看 / 对比实验 | `python main.py --list-exp` / `python main.py --compare` | 秒回（不启动 Spark） |

**规则**：想重算哪段，就删它和它后面的 checkpoint，再从它开始 `--resume`；上游自动加载，下游自动重算。

### 11.3 跑完验收（三步）

```
powershell -Command "Get-Content outputs\logs\recsys.log -Tail 5 -Encoding UTF8"
```

① 看到 `🎉 流水线执行完成`；

```
dir outputs\checkpoints
```

② 看到 7 个目录（eda /clean/split /feature/recall /rank/rerank）；

```
python -c "import pandas as pd; df = pd.read_csv('outputs/results/reorder_final_result.csv'); print(len(df), df['user_id'].nunique())"
```

③ 行数≈8.8 万、用户数≈4400，即通过。

### 11.4 注意事项

1. **结果文件会被覆盖**：每次重跑都会重写 `outputs\results\reorder_final_result.csv`，想留旧结果先复制。
2. **数据目录勿动**：`D:\Amazon_beauty_project\` 下的两个 JSON 是数据源，重跑只删 checkpoint，永不删数据。
3. **睡眠会暂停程序**：长时间运行时把系统睡眠设为 "从不"，或跑完再合盖。
4. **随机性说明**：全流程重跑时，split/recall 的随机抽样会导致结果与上次有约 0.1% 级别的微小差异，属正常现象。