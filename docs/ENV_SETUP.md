# 环境配置教程（已按你的电脑定制 · Windows 版）

> 适用对象：`amazon-beauty-recsys` 推荐系统项目。
> 本教程**已针对你的电脑现状定制**，你不需要安装 Java、不需要配置任何系统环境变量，
> 只需要用 Anaconda 建一个虚拟环境装依赖即可。
> 照着从上往下复制粘贴命令即可，全程约 20~40 分钟（含下载等待）。

---

## 0. 你的电脑现状盘点（我根据你的检查结果整理）

| 项目 | 你的情况 | 你需要做什么 |
|---|---|---|
| Java | ✅ 已装好：openjdk 11.0.29（可用） | **什么都不用做** |
| Python | ✅ 已装好：3.10.11 | 不用它，虚拟环境里会再装一个独立的 |
| conda（Anaconda） | ⬜ 待你确认 | 看第 1 步 |
| 项目数据 | ✅ `D:\Amazon_beauty_project\` 已有两个 JSON | 路径已配好，不用改 |
| VS Code | ⬜ 待你确认 | 看第 6 步 |

> **为什么 Java 不用管？** 你的 `java -version` 能打出 `openjdk 11.0.29`，说明 Java
> 已经在系统里且能被找到。PySpark 会自动发现它，**不需要** JAVA_HOME 环境变量，
> 也不需要你做任何 Java 相关操作。
>
> ⚠️ 唯一要注意的版本匹配：**你的 Java 是 11 → 项目必须用 pyspark 3.5.x**。
> `requirements.txt` 已固定 `pyspark==3.5.1`，装依赖时不要升级成 4.x（4.x 要 Java 17+）。

---

## 1. 检查 Anaconda（conda）装没装

按 `Win + R` → 输入 `cmd` 回车 → 输入：

```cmd
conda --version
```

- **能看到版本号**（如 `conda 24.x.x`）→ 你已装 Anaconda，**直接跳到第 3 步**。
- 提示 `'conda' 不是内部或外部命令` → 没装，继续第 2 步。

---

## 2. 安装 Anaconda（只有第 1 步没通过才做）

1. 打开清华镜像（比官网快很多）：
   https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/
2. 找到 `Anaconda3-2024.02-1-Windows-x86_64.exe`（或其他 2023/2024 版本）下载；
3. 双击安装，走到「Choose Install Location」这步，**不要用默认路径**，
   点 `Browse...` 换成一个**纯英文路径**，推荐 `D:\Anaconda3`：
   - ⚠️ 你的 Windows 用户名是中文（倞熙），默认路径 `C:\Users\倞熙\anaconda3`
     含中文，Anaconda 会拒绝安装（弹窗报错：Destination Folder cannot contain
     non-ascii characters），并提示部分 Python 库与中文路径不兼容——你安装时遇到的
     就是这个弹窗；
   - 路径里**不要有中文、空格、特殊符号**，`D:\Anaconda3` 这种最稳；
   - 装到 D 盘还有个好处：C 盘省空间，且和你的数据 `D:\Amazon_beauty_project` 同盘；
4. 继续 Next，最后一步勾选 **"Add Anaconda3 to my PATH"**（不加 PATH，后面 cmd 里用不了 conda）。
5. **重新打开** cmd（新装的环境变量要重开窗口才生效），验证：

```cmd
conda --version
```

看到版本号即成功，继续第 3 步。

---

## 3. 创建项目专用虚拟环境 recsys

在 cmd 里逐条执行（第 2 条成功后会看到命令行前面多出 `(recsys)`）：

```cmd
conda create -n recsys python=3.10 -y
conda activate recsys
```

验证一下环境里的 Python 版本：

```cmd
python --version
```

看到 `Python 3.10.x` 即成功。

> `recsys` 是环境名，可随意。**以后每次跑项目前都要先 `conda activate recsys`**。

---

## 4. 安装项目依赖（国内用清华镜像）

### 4.1 先让 pip 永久走清华源（一次配置，以后都生效）

```cmd
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

### 4.2 进入项目目录并安装依赖

```cmd
cd /d D:\amazon-beauty-recsys
pip install -r requirements.txt
```

> `requirements.txt` 里已固定 `pyspark==3.5.1`（和你的 Java 11 匹配）、
> `lightgbm`、`pandas`、`matplotlib` 等，一条命令全装好。

### 4.3 验证依赖装好了

```cmd
python -c "import pyspark, pandas, numpy, lightgbm, sklearn, matplotlib, yaml; print('ALL OK')"
```

输出 `ALL OK` 即成功。如果这里报错，把报错截图发我。

---

## 5. Windows 读本地文件的说明（重要但无需操作）

PySpark 在 Windows 上偶尔会报 `HADOOP_HOME` / `winutils.exe` 错误。本工程已在
`src/recsys/spark_utils.py` 内置规避方案，**正常情况下你什么都不用装**。
万一真的遇到，告诉我，我再给你备用方案。

---

## 6. 安装 VS Code 并打开项目

### 6.1 安装 VS Code
官网下载：https://code.visualstudio.com/ （下一步安装即可，装完打开）

### 6.2 安装 Python 扩展
打开 VS Code → 左侧「扩展」图标（或按 `Ctrl+Shift+X`）→ 搜索 **Python**（微软官方那个）
→ 点安装。Pylance 会自动跟着装。

### 6.3 打开项目并选择解释器
1. `文件 → 打开文件夹…` → 选中 `amazon-beauty-recsys` 文件夹；
2. 打开任意一个 `.py` 文件（比如 `main.py`），右下角会出现 Python 版本号 → 点它 →
   选 **`recsys`**（第 3 步建的环境，名字旁边会标 conda）；
3. 打开终端：`Ctrl + Shift + ` `（键盘 Tab 上面那个键）。终端顶部应显示 `(recsys)`。

> 如果终端没显示 `(recsys)`，手动输入 `conda activate recsys` 再回车。

---

## 7. 数据路径确认（你的数据位置已匹配，一般不用改）

用 VS Code 打开 `config/config.yaml`，看这两行：

```yaml
paths:
  inter_path: "D:/Amazon_beauty_project/Beauty_5.json"
  meta_path: "D:/Amazon_beauty_project/meta_Beauty.json"
```

你的数据正好在 `D:\Amazon_beauty_project\` 下，**这一节不用做任何修改**。
（如果哪天数据挪了位置，把这里的路径改成新的即可，注意斜杠用 `/`。）

---

## 8. 从零到跑通：验证步骤（按顺序执行）

在 VS Code 终端（确保顶部有 `(recsys)`）逐条执行：

```cmd
:: ① 跑单元测试（约 15 秒，验证代码本身没问题）
pytest tests/ -v

:: ② 只跑数据清洗（验证 Spark + Java + 数据全链路）
python main.py --steps clean
```

**先跑 ② 看结果**：日志出现 `关联后最终总行数` 就说明地基通了。
② 通过后再跑全流程（全流程较久，见下方提醒）：

```cmd
:: ③ 跑全流程：EDA→清洗→切分→特征→召回→精排→重排
python main.py
```

> ⏱️ 全流程里「精排采样」要做 `用户×商品` 笛卡尔积，Beauty 5-core（约 2 万用户 ×
> 1 万商品）在普通电脑上需要 **10~40 分钟**，属正常现象，耐心等，别关窗口。
> 内存不够的话先看第 9 节再跑。

### 出问题看哪里
日志同时输出到终端和 `outputs\logs\recsys.log`。出问题先看它最后 50 行：

```cmd
powershell -Command "Get-Content outputs/logs/recsys.log -Tail 50"
```

或者直接在 VS Code 里打开这个文件拉到最底部。把最后 50 行发我，我来帮你解。

---

## 9. 低配置电脑调参建议（内存 < 16G 必看）

打开 `config/config.yaml`，把这两处改小：

```yaml
spark:
  driver_memory: "2g"        # 默认 4g → 内存小就改 2g
  shuffle_partitions: 100    # 默认 200 → 改成 100 减少中间文件
```

如果还是内存不足 / 卡死：

1. **先只验证到召回**（跳过最重的精排/重排）：
   ```cmd
   python main.py --steps clean,split,feature,recall
   ```
   这样能验证 80% 的代码，几分钟就能跑完；
2. **关掉 checkpoint**（省磁盘）：命令末尾加 `--no-checkpoint`；
3. 跑之前关闭浏览器多开标签、微信等占内存的程序；
4. 实在跑不动，把随机负采样从「全局笛卡尔积」改成「按用户采样」——
   代码 `src/recsys/rank/sampling.py` 的 `sample_random_neg` 里已留注释说明。

> 推荐最低配置：8G 内存 + 4 核 CPU。5-core 数据（20 万条交互）能跑；
> 全量 Beauty 数据（200 万条）建议 ≥ 16G 内存。

---

## 10. 常见问题 FAQ

**Q1：`ModuleNotFoundError: No module named 'pyspark'`**
→ 没激活环境或没装依赖。先 `conda activate recsys`，再重新 `pip install -r requirements.txt`。

**Q2：`Cannot run program "java"` 或 `JAVA_HOME is not set`**
→ 你的机器已装 Java 11 且可用，**大概率不会遇到**。万一真遇到，不用去配系统环境变量，
   直接在虚拟环境里补装一个 Java（完全隔离，不动系统设置）：

   ```cmd
   conda activate recsys
   conda install -c conda-forge openjdk=11 -y
   ```

   装完**重开终端**再跑。

**Q3：`HADOOP_HOME and hadoop.home.dir are unset`**
→ 工程已内置规避方案，先什么都不做直接重跑；还不行把报错发我。

**Q4：`Failed to create local dir` 或磁盘满**
→ 中间产物默认写 `outputs\`。磁盘紧张就加 `--no-checkpoint` 跑，
   或把 `config.yaml` 里 `paths.checkpoints` 改到别的盘。

**Q5：`OutOfMemoryError: Java heap space`**
→ 内存不足。按第 9 节调低 `driver_memory`，并关掉占内存的程序。

**Q6：控制台中文乱码**
→ 在终端先输入 `chcp 65001` 回车再跑命令。日志文件本身是 UTF-8，不受影响。

**Q7：`pyarrow` 报错 / `ArrowTypeError`**
→ pandas 与 pyspark 的 `toPandas` 需要 pyarrow，执行：
   `pip install pyarrow==14.0.1`

**Q8：改完 config 后跑出来的结果没变**
→ checkpoint 复用了旧产物。删掉 `outputs\checkpoints\` 整个文件夹再跑
   （改特征/采样/切分逻辑后都要这么做）。

**Q9：`lightgbm` 报 `DLL load failed`**
→ 多半是环境缺 VC 运行库，换 conda 版：`conda install -c conda-forge lightgbm`

**Q10：Spark 日志刷屏（一大堆 INFO）**
→ 正常，工程已把日志级别设为 WARN。想更安静可以把 `spark_utils.py` 里的
   `setLogLevel("WARN")` 改成 `"ERROR"`。

**Q11：报 `UnsupportedClassVersionError ... class file version 61.0 ... up to 55.0`**
→ **这是 pyspark 版本装错了**：`61.0` 是 Java 17 的字节码，`55.0` 是 Java 11。
   你的 Java 是 11，必须用 pyspark 3.5.x。检查一下：
   ```cmd
   conda activate recsys
   pip show pyspark
   ```
   如果版本是 4.x，降回 3.5.1：
   ```cmd
   pip install "pyspark==3.5.1" -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```

**Q12：报 `AnalysisException: ... queries from raw JSON files are disallowed when the
referenced columns only include the internal corrupt record column ...`**
→ 数据文件不是标准 JSON。你的 `meta_Beauty.json` 是 Python 字典格式（键值用单引号，
   `{'asin': 'xxx'}`），Spark 解析不了，全部行进了 `_corrupt_record`。修复：
   1. 先转换格式（项目里已备好脚本）：
      ```cmd
      python tools/fix_meta_format.py
      ```
      它会生成标准 JSON 文件 `D:\Amazon_beauty_project\meta_Beauty_std.json`；
   2. `config/config.yaml` 的 `paths.meta_path` 已指向转换后的文件（不用再改）；
   3. 重新跑 `python main.py --steps clean`。
   （其他文件如果也报这个错，同理：先检查文件头部是不是单引号格式。）

---

## 11. 环境就绪速查表（完成前对照检查）

| 检查项 | 验证命令 | 期望结果 |
|---|---|---|
| 环境激活 | `conda activate recsys && python --version` | `Python 3.10.x` |
| Java（已装，无需操作） | `java -version` | `openjdk 11.x` |
| 依赖 | `python -c "import pyspark, lightgbm; print('OK')"` | `OK` |
| 单元测试 | `pytest tests/ -v` | `11 passed` |
| Spark 读数据 | `python main.py --steps clean` | 日志出现「关联后最终总行数」 |
| 全流程 | `python main.py` | 日志出现「重排结果已保存」 |

完成以上全部 = 环境就绪 ✅

---

## 12. 如何复用到新项目（换数据集 / 开新坑）

> 这套工程 + 这个 `recsys` 环境是一次性搭好的，**以后做别的推荐系统项目可以直接复用**，
> 不用重装 Java、不用重配环境变量、不用再踩一遍这些坑。下面分三种情况说明。

### 12.1 同一个项目换数据集（最常见）

只改一个文件：`config/config.yaml` 的 `paths` 两行。

```yaml
paths:
  inter_path: "D:/你的新数据/交互表.json"   # 用户-商品交互（含 user/item/rating/timestamp）
  meta_path: "D:/你的新数据/商品表.json"     # 商品元信息（含 item_id/title/brand/categories 等）
```

然后**务必删掉旧 checkpoint**（否则会复用上个数据集的中间结果）：

```cmd
conda activate recsys
cd /d D:\amazon-beauty-recsys
rmdir /s /q outputs\checkpoints
python main.py
```

两个常见的新数据坑（本项目的旧版数据都踩过）：

1. **商品表打不开 / 报 `_corrupt_record`**：说明你的 JSON 不是标准 JSON（比如是
   Python 字典打印出来的单引号格式）。先 `Get-Content 文件头几行` 看是不是 `{'asin': ...}`。
   是的话参考 `tools/fix_meta_format.py` 改成标准 JSON 再跑。
2. **字段名对不上**：工程约定交互表必须有 `item_id/user_id/rating/timestamp` 四列、
   商品表必须有 `item_id`。列名不同就先在数据里改名，或告诉我帮你改代码。

### 12.2 全新推荐项目（复制这套骨架）

1. 把 `amazon-beauty-recsys` 整个文件夹复制一份，改名（如 `my-recsys`）；
2. `conda activate recsys`（**同一个虚拟环境直接复用**，别新建，依赖全在里面）；
3. 删掉 `outputs\`、`tests\` 里和旧数据绑定的断言，改 `config.yaml` 的数据路径；
4. 你的新数据若字段不同，只需动 `src\recsys\data\clean.py` 的列名映射，
   流水线、checkpoint、召回→精排→重排的骨架全部原样可用。

### 12.3 三条铁律（复用时千万别违反）

| 铁律 | 原因 |
|---|---|
| **Java 必须 11 + pyspark 必须 3.5.1** | pyspark 4.x 要 Java 17，你的 Java 11 会报 `UnsupportedClassVersionError`，整个环境报废 |
| **不要装/改任何系统环境变量** | 工程已内置 winutils 注入和 driver 回环绑定，改系统设置反而可能弄坏 |
| **改了特征/采样/切分逻辑必须删 checkpoint** | 否则 Spark 懒加载 + checkpoint 复用，跑出来还是旧结果 |

### 12.4 每次开跑的固定动作（30 秒）

```cmd
conda activate recsys        # ① 激活环境（cmd / Anaconda Prompt 里，PowerShell 没配好别用）
cd /d 你的项目目录            # ② 进项目
python main.py               # ③ 跑全流程；断点续跑加 --resume
```

> 换机器 / 换电脑后重装环境，按本教程第 1~8 节重来一遍即可，工程代码本身不用改。
