"""修复 meta_Beauty.json 文件格式（Python repr → 标准 JSON Lines）。

背景：D:\\Amazon_beauty_project\\meta_Beauty.json 实际是 Python dict 的 repr 格式
（键值用单引号，如 {'asin': 'xxx'}），并非标准 JSON（标准 JSON 必须用双引号）。
Spark 读取时每行都会被判为损坏记录（_corrupt_record），导致 clean 阶段报错：
    AnalysisException: ... queries from raw JSON files are disallowed when the
    referenced columns only include the internal corrupt record column ...

本脚本逐行用 ast.literal_eval 解析（它能正确识别单引号/双引号混合的 Python 字面量），
再转成标准 JSON Lines 输出到新文件，一次转换永久生效。

用法（在 recsys 环境里执行）：
    python tools/fix_meta_format.py
"""
import ast
import io
import json
import sys

SRC = r"D:\Amazon_beauty_project\meta_Beauty.json"
DST = r"D:\Amazon_beauty_project\meta_Beauty_std.json"


def main() -> None:
    n_ok = n_err = 0
    bad_samples = []
    with io.open(SRC, "r", encoding="utf-8", errors="replace") as fin, \
         io.open(DST, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = ast.literal_eval(line)
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                n_ok += 1
            except Exception as exc:  # noqa: BLE001 - 单行容错
                n_err += 1
                if len(bad_samples) < 5:
                    bad_samples.append((n_ok + n_err, line[:120], str(exc)))
            if n_ok % 200_000 == 0 and n_ok > 0:
                print(f"已转换 {n_ok:,} 行...")

    print("=" * 60)
    print(f"转换完成：成功 {n_ok:,} 行，失败 {n_err:,} 行")
    for idx, snippet, msg in bad_samples:
        print(f"  第 {idx} 行失败: {snippet!r} -> {msg}")
    if n_err > 0:
        print("注意：失败的若干行已跳过（视为损坏数据，Spark 侧同样会过滤）。")
    print(f"输出文件：{DST}")
    if n_ok == 0:
        print("错误：成功行数为 0，请检查源文件路径。")
        sys.exit(1)


if __name__ == "__main__":
    main()
