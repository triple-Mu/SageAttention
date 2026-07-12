# 从 --set source 报告提取 per-PC 数据（无 lineinfo，按 SASS 指令归因）：
#   1) 按 SASS opcode 聚合 inst_executed（定位指令膨胀来源）
#   2) 按 PC 排 pcsamp stall 热点（top N，含 SASS 文本与 stall 分解）
# 用法：python extract_source_stalls.py <reports_dir> <analysis_out_dir> <tag1> [tag2 ...]
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "/opt/nvidia/nsight-compute/2025.1.1/extras/python")
import ncu_report

TOP_N = 40


def per_pc(action, metric):
    """返回 {pc: value}；metric 不存在或无 correlation 时返回 None。"""
    try:
        m = action[metric]
    except Exception:
        return None
    n = m.num_instances()
    if n == 0 or not m.has_correlation_ids():
        return None
    cor = m.correlation_ids()
    out = {}
    for i in range(n):
        pc = cor.as_uint64(i)
        try:
            v = m.as_uint64(i)
        except Exception:
            v = int(m.as_double(i))
        out[pc] = v
    return out


def main():
    reports_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    for tag in sys.argv[3:]:
        rep = ncu_report.load_report(str(reports_dir / f"{tag}.ncu-rep"))
        a = rep.range_by_idx(0).action_by_idx(0)

        _sass_cache = {}

        def sass_at(pc):
            if pc not in _sass_cache:
                try:
                    _sass_cache[pc] = a.sass_by_pc(pc) or "?"
                except Exception:
                    _sass_cache[pc] = "?"
            return _sass_cache[pc]

        # 找出全部带 correlation 的 pcsamp stall 指标
        stall_metrics = [n for n in a.metric_names()
                         if n.startswith("smsp__pcsamp_warps_issue_stalled_") and "not_issued" not in n]
        pc_stalls = {m: per_pc(a, m) for m in stall_metrics}
        pc_stalls = {m: d for m, d in pc_stalls.items() if d}
        inst = per_pc(a, "inst_executed") or {}
        samp_total = per_pc(a, "smsp__pcsamp_sample_count") or {}

        def opcode(pc):
            s = sass_at(pc)
            # SASS 形如 "        FFMA R4, R5, R6, R4 ;"，取第一个 token（剥掉 @P 谓词）
            toks = s.strip().split()
            if toks and toks[0].startswith("@"):
                toks = toks[1:]
            return toks[0].rstrip(";") if toks else "?"

        with (out_dir / f"sass_opcode_{tag}.txt").open("w") as f:
            # 1) inst_executed 按 opcode 聚合
            by_op = defaultdict(int)
            for pc, v in inst.items():
                by_op[opcode(pc)] += v
            total = sum(by_op.values())
            f.write(f"== {tag}: warp inst_executed by SASS opcode (total {total:,}) ==\n")
            for op, v in sorted(by_op.items(), key=lambda kv: -kv[1]):
                if v / max(total, 1) < 0.002:
                    continue
                f.write(f"{op:<20}{v:>16,}  {v / total * 100:6.2f}%\n")

            # 2) stall 样本按 opcode 聚合（总样本 + 各 stall 原因）
            f.write("\n== stall samples by SASS opcode ==\n")
            samp_by_op = defaultdict(int)
            for pc, v in samp_total.items():
                samp_by_op[opcode(pc)] += v
            tot_s = sum(samp_by_op.values())
            f.write(f"total samples: {tot_s:,}\n")
            for op, v in sorted(samp_by_op.items(), key=lambda kv: -kv[1])[:20]:
                f.write(f"{op:<20}{v:>12,}  {v / max(tot_s, 1) * 100:6.2f}%\n")

            # 3) top-N PC stall 热点
            f.write(f"\n== top {TOP_N} PCs by pcsamp samples ==\n")
            for pc, v in sorted(samp_total.items(), key=lambda kv: -kv[1])[:TOP_N]:
                parts = []
                for m, d in pc_stalls.items():
                    sv = d.get(pc, 0)
                    if sv > 0.05 * v:
                        short = m[len("smsp__pcsamp_warps_issue_stalled_"):]
                        parts.append(f"{short}={sv}")
                parts.sort(key=lambda p: -int(p.split("=")[1]))
                f.write(f"pc=0x{pc:08x} samples={v:>7,} inst_exec={inst.get(pc, 0):>12,}\n")
                f.write(f"    {sass_at(pc).strip()}\n")
                f.write(f"    {' '.join(parts[:6])}\n")
        print(f"wrote {out_dir}/sass_opcode_{tag}.txt")


if __name__ == "__main__":
    main()
