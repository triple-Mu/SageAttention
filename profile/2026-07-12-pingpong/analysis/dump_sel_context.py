# 从 source 报告按 PC 列出指定 opcode 的 inst_executed 与前后 SASS 上下文，
# 用于定位 pp 版新增 SEL 的来源。用法：python dump_sel_context.py <rep> <opcode前缀> [ctx行数]
import sys
from pathlib import Path

sys.path.insert(0, "/opt/nvidia/nsight-compute/2025.1.1/extras/python")
import ncu_report

rep_path, op_prefix = sys.argv[1], sys.argv[2]
ctx = int(sys.argv[3]) if len(sys.argv) > 3 else 3

rep = ncu_report.load_report(rep_path)
action = rep.range_by_idx(0).action_by_idx(0)

m = action["inst_executed"]
cor = m.correlation_ids()
pcs = [(cor.as_uint64(i), m.as_uint64(i)) for i in range(m.num_instances())]
pcs.sort()
pc_order = [pc for pc, _ in pcs]
execs = dict(pcs)

sass = {pc: action.sass_by_pc(pc) for pc in pc_order}


def opcode(text):
    if not text:
        return ""
    t = text.strip()
    if t.startswith("@"):          # 谓词前缀
        t = t.split(None, 1)[1] if len(t.split(None, 1)) > 1 else t
    return t.split(None, 1)[0].rstrip(";")


hits = [(execs[pc], pc) for pc in pc_order
        if opcode(sass[pc]).startswith(op_prefix) and execs[pc] > 0]
hits.sort(reverse=True)
total = sum(e for e, _ in hits)
print(f"{op_prefix}: {len(hits)} 个 PC，总 inst_executed = {total:,}")

# 按 inst_exec 分桶：看是不是热循环体（同一 exec 值的一簇 PC）
from collections import Counter
buckets = Counter(e for e, _ in hits)
for e, n in buckets.most_common(8):
    print(f"  exec={e:>12,} × {n} 个 PC")

print("\n== 前 12 个热 SEL 的上下文 ==")
for e, pc in hits[:12]:
    idx = pc_order.index(pc)
    print(f"\n-- pc={hex(pc)} exec={e:,} --")
    for j in range(max(0, idx - ctx), min(len(pc_order), idx + ctx + 1)):
        mark = ">>" if j == idx else "  "
        print(f" {mark} {sass[pc_order[j]].strip()}")
