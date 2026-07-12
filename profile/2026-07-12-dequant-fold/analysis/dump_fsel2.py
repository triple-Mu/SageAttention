import sys
from collections import Counter
sys.path.insert(0, "/opt/nvidia/nsight-compute/2025.1.1/extras/python")
import ncu_report
rep = ncu_report.load_report(sys.argv[1])
a = rep.range_by_idx(0).action_by_idx(0)
m = a["inst_executed"]
cor = m.correlation_ids()
pcs = sorted((cor.as_uint64(i), m.as_uint64(i)) for i in range(m.num_instances()))
lines = []
for pc, ex in pcs:
    try:
        s = (a.sass_by_pc(pc) or "?").strip()
    except Exception:
        s = "?"
    lines.append((pc, ex, s))
# FSEL 按 exec 分类
c = Counter()
for pc, ex, s in lines:
    if "FSEL" in s:
        c[ex] += 1
print("FSEL PCs by exec count:", dict(c))
# 打印 exec=32768 的 FSEL 第一簇邻域
idx = [i for i, (pc, ex, s) in enumerate(lines) if "FSEL" in s and ex == 32768]
if idx:
    lo = max(0, idx[0] - 8)
    print("\n".join(f"{pc:#x} x{ex:>7,}  {s}" for pc, ex, s in lines[lo:lo + 40]))
