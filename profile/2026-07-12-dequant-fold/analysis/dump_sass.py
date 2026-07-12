# 按 PC 序 dump 主循环 SASS（inst_executed 高的 PC 段），看 F2FP/FSEL/PRMT 邻域
import sys
from pathlib import Path
sys.path.insert(0, "/opt/nvidia/nsight-compute/2025.1.1/extras/python")
import ncu_report

rep = ncu_report.load_report(sys.argv[1])
a = rep.range_by_idx(0).action_by_idx(0)
m = a["inst_executed"]
cor = m.correlation_ids()
pcs = []
for i in range(m.num_instances()):
    pcs.append((cor.as_uint64(i), m.as_uint64(i)))
pcs.sort()
# 找 F2FP 所在 PC，打印其前后邻域
lines = []
for pc, ex in pcs:
    try:
        s = a.sass_by_pc(pc) or "?"
    except Exception:
        s = "?"
    lines.append((pc, ex, s.strip()))
idx_f2fp = [i for i, (pc, ex, s) in enumerate(lines) if "F2FP.SATFINITE" in s]
if idx_f2fp:
    lo = max(0, idx_f2fp[0] - 40)
    hi = min(len(lines), idx_f2fp[-1] + 12)
    for pc, ex, s in lines[lo:hi]:
        print(f"{pc:#x} x{ex:>9,}  {s}")
