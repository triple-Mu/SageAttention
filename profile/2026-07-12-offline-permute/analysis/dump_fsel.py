import sys
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
idx = [i for i, (pc, ex, s) in enumerate(lines) if "FSEL" in s]
print(f"FSEL PC count: {len(idx)}")
# 打印第一个 FSEL 簇的邻域
if idx:
    lo = max(0, idx[0] - 6)
    hi = min(len(lines), idx[0] + 44)
    out = []
    for pc, ex, s in lines[lo:hi]:
        out.append(f"{pc:#x} x{ex:>9,}  {s}")
    print("\n".join(out))
