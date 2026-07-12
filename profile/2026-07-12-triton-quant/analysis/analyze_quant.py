# 解析 ncu 报告：每个报告含多个量化 kernel action，逐 kernel 提取关键 metric。
# 用法（容器内）：python analyze_quant.py <reports_dir> <analysis_out_dir> <tag1> [tag2 ...]
import json
import sys
from pathlib import Path

sys.path.insert(0, "/opt/nvidia/nsight-compute/2025.1.1/extras/python")
import ncu_report

KEY_METRICS = [
    "gpu__time_duration.sum",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__bytes_read.sum.pct_of_peak_sustained_elapsed",
    "dram__bytes_write.sum.pct_of_peak_sustained_elapsed",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__t_sector_hit_rate.pct",
    "launch__grid_size",
    "launch__block_size",
    "launch__registers_per_thread",
    "launch__waves_per_multiprocessor",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "smsp__issue_active.avg.per_cycle_active",
    "smsp__sass_inst_executed_op_local_ld.sum",
    "smsp__sass_inst_executed_op_local_st.sum",
    "smsp__sass_inst_executed_op_global_ld.sum",
    "smsp__sass_inst_executed_op_global_st.sum",
]


def safe(action, name):
    try:
        return action[name].value()
    except Exception:
        return None


def fmt(v):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:,.3f}"
    return f"{v:,}"


def main():
    reports_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    for tag in sys.argv[3:]:
        rep = ncu_report.load_report(str(reports_dir / f"{tag}.ncu-rep"))
        rng = rep.range_by_idx(0)
        actions = [rng.action_by_idx(i) for i in range(rng.num_actions())]
        names = [f"{i}:{a.name()[:58]}" for i, a in enumerate(actions)]   # 前缀防重名覆盖

        allm = {}
        for a, n in zip(actions, names):
            m = {}
            for mn in sorted(a.metric_names()):
                try:
                    m[mn] = a[mn].value()
                except Exception:
                    pass
            allm[n] = m
        (out_dir / f"metrics_all_{tag}.json").write_text(json.dumps(allm, indent=1, default=str))

        with (out_dir / f"kernels_{tag}.txt").open("w") as f:
            header = f"{'metric':<64}" + "".join(f"{n[:30]:>32}" for n in names)
            f.write(header + "\n" + "=" * len(header) + "\n")
            for mtr in KEY_METRICS:
                vals = [safe(a, mtr) for a in actions]
                if all(v is None for v in vals):
                    continue
                f.write(f"{mtr:<64}" + "".join(f"{fmt(v):>32}" for v in vals) + "\n")
        print(f"[{tag}] kernels: {names}")
        print(f"wrote {out_dir}/kernels_{tag}.txt")


if __name__ == "__main__":
    main()
