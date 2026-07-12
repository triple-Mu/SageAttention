# 解析 ncu 报告：提取六维度关键 metric + rule engine 建议，产出并排对比与全量 JSON。
# 用法（容器内）：python analyze_sm90.py <reports_dir> <analysis_out_dir> <tag1> [tag2 ...]
#   tag 形如 full_cuda_d128（对应 reports/<tag>.ncu-rep）
import json
import sys
from pathlib import Path

sys.path.insert(0, "/opt/nvidia/nsight-compute/2025.1.1/extras/python")
import ncu_report

# 六维度关键 metric（sm90 经典名）
KEY_METRICS = [
    # 时长 / SOL
    "gpu__time_duration.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__cycles_elapsed.avg",
    "sm__cycles_active.avg",
    # Launch / Occupancy
    "launch__grid_size",
    "launch__block_size",
    "launch__registers_per_thread",
    "launch__shared_mem_per_block_static",
    "launch__shared_mem_per_block_dynamic",
    "launch__shared_mem_per_block_driver",
    "launch__waves_per_multiprocessor",
    "launch__occupancy_per_block_size",
    "launch__occupancy_per_register_count",
    "launch__occupancy_per_shared_mem_size",
    "launch__occupancy_limit_blocks",
    "launch__occupancy_limit_registers",
    "launch__occupancy_limit_shared_mem",
    "launch__occupancy_limit_warps",
    "sm__maximum_warps_per_active_cycle_pct",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sm__warps_active.avg.per_cycle_active",
    # Scheduler
    "smsp__warps_active.avg.per_cycle_active",
    "smsp__warps_eligible.avg.per_cycle_active",
    "smsp__issue_active.avg.per_cycle_active",
    "smsp__inst_issued.avg.per_cycle_active",
    "smsp__issue_inst0.avg.pct_of_peak_sustained_active",
    # Tensor core / pipe 利用率
    "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active",
    "sm__pipe_tensor_op_imma_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "sm__pipe_tensor_op_imma_cycles_active.avg.pct_of_peak_sustained_active",
    "sm__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active",
    "sm__inst_executed_pipe_tensor_op_imma.avg.pct_of_peak_sustained_active",
    "sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active",
    "sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active",
    "sm__inst_executed_pipe_fp16.avg.pct_of_peak_sustained_active",
    "sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_active",
    "sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active",
    "sm__instruction_throughput.avg.pct_of_peak_sustained_active",
    "smsp__inst_executed.sum",
    # Memory
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__bytes_read.sum",
    "dram__bytes_read.sum.pct_of_peak_sustained_elapsed",
    "dram__bytes_write.sum.pct_of_peak_sustained_elapsed",
    "l1tex__throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__t_sector_hit_rate.pct",
    "lts__t_sector_hit_rate.pct",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
    "smsp__sass_inst_executed_op_local_ld.sum",
    "smsp__sass_inst_executed_op_local_st.sum",
    "smsp__sass_inst_executed_op_shared_ld.sum",
    "smsp__sass_inst_executed_op_shared_st.sum",
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


def load_action(reports_dir, tag):
    rep = ncu_report.load_report(str(reports_dir / f"{tag}.ncu-rep"))
    return rep.range_by_idx(0).action_by_idx(0)


def stall_ratios(action):
    """全部 smsp__average_warps_issue_stalled_*_per_issue_active.ratio，按值降序。"""
    out = {}
    for n in action.metric_names():
        if n.startswith("smsp__average_warps_issue_stalled_") and n.endswith("_per_issue_active.ratio"):
            v = safe(action, n)
            if v is not None:
                short = n[len("smsp__average_warps_issue_stalled_"):-len("_per_issue_active.ratio")]
                out[short] = v
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def main():
    reports_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    tags = sys.argv[3:]
    out_dir.mkdir(parents=True, exist_ok=True)

    actions = {}
    for tag in tags:
        a = load_action(reports_dir, tag)
        actions[tag] = a
        # 全量 metric JSON 存档
        allm = {}
        for n in sorted(a.metric_names()):
            try:
                m = a[n]
                allm[n] = {"v": m.value(), "u": m.unit()}
            except Exception as e:
                allm[n] = {"err": str(e)}
        (out_dir / f"metrics_all_{tag}.json").write_text(json.dumps(allm, indent=1, default=str))
        print(f"[{tag}] kernel = {a.name()[:100]}")

    # 并排对比表
    with (out_dir / f"compare_{'_vs_'.join(tags)}.txt").open("w") as f:
        header = f"{'metric':<72}" + "".join(f"{t:>24}" for t in tags)
        f.write(header + "\n" + "=" * len(header) + "\n")
        for m in KEY_METRICS:
            vals = [safe(actions[t], m) for t in tags]
            if all(v is None for v in vals):
                continue
            f.write(f"{m:<72}" + "".join(f"{fmt(v):>24}" for v in vals) + "\n")
        f.write("\n-- warp stall ratios (per issue-active) --\n")
        allkeys = []
        for t in tags:
            for k in stall_ratios(actions[t]):
                if k not in allkeys:
                    allkeys.append(k)
        srs = {t: stall_ratios(actions[t]) for t in tags}
        # 按第一个 tag 的值排序
        allkeys.sort(key=lambda k: -(srs[tags[0]].get(k) or 0))
        for k in allkeys:
            f.write(f"{'stalled_' + k:<72}" + "".join(f"{fmt(srs[t].get(k)):>24}" for t in tags) + "\n")
    print(f"wrote {out_dir}/compare_{'_vs_'.join(tags)}.txt")


if __name__ == "__main__":
    main()
