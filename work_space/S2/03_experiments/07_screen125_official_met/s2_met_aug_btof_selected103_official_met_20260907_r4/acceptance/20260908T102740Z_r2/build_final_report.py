"""Build read-only-source summaries and verify the retrieved evidence inventory."""
import csv
import hashlib
import itertools
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[2]
ROOTS = {
    "extend": BASE / "s2_met_aug_bef_extend175_20260907_r4",
    "met_r2": BASE / "s2_met_aug_btof_selected103_official_met_20260907_r2",
    "met_r4": BASE / "s2_met_aug_btof_selected103_official_met_20260907_r4",
}
METHODS = ("B", "C40", "D", "E_SNFH", "F")
LABELS = ("et", "rc", "tc", "wt")
META = {"artifact_status": "experimental_unvalidated", "operator_approved": False,
        "formal_gate_status": "not_run_not_passed"}
OUTPUTS = []


def read_json(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def finite_number(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (ValueError, TypeError):
        return None


def write_json(name, value):
    path = HERE / name
    with path.open("x") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    OUTPUTS.append(path)


def write_csv(name, rows):
    path = HERE / name
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("x", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    with path.open(encoding="utf-8-sig", newline="") as f:
        require(len(list(csv.DictReader(f))) == len(rows), "CSV output row count")
    OUTPUTS.append(path)


def write_md(name, lines):
    path = HERE / name
    with path.open("x") as f:
        f.write("\n".join(lines) + "\n")
    OUTPUTS.append(path)


def table(headers, rows):
    return ["| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |"] + [
        "| " + " | ".join(map(str, row)) + " |" for row in rows]


def p_text(value):
    return "0 (bootstrap 未观测反向尾部)" if value == 0 else f"{value:.4f}"


def rank_groups(metric, summaries, direction):
    values = {m: summaries[m][metric] for m in METHODS}
    ordered = sorted(METHODS, key=lambda m: values[m], reverse=direction == "higher_is_better")
    groups = []
    for value, names in itertools.groupby(ordered, key=lambda m: values[m]):
        groups.append(" = ".join(names))
    return " > ".join(groups)


remote = read_json(HERE / "REMOTE_ACCEPTANCE.json")
receipt = read_json(HERE / "LOCAL_RETRIEVAL_VERIFICATION.json")
snapshot_path = HERE.parent / "20260908T102740Z/SCHEDULER_SNAPSHOT.json"
snapshot = read_json(snapshot_path)
paired_path = ROOTS["met_r4"] / "cohort/fixed103/paired_comparison/PAIRED_COMPARISON.json"
paired = read_json(paired_path)
require(remote["status"] == receipt["status"] == paired["status"] == "pass", "Acceptance status")
require(snapshot["sacct"]["rc"] == snapshot["squeue"]["rc"] == 0, "scheduler snapshot")
require(not snapshot["squeue"]["stdout"].strip(), "unexpected active jobs")
job_rows = [line.split("|") for line in snapshot["sacct"]["stdout"].splitlines()]
expected_jobs = {"3652699", "3652700", "3652701", "3652702", "3652705", "3652706", "3652758", "3652759", "3652760"}
require({r[0] for r in job_rows} == expected_jobs, "scheduler job set")
require(all(r[2:4] == ["COMPLETED", "0:0"] for r in job_rows), "scheduler completion")
require(all("gres/gpu=1" in r[6] for r in job_rows if r[0] != "3652760"), "GPU allocation")
for key in ("extend", "met_r4", "met_r2"):
    for name, log in snapshot[key]["logs"].items():
        if key == "met_r2" and ("3652703" in name or "3652704" in name):
            continue
        require(not log["errors"], "current-attempt log error: " + name)

inventory_count = Counter()
json_count = 0
for item in receipt["files"]:
    path = ROOTS[item["root_key"]] / item["relative_path"]
    require(path.stat().st_size == item["size"] and sha(path) == item["sha256"], "Local SHA: " + str(path))
    require(not path.name.endswith((".pth", ".pt", ".nii", ".nii.gz", ".jsonl")), "Excluded heavy artifact")
    if path.suffix == ".json":
        read_json(path)
        json_count += 1
    inventory_count[item["root_key"]] += 1
require(len(receipt["files"]) == remote["light_file_count"] == 178, "light file count")
require(sha(paired_path) == remote["paired"]["json_sha256"], "paired local SHA")
require(paired["metric_count"] == 72 and paired["primary_metric_count"] == 24, "metric contract")
require(len(paired["pairs"]) == 10 and remote["paired"]["numeric_records_reproduced"] == 720, "pair records")

data, summaries, ns = {}, {}, {}
metric_order = None
for method in METHODS:
    root = ROOTS["met_r2"] if method in ("B", "E_SNFH", "F") else ROOTS["met_r4"]
    path = root / f"cohort/fixed103/models/{method}/evaluation/leaderboard_metrics.csv"
    with path.open() as f:
        reader = csv.DictReader(f)
        rows, columns = list(reader), reader.fieldnames
    require(len(rows) == 106 and [r["subject_id"] for r in rows[-3:]] == ["mean", "std", "median"], "official CSV shape")
    if metric_order is None:
        metric_order = columns[1:]
    require(columns[1:] == metric_order, "metric order drift")
    data[method], summaries[method], ns[method] = rows[:103], {}, {}
    for metric in metric_order:
        values = [finite_number(r[metric]) for r in rows[:103]]
        valid = [v for v in values if v is not None]
        aggregate = finite_number(rows[-3][metric])
        require(valid and aggregate is not None and math.isclose(mean(valid), aggregate, rel_tol=0, abs_tol=1e-12), "official mean consistency")
        require(aggregate == remote["official_met"][method]["means"][metric], "remote/local mean")
        summaries[method][metric], ns[method][metric] = aggregate, len(valid)

directions = {r["metric"]: r["direction"] for r in paired["pairs"][0]["metrics"]}
primary = {r["metric"] for r in paired["pairs"][0]["metrics"] if r["primary_endpoint"]}
all_pair_rows = []
for pair, expected in zip(paired["pairs"], itertools.combinations(METHODS, 2)):
    a, b = expected
    require((pair["method_a"], pair["method_b"]) == expected and len(pair["metrics"]) == 72, "pair shape")
    for record, metric in zip(pair["metrics"], metric_order):
        require(record["metric"] == metric, "pair metric coverage")
        require(record["wins_a"] + record["ties"] + record["wins_b"] == record["paired_complete_case_count"], "paired WTL sum")
        require(record["paired_complete_case_count"] + record["excluded_nonfinite_pair_count"] == 103, "paired exclusion count")
        all_pair_rows.append({**record, **META})

mean_rows, ranking_rows = [], []
for metric in metric_order:
    direction = directions[metric]
    best_order = sorted(METHODS, key=lambda m: summaries[m][metric], reverse=direction == "higher_is_better")
    for method in METHODS:
        value = summaries[method][metric]
        better = sum((summaries[m][metric] > value if direction == "higher_is_better" else summaries[m][metric] < value) for m in METHODS)
        mean_rows.append({"method": method, "selected_epoch": remote["official_met"][method]["selected_epoch"],
                          "metric": metric, "official_mean": value, "finite_case_count": ns[method][metric],
                          "excluded_nonfinite_case_count": 103 - ns[method][metric], "rank_with_ties": 1 + better,
                          "pipeline_direction": direction, "primary_endpoint": metric in primary, **META})
    if metric in paired["rankings"]:
        require(list(paired["rankings"][metric]) == best_order, "source ranking reproduction")
    ranking_rows.append({"metric": metric, "pipeline_direction": direction,
                         "ranking_best_to_worst_ties_preserved": rank_groups(metric, summaries, direction),
                         "primary_endpoint": metric in primary, **META})

extension_rows = []
for method, result in remote["extension"].items():
    for candidate in result["candidates"]:
        extension_rows.append({"method": method, "candidate_epoch": candidate["total_epoch"],
                               "ema_fg_dice": candidate["ema_fg_dice"], "selected": candidate["total_epoch"] == result["selected_epoch"],
                               "checkpoint_sha256": candidate["sha256"], "finite": candidate["finite"], **META})

write_csv("SCREEN125_OFFICIAL_MET_MEANS.csv", mean_rows)
write_csv("SCREEN125_METRIC_RANKINGS.csv", ranking_rows)
write_csv("SCREEN125_PAIRED_ALL_720.csv", all_pair_rows)
write_csv("EXTEND175_CHECKPOINT_SELECTION.csv", extension_rows)

frontmatter = ["---", "artifact_status: experimental_unvalidated", "operator_approved: false",
               "formal_gate_status: not_run_not_passed", "---", ""]
appendix = frontmatter + ["# Screen125 全部配对结果", "", "数据源为冻结官方 MET paired JSON。5 方法、10 对、每对 72 指标；此处展示原管线标记的 24 个主指标及 4 个 all-instance FP 指标，全部 720 行见 [CSV](SCREEN125_PAIRED_ALL_720.csv)。", "",
    "正向收益按原管线方向统一：越大越好的指标为 A-B；FP、HD95 等越小越好的指标为 B-A。胜/平/负始终从 A 视角计数；n 是该指标两方法均 finite 的病例数，不自动等于 103。", "",
    "95%CI 为原始百分位 bootstrap CI；p 是原管线未做多重比较校正的双尾 bootstrap 尾比例，不是独立验证的正式优越性检验。原始 p=0 只表示有限次抽样未观测到反向尾部，不能解释为真实概率为零。", ""]
for pair in paired["pairs"]:
    appendix += [f"## {pair['method_a']} 对 {pair['method_b']}", ""]
    records = [r for r in pair["metrics"] if r["primary_endpoint"] or r["metric"].startswith("all_instance_fp_")]
    appendix += table(["指标", "有效 n", "A胜/平/负", "A相对B收益", "95%CI", "未校正 p"], [
        [r["metric"], r["paired_complete_case_count"], f"{r['wins_a']}/{r['ties']}/{r['wins_b']}",
         f"{r['benefit_mean_a_over_b']:+.6f}", f"[{r['benefit_ci95_low']:+.6f}, {r['benefit_ci95_high']:+.6f}]", p_text(r["paired_two_sided_bootstrap_p"])] for r in records]) + [""]
write_md("SCREEN125_PAIRED_PRIMARY_AND_FP.md", appendix)

now = datetime.now(timezone.utc)
report = frontmatter + ["# ECNU S2 Screen125 与 Extend175 最终验收", "", "## 结论", "",
    "本次任务的计算与轻量证据验收通过；这不是正式科学优越性 gate 通过。三路 Extend175 训练和原五路 Screen125 selected-checkpoint 官方 MET 横评均完成，9 个相关作业全部 COMPLETED / 0:0。", "",
    "Screen125 中，E_SNFH125 的 ET/TC/WT 病灶 Dice 均值最高；F125 的 TC 检测 F1 最高；D50 的小病灶 F1 更高但整体 FP 更多，C40-50 的 ET/TC FP 更少但小病灶 F1 较低。不存在所有指标统一第一的路线，不自创综合分数。", "",
    "Extend175 三路均按预注册规则选择 epoch175，内部 EMA 均高于125。这里只完成扩展训练与 checkpoint 验收，未把175模型作为本表的官方 MET 结果，也未据此证明175的官方分割效果提高。", "",
    "## 时间与环境", ""]
report += table(["事件", "时间/状态"], [
    ["本机调度快照 UTC", snapshot["local_snapshot_started_utc"]],
    ["远端调度快照 UTC", snapshot["remote_utc"]],
    ["本机拉回开始 UTC", receipt["local_retrieval_started_utc"]],
    ["远端深度核验开始 UTC", remote["remote_verified_at_utc"]],
    ["远端深度核验结束 UTC", remote["remote_verified_completed_utc"]],
    ["本机拉回完成 UTC", receipt["local_retrieval_completed_utc"]],
    ["本机报告生成 UTC", now.isoformat()],
    ["本机报告生成北京时间", now.astimezone(ZoneInfo("Asia/Shanghai")).isoformat()],
    ["远端主机", snapshot["hostname"]],
    ["/public", "248T 总量；217T 已用；32T 可用；88% 使用率（调度快照时）"],
    ["连接清理", "本次显式启动的分流 VPN 已关闭；无持续 SSH"]])
report += ["", "本机和远端时间来自各自时钟；远端约落后本机10分钟。不可混用两者推断额外训练耗时。此次没有重复提交、续跑、改动远端证据或查询无关作业。", "",
           "## A. Extend175 独立收敛实验", "", "候选固定为累计125/150/175；按内部 ema_fg_dice 最高选择，并列取最早。三路从各自125完整恢复，训练预算累计175，不替换原五路125预算结果。", ""]
report += table(["路线", "EMA125", "EMA150", "EMA175", "选择", "175-125"], [
    [m] + [f"{c['ema_fg_dice']:.6f}" for c in remote["extension"][m]["candidates"]] + [175, f"{remote['extension'][m]['delta_from125']:+.6f}"]
    for m in ("B", "E_SNFH", "F")])
report += ["", "验收9份候选 checkpoint 的 current_epoch、内容SHA、网络/优化器/AMP/日志数值 finite，源125 SHA一致；TRAINING_VALIDATION pass，COMPLETE存在。125副本权限0400，150/175为0600；checkpoint_latest在训练正常结束后由训练器移除，不作为失败。", "",
           "[候选与SHA明细](EXTEND175_CHECKPOINT_SELECTION.csv)。内部 EMA 不是完整103例的官方 MET，不应与下面的病灶 Dice数值直接相减或排名。", "",
           "## B. 原五路 Screen125 官方 MET", "",
           "冻结选择为 B125、C40-50、D50、E_SNFH125、F125；五路训练预算均为125，但从预注册候选中选择的epoch可以不同。B/E/F复用MET r2成功结果，C40/D使用MET恢复r4成功结果。", "",
           "同一冻结103例、真实T2W输入及同一官方MET管线；共515份预测，输入412个通道文件、103份参考标签，病例集合、checkpoint SHA、预测geometry均核验通过。C40/D符合单GPU、sequential_in_process、0预处理/导出worker、无multiprocessing queue合同。", "",
           "以下均值直接取官方CSV mean行。括号为各方法该指标实际finite病例数；所有103例都处理完毕，但无病灶/指标未定义的条目按官方语义保留缺失，不填0。各方法均值可能来自不同有效子集，严格差异请看同病例paired表。", ""]
families = [("病灶 Dice (DSC)", "lesionwise_dsc_mean", "越高越好"),
            ("全部病灶检测 F1", "all_instance_f1", "越高越好"),
            ("大病灶检测 F1", "large_instance_f1", "越高越好"),
            ("小病灶检测 F1", "small_instance_f1", "越高越好"),
            ("病灶 NSD", "lesionwise_nsd_mean", "越高越好"),
            ("病灶 HD95", "lesionwise_hd95_mean", "越低越好；单位沿用官方输出"),
            ("全部病灶 FP/例", "all_instance_fp", "越低越好；分母103"),
            ("全部病灶 FN/例", "all_instance_fn", "越低越好；分母103")]
for title, prefix, description in families:
    report += [f"### {title}", "", description + "；数值格式：均值 (n)。", ""]
    report += table(["路线/选择epoch", "ET", "RC", "TC", "WT"], [
        [f"{m}/{remote['official_met'][m]['selected_epoch']}"] + [f"{summaries[m][prefix+'_'+label]:.4f} ({ns[m][prefix+'_'+label]})" for label in LABELS]
        for m in METHODS]) + [""]
report += ["### 主要指标均值排名", "", "仅为官方均值描述性排序，保留并列；不是共同有效子集上的显著性排序。", ""]
report += table(["指标", "ET", "RC", "TC", "WT"], [
    [title] + [rank_groups(prefix + "_" + label, summaries, directions[prefix + "_" + label]) for label in LABELS]
    for title, prefix in (("病灶Dice", "lesionwise_dsc_mean"), ("全部病灶F1", "all_instance_f1"), ("小病灶F1", "small_instance_f1"), ("FP/例(少者先)", "all_instance_fp"))])
report += ["", "完整72指标的均值、有效n与排名见 [均值表](SCREEN125_OFFICIAL_MET_MEANS.csv) 和 [逐指标排名](SCREEN125_METRIC_RANKINGS.csv)。标准差方向等辅助列忠实保留原管线约定，不把低离散度单独解释成临床收益。", "",
           "## 同病例配对统计", "",
           "5方法、10对、每对72指标，共720组统计；已按原始顺序复算20,000次bootstrap，seed=20260831，数值与源JSON一致。每个指标单独取两方法均finite的病例交集；n不同于总队列103。", "",
           "E_SNFH 对 B、F 的病灶Dice如下，展示全部4标签而非只列优势标签。这里统一转为E视角，收益=E-对照；胜/平/负也是E视角。", ""]
focus_rows = []
for comparator in ("B", "F"):
    pair = next(p for p in paired["pairs"] if {p["method_a"], p["method_b"]} == {"E_SNFH", comparator})
    for label in LABELS:
        r = next(r for r in pair["metrics"] if r["metric"] == "lesionwise_dsc_mean_" + label)
        invert = r["method_a"] != "E_SNFH"
        delta = -r["benefit_mean_a_over_b"] if invert else r["benefit_mean_a_over_b"]
        lo, hi = (-r["benefit_ci95_high"], -r["benefit_ci95_low"]) if invert else (r["benefit_ci95_low"], r["benefit_ci95_high"])
        wins, losses = (r["wins_b"], r["wins_a"]) if invert else (r["wins_a"], r["wins_b"])
        focus_rows.append([comparator, label.upper(), r["paired_complete_case_count"], f"{wins}/{r['ties']}/{losses}", f"{delta:+.6f}", f"[{lo:+.6f}, {hi:+.6f}]", p_text(r["paired_two_sided_bootstrap_p"])])
report += table(["E对照", "标签", "n", "E胜/平/负", "E-对照", "95%CI", "未校正p"], focus_rows)
report += ["", "E对B的WT病灶Dice为 +0.033888，95%CI [0.006868, 0.063250]，原始p=0.0130；但这是重复使用开发验证集、未做多重比较校正的探索性结果。E对F四标签Dice的CI均跨0，不能宣布E已严格胜过F。", "",
    "FP保护不能只看召回：E相比B的ET/TC/WT FP均值分别减少0.203883/0.087379/0.116505个/例，但对应paired CI均跨0。D的小病灶F1更高同时总体FP更高，不能忽略这一代价。未预注册FP非劣界值，因此本次不宣布FP非劣性gate通过。", "",
    "[10对完整主指标及FP配对表](SCREEN125_PAIRED_PRIMARY_AND_FP.md) 包含各标签病灶Dice/F1/HD95/NSD及大、小病灶F1和FP的n、胜平负、CI、p；[全部720行CSV](SCREEN125_PAIRED_ALL_720.csv) 同时保留TP/FN、标准差、effect size等原始统计。", "",
    "## 证据与边界", "",
    f"轻量文件共 {len(receipt['files'])} 份，{receipt['total_bytes']:,} bytes；新拉回{receipt['created']}份、本地已有且SHA一致{receipt['reused_identical']}份。逐文件SHA和JSON可解析性复核完成，未拉checkpoint、NIfTI、events JSONL。", "",
    "源CSV病例名有`.nii.gz`后缀，manifest是裸病例ID；仅在验收比较中去掉该固定后缀，未改写源文件或配对顺序。首次本地附加验收脚本因这一格式差异停止的记录保留在`../20260908T102740Z/REMOTE_ACCEPTANCE.json`，它不是训练或官方评估失败。本目录是修正比较方式后的附加验收结果。", "",
    "所有旧attempt的失败证据、launch audit、reservation保持不变。r2中旧C40/D的失败记录仍属历史失败，最终C40/D仅引用r4；未伪装旧attempt成功。", "",
    "固定103同时用于checkpoint选择与本次报告，只能作exploratory。当前五路不是匹配重训的无增强对照试验，也不是完整FixV3对无增强的正式检验；路线E_SNFH不应与历史其他名为E的基线混淆。G1图像重建、offline补全后分割、online增强后分割应分层解释；本次是增强训练模型在真实T2W输入上的分割评估，没有重跑G1图像质量或locked test。", "",
    "本次不改预注册规则，不用175替换125横评，不以内部EMA或重建质量充当官方分割收益。若要作正式优越性结论，需要预注册主要终点、同预算对照、多随机种子及独立未参与选择的测试集；这不属于本次已完成验收。", "",
    "### 相关作业", ""]
report += table(["Job", "名称", "状态", "退出码", "耗时", "节点"], [r[:6] for r in job_rows])
report += ["", "### 索引", "",
    "- [远端深度验收](REMOTE_ACCEPTANCE.json)",
    "- [本地拉回SHA清单](LOCAL_RETRIEVAL_VERIFICATION.json)",
    "- [调度/日志即时快照](../20260908T102740Z/SCHEDULER_SNAPSHOT.json)",
    "- [最终本地验收](FINAL_ACCEPTANCE_VALIDATION.json)",
    "- [Screen125恢复r4计划](../../SELECTED103_OFFICIAL_MET_RECOVERY_PLAN_20260907_R4.json)",
    "- [原始paired JSON](../../cohort/fixed103/paired_comparison/PAIRED_COMPARISON.json)",
    "", "复核代码与过程记录保留于acceptance目录，用于复现本次只读核验；不进入训练runtime。"]
write_md("FINAL_ACCEPTANCE_20260908.md", report)
write_json("FINAL_ACCEPTANCE_VALIDATION.json", {**META, "status": "pass", "acceptance_scope": "computational_artifact_acceptance_only",
    "local_report_generated_utc": now.isoformat(), "local_report_generated_asia_shanghai": now.astimezone(ZoneInfo("Asia/Shanghai")).isoformat(),
    "remote_verified_at_utc": remote["remote_verified_at_utc"], "remote_verified_completed_utc": remote["remote_verified_completed_utc"],
    "source_snapshot_sha256": sha(snapshot_path), "remote_acceptance_sha256": sha(HERE / "REMOTE_ACCEPTANCE.json"),
    "retrieval_receipt_sha256": sha(HERE / "LOCAL_RETRIEVAL_VERIFICATION.json"),
    "reverified_light_files": len(receipt["files"]), "json_files_parsed": json_count, "files_per_root": dict(inventory_count),
    "completed_jobs": sorted(expected_jobs), "extension_candidates_checked": 9, "extension_selected_epochs": {m: r["selected_epoch"] for m, r in remote["extension"].items()},
    "official_met_prediction_counts": {m: 103 for m in METHODS}, "official_met_csv_case_rows": {m: len(data[m]) for m in METHODS},
    "metric_means_recomputed": len(mean_rows), "paired_record_count": len(all_pair_rows), "paired_bootstrap_reproduced_remotely": 720,
    "bootstrap_replicates": 20000, "bootstrap_seed": 20260831, "extension175_official_met_evaluated": False,
    "same_development_cohort_used_for_selection_and_reporting": True, "formal_superiority_claim_allowed": False,
    "prior_failed_acceptance_preserved": True, "vpn_closed_by_retrieval_driver": True})
write_json("FINAL_DELIVERABLES_SHA256.json", {**META, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "files": [{"path": p.name, "size": p.stat().st_size, "sha256": sha(p)} for p in OUTPUTS]})
print(json.dumps({"status": "pass", "report": str(HERE / "FINAL_ACCEPTANCE_20260908.md"),
                  "light_files": len(receipt["files"]), "means": len(mean_rows), "pairs": len(all_pair_rows), "json_files": json_count}, ensure_ascii=False))
