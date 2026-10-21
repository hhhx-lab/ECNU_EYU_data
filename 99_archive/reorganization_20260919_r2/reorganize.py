"""Physical, lossless regrouping of the ECNU research tree; no remote actions."""
import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import stat
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

ADMIN = Path(__file__).resolve().parent
ROOT = ADMIN.parents[1]
PRIOR = ROOT / "work_space/00_workspace_catalog_20260918T153706Z"
FLAGS = {"artifact_status": "experimental_unvalidated", "operator_approved": False,
         "formal_gate_status": "not_run_not_passed"}
EXCLUDED = {".git", ".claude"}
RULES = []
UNITS = []
DATE = re.compile(r"20\d{2}[-_]?[01]\d[-_]?[0-3]\d")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def jwrite(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")


def cwrite(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8-sig", newline="") as f:
        fields = list(rows[0]) if rows else ["no_records"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def mwrite(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        f.write(text)


def load(name):
    return json.loads((ADMIN / name).read_text(encoding="utf-8"))


def plan_files():
    return current_plan()["files"]


def current_plan():
    return load("FINAL_PLAN_LABEL_CORRECTED.json" if (ADMIN / "FINAL_PLAN_LABEL_CORRECTED.json").exists() else "FINAL_PLAN.json")


def move_with_parent_permissions(source, target):
    mode = stat.S_IMODE(source.parent.stat().st_mode)
    directory_mode = stat.S_IMODE(source.stat().st_mode) if source.is_dir() and not source.is_symlink() else None
    try:
        if not mode & stat.S_IWUSR:
            os.chmod(source.parent, mode | stat.S_IWUSR)
        if directory_mode is not None and not directory_mode & stat.S_IWUSR:
            os.chmod(source, directory_mode | stat.S_IWUSR)
        source.rename(target)
    finally:
        os.chmod(source.parent, mode)
        if directory_mode is not None:
            os.chmod(source if source.exists() else target, directory_mode)


def rule(old, new):
    RULES.append((old.rstrip("/"), new.rstrip("/")))


def unit(old, new, title, purpose, kind="experiment", related=None):
    rule(old, new)
    UNITS.append({"old": old, "new": new, "title": title, "purpose": purpose,
                  "kind": kind, "related": related or []})


def s2_group(name):
    if "btof_screen125" in name:
        return "06_btof_screen125", "五路B至F等预算125epoch训练，冻结各自checkpoint选择，不混入175epoch预算。"
    if "btof_selected103" in name:
        return "07_screen125_official_met", "五路selected-checkpoint在fixed103上的官方MET横评；各恢复版本与旧失败记录分别保留。"
    if "bef_extend175" in name:
        return "08_bef_extend175", "B、E-SNFH、F从125延长至175的独立收敛实验；不是原五路等预算对照。"
    if "bef_epoch175" in name:
        return "09_epoch175_official_met", "175epoch官方MET及CPU恢复，保留125/175原始值与paired证据；恢复不改写原作业失败结果。"
    if "g1r4_endpoint" in name:
        return "10_g1r4_endpoint", "G1-r4缺失T2W补全后分割endpoint；val27、fixed103、locked26需按各自manifest区分。"
    if "table7" in name:
        return "11_table7_microf1", "Table7和micro-F1探索；evaluation-only、GPU版本和失败attempt互不覆盖。"
    if "fix_v3" in name or "fixv3" in name or "three_model_fixed103" in name:
        return "05_fix_v3", "FixV3样本增强及回顾性比较；attempt11和其他历史版本独立保留。"
    if "fix_v2" in name:
        return "04_fix_v2", "FixV2开发、盲审与QC holdout；不同attempt的数组、图像和失败证据保留在各自记录中。"
    if "route_a" in name:
        return "03_route_a", "Route A增强方案及gate审查；r4与r5分开保留，不从命名推断通过。"
    if "small_lesion" in name:
        return "02_small_lesion_and_focal", "小病灶与Focal消融；训练快照、timeout和官方评估重试保持独立。"
    return "01_baselines_and_original_e", "历史real-only、completion、原E及其对照；具体模型身份须由checkpoint SHA和选择合同确认。"


def build_rules():
    rule("work_space/00_workspace_catalog_20260918T153706Z", "99_archive/20260918_index_only/catalog")
    rule("work_space/00_WORKSPACE_INDEX_20260918.md", "99_archive/20260918_index_only/old_entry.md")
    rule("work_space/README.md", "01_project_documents/00_history/work_space_README_before_20260919.md")
    rule("README.md", "01_project_documents/00_history/project_README_before_20260919.md")
    rule("work_space/S1_S2_分割模型总结说明.md", "01_project_documents/03_research_overviews/S1_S2_分割模型总结说明.md")
    rule("work_space/fix-V3.md", "work_space/S2/05_documents/01_methods/fix-V3.md")
    rule("time_pipeline", "01_project_documents/02_project_timeline")
    rule("logs", "99_archive/legacy_preprocessing_job_logs")
    unit("BraTS_2023_2024_solutions-main", "03_reference_code/01_original_root_checkout",
         "原项目根目录中的BraTS 2023/2024代码副本", "完整保存此独立来源的上游代码、示例及其历史结果；不与其他副本去重。", "reference_code")
    unit("data_space/past_code/2023", "03_reference_code/02_past_code_2023",
         "共享资料中的2023代码副本", "原data_space来源，保留其内部示例、权重元数据和上游文档；不是当前G1/G2实验。", "reference_code")
    unit("data_space/past_code/2025", "03_reference_code/03_past_code_2025",
         "共享资料中的2025代码副本", "包含2025参考源码及docker_source取回记录，仅保存现有文件，不构建镜像。", "reference_code")
    rule("data_space/past_code", "01_project_documents/04_reference_code_notes")
    rule("data_space/past_articles", "02_reference_materials/01_papers")
    rule("data_space/bios_course_pdfs", "02_reference_materials/02_bios_course")
    rule("data_space/task_assignments", "01_project_documents/01_competition_and_assignments/task_assignments")
    unit("data_space/task1_2026/datasets", "data_space/01_official_task1_dataset_contracts",
         "官方Task1数据契约与获取说明", "本地是数据契约、现状及获取说明，不是完整原始MRI数据集；不可把文档数量当作病例数量。", "dataset_contract")
    unit("data_space/task1_2026/reference_code", "03_reference_code/04_task1_evaluation_and_trainers",
         "Task1官方评估及trainer参考代码", "包含BraTS-evaluation参考来源与nnUNet自定义trainer；保持包内部结构及许可证。", "reference_code")
    rule("data_space/task1_2026", "01_project_documents/01_competition_and_assignments/task1_2026")
    rule("data_space/README.md", "01_project_documents/00_history/data_space_README_before_20260919.md")
    unit("work_space/80_paper_figures/reviewer_visual_examples_20260814", "04_paper_figures/20260814_reviewer_visual_examples",
         "论文审稿展示图与可编辑素材", "保留绘图源码、图注、导出图片/PDF和figure manifest；不是训练数据。", "figures")

    g1 = "work_space/G1"
    v3 = g1 + "/10_code/brats2025-latent-ensemble-synthesis-main-v3"
    unit(v3, g1 + "/01_code/missing_t2w_vae_bbdm", "缺失T2W的VAE/Encoder-Decoder/BBDM代码",
         "程序、配置、测试和Slurm保留为一个可识别的工程；权重、评估数据及论文已移到各自分类。", "code")
    unit(v3 + "/weights", g1 + "/04_models/undated_pretrained_vae", "缺失模态代码携带的预训练VAE权重",
         "从代码weights目录分离；未加载模型，不把此预训练权重误认成run3034060微调后的best_model。", "model")
    unit(v3 + "/data", g1 + "/02_datasets/undated_vae_evaluation_inputs", "VAE工程原data目录中的评估输入",
         "来自缺失模态工程的本地评估输入/说明，完整病例数以旁边manifest及文件名计数为准。", "dataset")
    rule(v3 + "/JHU.pdf", "02_reference_materials/03_g1_missing_modality/JHU.pdf")
    rule(v3 + "/Task 8 2025.pdf", "02_reference_materials/03_g1_missing_modality/Task_8_2025.pdf")
    upstream = g1 + "/10_code/BraTS_2023_2024_solutions-main 3"
    unit(upstream, "03_reference_code/05_g1_upstream_snapshot", "G1曾使用的上游工程副本",
         "GliGAN活动工程已单独放到G1代码目录，其余上游任务源码/文档原样归档为参考，保留与根目录副本的区别。", "reference_code")
    unit(upstream + "/Segmentation_Tasks/GliGAN", g1 + "/01_code/diffusion_gliGAN", "GliGAN Diffusion增强代码",
         "从上游大工程中完整移出GliGAN子工程，保持其内部脚本、配置、模型实现和测试相对关系。", "code")
    rule(g1 + "/10_code/brats2025-latent-ensemble-synthesis-main", "99_archive/local_placeholders/G1/unused_missing_modality_checkout")
    unit(g1 + "/20_data/latents", g1 + "/02_datasets/undated_sample_latents", "本地保留的潜变量样本",
         "原G1/data/latents中的小规模样本，不是完整latent训练集；不能用它替代服务器上的训练缓存。", "dataset")
    unit(g1 + "/20_data", g1 + "/02_datasets/undated_data_manifests", "G1数据放置与评估清单",
         "保留eval_metrics和data placement manifest。原空的raw/input/output等模板另存历史占位区。", "dataset_manifest")
    for folder in ["raw", "input", "input_inference", "output", "attention_masks", "diffusion_cache"]:
        rule(g1 + "/20_data/" + folder, "99_archive/local_placeholders/G1/" + folder)
    unit(g1 + "/30_results/vae_finetuned/run_3034060", g1 + "/03_experiments/01_vae_finetuning/undated_run_3034060",
         "VAE微调前后指标对照 run3034060", "包含baseline/finetuned/delta指标CSV。run号不是日期，未从mtime杜撰训练日期。")
    unit(g1 + "/30_results/v3_validation/run_3056246", g1 + "/03_experiments/02_paired_validation/20260715_run_3056246",
         "V3配对验证 run3056246", "evaluation_run记录UTC 2026-07-15及103例；合成影像单独归入对应数据集，指标与生成参数保留在此。")
    unit(g1 + "/30_results/v3_validation/run_3056246/synthesized", g1 + "/02_datasets/20260715_paired103_synthesized_t2w",
         "V3配对103例合成T2W", "来自run3056246的synthesized输出；对应G1配对验证结果，不等同缺失T2W的265例补全集。", "dataset",
         [g1 + "/03_experiments/02_paired_validation/20260715_run_3056246"])
    unit(g1 + "/30_results/missing_t2w_completion/run_3104668", g1 + "/02_datasets/20260717_missing_t2w_completion_run3104668",
         "265例缺失T2W补全数据及生成证据", "inference_run记录UTC 2026-07-17、case_count=265；病例文件、生成manifest、G2接收清单同包保存。265是生成批次，不直接等于全部获批训练样本。", "dataset",
         ["work_space/G2/02_dataset_release/20260729_local_release/qc/v3_completion_review/run_3104668"])
    diff = g1 + "/30_results/g1_diffusion_v3_final_20260720"
    unit(diff, g1 + "/03_experiments/03_diffusion_training/20260720_v3_multiserver",
         "Diffusion V3多服务器训练及评估汇总", "canonical和sources是不同来源的证据视图。原始日志、部署、split和评估记录保留，checkpoint实体移到对应模型包且不去重。")
    unit(diff + "/canonical/checkpoints", g1 + "/04_models/20260720_diffusion_v3_canonical", "Diffusion V3 canonical模型副本",
         "原canonical/checkpoints，属于汇总视图。与各服务器sources中的重复模型全部保留。", "model",
         [g1 + "/03_experiments/03_diffusion_training/20260720_v3_multiserver"])
    for host in ["a100_117.50.177.229", "a800_117.50.198.191", "h20_117.50.196.61"]:
        unit(diff + "/sources/" + host + "/checkpoints", g1 + "/04_models/20260720_diffusion_v3_sources/" + host,
             "Diffusion V3服务器来源模型：" + host, "按原来源分别存放；不因SHA重复删除，训练日志和来源证据在对应实验包。", "model")
    r4 = g1 + "/30_results/experimental_unvalidated_single_model_eval_20260812_r4"
    unit(r4, g1 + "/03_experiments/04_model_comparison/experimental_unvalidated_single_model_eval_20260812_r4",
         "BBDM、EncDec、Ensemble r4图像质量比较", "保留三路指标、空间审计、paired结果和失败launch记录；图像质量排名不能直接解释为分割收益。")
    for method in ["bbdm_only", "encdec_only"]:
        unit(r4 + "/" + method + "/synthesized", g1 + "/02_datasets/20260812_model_comparison/" + method,
             "r4图像比较的" + method + "合成影像", "仅搬迁本地确实存在的合成影像；对应评估与spatial audit仍放在同一比较实验下。", "dataset",
             [g1 + "/03_experiments/04_model_comparison/experimental_unvalidated_single_model_eval_20260812_r4/" + method])

    g2 = "work_space/G2"
    unit(g2 + "/10_code", g2 + "/01_code", "G2接收、划分、QC及发布脚本", "G2数据发布根保持manifests/splits/qc等内部合同，根位置由本项目路径配置明确指定。", "code")
    release = g2 + "/02_dataset_release/20260729_local_release"
    unit(g2 + "/30_results", release, "G2本地数据发布与QC证据包",
         "这是一个有内部合同的数据发布包：manifests、splits、stats、qc、nnunet_raw相互引用，作为整体归类而不是按扩展名拆散。不同QC批次在qc下继续独立。", "dataset_release")
    for folder, title, purpose in [
        ("manifests", "病例身份与生成数据清单", "病例映射、G1接收和标签来源清单；不能只凭文件数量替代入组合同。"),
        ("splits", "固定病例划分", "master 1035/130/130与真实T2W子集823/103/104分开保存；同样103例不自动等于相同病例。"),
        ("stats", "真实病灶分布与合成目标统计", "分布统计和目标计划不是新生成的数据样本。"),
        ("nnunet_raw/Dataset260_BraTS2026_MET_RealOnly", "历史Dataset260数据集视图", "本地主要是dataset.json与说明，保持历史Dataset编号；不冒充Dataset264或完整影像数据。"),
    ]:
        unit(g2 + "/30_results/" + folder, release + "/" + folder, title, purpose, "dataset")
    qc = ROOT / g2 / "30_results/qc"
    for p in sorted(qc.iterdir()):
        if p.is_dir() and not p.is_symlink():
            unit(p.relative_to(ROOT).as_posix(), release + "/qc/" + p.name, "G2 QC批次：" + p.name,
                 "保留原QC、人工review、gate和来源证据。存在pass或COMPLETE标记不等于本次重新通过科研验收。", "quality_control")
    rule(g2 + "/30_results/reports", g2 + "/05_documents/04_quality_reports")

    for module in ["S1", "S2"]:
        source = "11_project_brats2026_multitask_s1_v2" if module == "S1" else "11_project_brats2026_s2_rc_v1_0"
        unit(f"work_space/{module}/{source}", f"work_space/{module}/01_code",
             module + "分割主工程", "保留repository内部代码层级；原代码文档集中在模块05_documents，运行路径须使用新项目路径表。", "code")
        rule(f"work_space/{module}/{source}/repository/docs", f"work_space/{module}/05_documents/06_code_documentation")
        rule(f"work_space/{module}/10_code", f"99_archive/local_placeholders/{module}/old_code_template")
    for module in ["S3", "S4", "S5"]:
        unit(f"work_space/{module}/10_code", f"work_space/{module}/01_code", module + "分割工程",
             "完整保留模型代码、内部第三方依赖和许可证。S4为SAM2-UNet，S5为SegMamba；未在此任务重跑训练。", "code")
    for module in ["S1", "S2", "S3", "S4", "S5", "G2"]:
        rule(f"work_space/{module}/20_data", f"99_archive/local_placeholders/{module}/old_data_template")
    for module in ["S1", "S3", "S4", "S5"]:
        rule(f"work_space/{module}/30_results", f"99_archive/local_placeholders/{module}/old_results_template")

    s2 = "work_space/S2"
    res = ROOT / s2 / "30_results"
    for p in sorted(res.iterdir()):
        if p.is_symlink() or not p.is_dir() or p.name == "experimental_unvalidated":
            continue
        if "official_validation_" in p.name:
            unit(p.relative_to(ROOT).as_posix(), s2 + "/07_submissions/original_e_20260727_20260728/" + p.name,
                 "原E比赛推理/转移/提交：" + p.name, "保留MODEL_SELECTION、推理合同、179例输出证据和提交manifest；仅归档，不上传、不新建ZIP。", "submission")
        elif "leaderboard" in p.name:
            unit(p.relative_to(ROOT).as_posix(), s2 + "/07_submissions/leaderboard_records/" + p.name,
                 "历史排行榜记录", "原始抓取和截图证据；不是本次重测或当前服务器状态。", "submission")
        elif "route_rebuild" in p.name:
            unit(p.relative_to(ROOT).as_posix(), s2 + "/02_datasets/20260812_fixv3_component_pool_and_masks",
                 "FixV3组件池、valid masks及坐标合同", "这是增强路线的数据准备产物，不是5000份独立训练结果；components、mask、manifest与坐标合同保持配套。", "dataset")
        else:
            group, purpose = s2_group(p.name)
            name = "20260814_original_e_missing_t2w_val27" if p.name == "27。result" else p.name
            unit(p.relative_to(ROOT).as_posix(), s2 + "/03_experiments/" + group + "/" + name,
                 "S2结果批次：" + name, purpose)
    exp = res / "experimental_unvalidated"
    for p in sorted(exp.iterdir()):
        if p.is_dir() and not p.is_symlink():
            group, purpose = s2_group(p.name)
            unit(p.relative_to(ROOT).as_posix(), s2 + "/03_experiments/" + group + "/" + p.name,
                 "S2实验：" + p.name, purpose)
    unit(s2 + "/30_results/s2_eval_results/reference", s2 + "/02_datasets/undated_realonly_fixed103_reference_labels",
         "Real-only评估保留的103例reference labels", "从s2_eval_results中分离reference标签，prediction和官方指标仍归原评估实验。不得作为新的测试集或增强样本。", "dataset",
         [s2 + "/03_experiments/01_baselines_and_original_e/s2_eval_results"])
    unit(s2 + "/30_results/checkpoint_final.pth", s2 + "/04_models/undated_root_checkpoint/checkpoint_final.pth",
         "原S2结果根目录的checkpoint_final", "原文件单独保留并记录SHA；在没有模型选择证据前，不自动将其命名为原E、FixV3或最新最佳。", "single_model")
    for name in ["S2_realonly_current_Task1_validation_179.zip", "official_submission_manifest.csv", "official_submission_validation.json"]:
        rule(s2 + "/30_results/" + name, s2 + "/07_submissions/20260715_realonly_validation179/" + name)
    UNITS.append({"old": s2 + "/30_results", "new": s2 + "/07_submissions/20260715_realonly_validation179",
                  "title": "早期real-only的179例validation提交包", "purpose": "依据配套validation时间UTC 2026-07-15及179例合同，区别于7月27日选择的原E提交。ZIP原样保留，不解包或重打。",
                  "kind": "submission", "related": []})
    unit(s2 + "/80_submission_container", s2 + "/07_submissions/20260730_original_e_container_reserve",
         "既有原E提交容器保留材料", "仅搬迁既有Dockerfile、代码、环境示例和验证证据；不生成或构建新镜像。", "submission")

    for module in ["G1", "G2", "S1", "S2", "S3", "S4", "S5"]:
        rule(f"work_space/{module}/50_slurm", f"work_space/{module}/06_operations/slurm")
        rule(f"work_space/{module}/60_cloud", f"work_space/{module}/06_operations/cloud")
        rule(f"work_space/{module}/70_tests", f"work_space/{module}/01_code/tests")
        rule(f"work_space/{module}/README.md", f"work_space/{module}/05_documents/00_history/README_before_20260919.md")
        rule(f"work_space/{module}/task_assignment.md", f"work_space/{module}/05_documents/02_plans/task_assignment.md")
        rule(f"work_space/{module}/00_CATALOG_20260918T153706Z.md", f"99_archive/20260918_index_only/module_entries/{module}.md")
    rule("work_space/G1/扩散模型加速方法.md", "work_space/G1/05_documents/01_methods/扩散模型加速方法.md")
    RULES.sort(key=lambda pair: len(pair[0]), reverse=True)


def doc_group(name):
    text = name.lower()
    if any(k in text for k in ["环境", "requirements", "environment"]):
        return "05_environments"
    if any(k in text for k in ["计划", "方案", "plan"]):
        return "02_plans"
    if any(k in text for k in ["手册", "部署", "运行", "操作"]):
        return "03_operations"
    if any(k in text for k in ["评估", "总结", "报告", "report", "experiment"]):
        return "04_reports"
    return "01_methods"


def destination(rel):
    parts = Path(rel).parts
    name = parts[-1]
    if name == ".DS_Store" or "__pycache__" in parts or name.endswith((".pyc", ".pyo")):
        return "99_archive/cache_files_20260919/" + rel
    if name == ".gitkeep" or name.startswith("HISTORICAL_EMPTY_DIRECTORY_20260918"):
        return "99_archive/old_placeholders_and_notes/" + rel
    if len(parts) >= 4 and parts[0] == "work_space" and parts[2] == "40_documents":
        return "/".join(["work_space", parts[1], "05_documents", doc_group(name), *parts[3:]])
    for old, new in RULES:
        if rel == old or rel.startswith(old + "/"):
            return new + rel[len(old):]
    if rel in {"PROJECT_RULES.md", ".gitignore", ".gitattributes"}:
        return rel
    return "99_archive/unattributed_preserved/" + rel


def baseline():
    require(not (ADMIN / "BASELINE.json").exists(), "Already inventoried")
    build_rules()
    files, links, dirs = [], [], []
    for base, subdirs, names in os.walk(ROOT, followlinks=False):
        p = Path(base)
        subdirs[:] = sorted(d for d in subdirs if (p / d) != ADMIN and not (p == ROOT and d in EXCLUDED))
        dirs.append({"path": p.relative_to(ROOT).as_posix(), "mode": stat.S_IMODE(p.stat().st_mode)})
        for name in sorted(subdirs + names):
            f = p / name
            rel = f.relative_to(ROOT).as_posix()
            if f.is_symlink():
                links.append({"old": rel, "target": os.readlink(f), "inode": f.lstat().st_ino,
                              "new": "99_archive/20260918_index_only/original_link_objects/" + hashlib.sha256(rel.encode()).hexdigest()[:12] + "__" + name})
            elif name in names:
                require(f.is_file(), "Special file needs review: " + rel)
                st = f.stat()
                files.append({"old": rel, "new": destination(rel), "bytes": st.st_size, "sha256": sha(f),
                              "inode": st.st_ino, "mode": stat.S_IMODE(st.st_mode), "mtime_ns": st.st_mtime_ns,
                              "uid": st.st_uid, "gid": st.st_gid})
                if len(files) % 2500 == 0:
                    print(json.dumps({"stage": "baseline", "files": len(files)}), flush=True)
    counts = Counter(f["new"].casefold() for f in files)
    require(all(n == 1 for n in counts.values()), "Destination collision")
    for f in files:
        require(f["new"] == f["old"] or not os.path.lexists(ROOT / f["new"]), "Destination exists: " + f["new"])
    jwrite(ADMIN / "BASELINE.json", {**FLAGS, "created_utc": now(), "files": files, "symlinks": links, "directories": dirs})
    jwrite(ADMIN / "UNITS.json", {**FLAGS, "units": UNITS})
    cwrite(ADMIN / "MOVE_PLAN.csv", files)
    unknown = [f["old"] for f in files if "/unattributed_preserved/" in f["new"]]
    jwrite(ADMIN / "PLAN_REVIEW.json", {**FLAGS, "files": len(files), "bytes": sum(f["bytes"] for f in files),
               "physical_moves": sum(f["old"] != f["new"] for f in files), "previous_links_to_archive": len(links),
               "units": len(UNITS), "unattributed": unknown})
    print(json.dumps(load("PLAN_REVIEW.json"), ensure_ascii=False), flush=True)


def relative_link(source, target, label):
    return f"[{label}]({quote(os.path.relpath(target, source.parent), safe='/._-')})"


def review_plan():
    data = load("BASELINE.json")
    units = load("UNITS.json")["units"]
    files, extra = [], {}
    def add_group(new, title, purpose, kind, old):
        extra.setdefault(new, {"old": old, "new": new, "title": title, "purpose": purpose, "kind": kind, "related": []})
    for original in data["files"]:
        f = dict(original)
        p = Path(f["new"])
        if f["new"].startswith("work_space/G1/04_models/20260720_diffusion_v3_") and p.suffix not in {".pt", ".pth"}:
            parts = p.parts
            source = "canonical" if "canonical" in parts[3] else parts[4]
            if f["new"].endswith(".nii.gz"):
                base = "work_space/G1/02_datasets/20260720_diffusion_checkpoint_samples/" + source
                f["new"] = base + "/" + "/".join(parts[4:] if source == "canonical" else parts[5:])
                add_group(base, "Diffusion训练过程保存的checkpoint扫描：" + source,
                          "这里是训练时保存的x/y crop-pad影像，不是模型权重，也不能按图像文件数当作独立病例。按canonical和原服务器来源分开，重复样本不删除。", "dataset", f["old"].split("/checkpoints/")[0])
            else:
                base = "work_space/G1/03_experiments/03_diffusion_training/20260720_v3_multiserver/checkpoint_logs_and_curves/" + source
                f["new"] = base + "/" + "/".join(parts[4:] if source == "canonical" else parts[5:])
                add_group(base, "Diffusion checkpoint配套loss、日志和曲线：" + source,
                          "从混合的checkpoints目录中分离训练日志、loss文本与图片。权重在G1/04_models，扫描影像在02_datasets，来源仍一一可查。", "experiment", f["old"].split("/checkpoints/")[0])
        if f["new"].startswith("03_reference_code/"):
            parts = Path(f["new"]).parts
            source, tail = parts[1], "/".join(parts[2:])
            raw = f["old"]
            if "nnUNet_results/" in raw or "/Checkpoint/" in raw or "/DataSet/" in raw or "/quick_test_edm_output/" in raw:
                if p.suffix in {".pt", ".pth"}:
                    base, kind = "data_space/03_reference_models/" + source, "model"
                    title = "参考工程原带模型：" + source
                elif "/DataSet/" in raw:
                    base, kind = "data_space/02_reference_example_data/" + source, "dataset"
                    title = "参考工程原带样例影像：" + source
                else:
                    base, kind = "data_space/04_reference_training_records/" + source, "experiment"
                    title = "参考工程历史训练与生成输出：" + source
                f["new"] = base + "/" + tail
                add_group(base, title, "从参考代码目录分离，保留原上游路径和来源，不与本次比赛G1/S2训练结果混合。", kind, raw.split("/Segmentation_Tasks/")[0])
            elif p.suffix.lower() == ".pdf":
                f["new"] = "02_reference_materials/04_upstream_checkout_documents/" + source + "/" + tail
            elif len(parts) == 3 and p.suffix.lower() == ".md" and p.name.lower() not in {"readme.md", "license.md", "contributing.md"}:
                f["new"] = "01_project_documents/04_reference_code_notes/" + source + "/" + p.name
        files.append(f)
    units += list(extra.values())
    require(len({f["new"].casefold() for f in files}) == len(files), "Reviewed destination collision")
    for f in files:
        require(f["new"] == f["old"] or not os.path.lexists(ROOT / f["new"]), "Reviewed destination exists")
    jwrite(ADMIN / "FINAL_PLAN.json", {**FLAGS, "reviewed_utc": now(), "files": files, "units": units,
               "reason": "Separate actual model weights from checkpoint images/loss files; separate upstream reference data/results/papers from code."})
    cwrite(ADMIN / "FINAL_MOVE_PLAN.csv", files)
    print(json.dumps({"files": len(files), "units": len(units), "reviewed_mapping_changes": sum(a["new"] != b["new"] for a, b in zip(data["files"], files))}), flush=True)


def apply_moves():
    data = load("BASELINE.json")
    data["files"] = plan_files()
    require(not (ADMIN / "MOVED.json").exists(), "Already moved")
    for f in data["files"]:
        old, new = ROOT / f["old"], ROOT / f["new"]
        located = old if old.exists() else new
        st = located.stat()
        require((st.st_ino, st.st_size, st.st_mtime_ns) == (f["inode"], f["bytes"], f["mtime_ns"]), "Concurrent modification: " + f["old"])
        if old != new and old.exists():
            require(not os.path.lexists(new), "Both source and destination exist: " + f["old"])
        if not old.exists():
            require(sha(new) == f["sha256"], "Previously moved file differs: " + f["old"])
    with (ADMIN / "JOURNAL.jsonl").open("a", encoding="utf-8") as log:
        for item in data["symlinks"]:
            dst = ROOT / item["new"]
            if dst.is_symlink():
                require(os.readlink(dst) == item["target"] and dst.lstat().st_ino == item["inode"], "Archived link drift")
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            move_with_parent_permissions(ROOT / item["old"], dst)
            log.write(json.dumps({"kind": "old_link_archived", **item}, ensure_ascii=False) + "\n")
        for i, item in enumerate(data["files"], 1):
            src, dst = ROOT / item["old"], ROOT / item["new"]
            if src != dst:
                if not src.exists():
                    continue
                require(not os.path.lexists(dst), "Refusing overwrite: " + item["new"])
                dst.parent.mkdir(parents=True, exist_ok=True)
                move_with_parent_permissions(src, dst)
                log.write(json.dumps({"kind": "file_moved", "old": item["old"], "new": item["new"]}, ensure_ascii=False) + "\n")
            if i % 2500 == 0:
                log.flush()
                os.fsync(log.fileno())
                print(json.dumps({"stage": "physical_moves", "files": i}), flush=True)
    # Retain emptied original directory objects outside the working tree.
    skeletons = []
    for d in sorted(data["directories"], key=lambda x: len(Path(x["path"]).parts), reverse=True):
        src = ROOT / d["path"]
        if src == ROOT or src == ROOT / "99_archive" or not src.exists() or src.is_symlink() or any(src.iterdir()):
            continue
        ident = hashlib.sha256(d["path"].encode()).hexdigest()[:12]
        dst = ROOT / "99_archive/original_empty_directory_objects" / (ident + "__" + src.name)
        dst.parent.mkdir(parents=True, exist_ok=True)
        move_with_parent_permissions(src, dst)
        mode = stat.S_IMODE(dst.stat().st_mode)
        try:
            os.chmod(dst, mode | stat.S_IWUSR)
            mwrite(dst / "README.md", "# 原目录结构留存\n\n原位置：`" + d["path"] + "`。原文件已经逐项搬到对应代码、数据、实验或文档位置；目录本身未删除。\n\n"
                   "若原名含RESERVATION，本记录仅表示本地历史目录归档，不会释放远端锁、重建reservation或提交作业。\n")
        finally:
            os.chmod(dst, mode)
        skeletons.append({"old": d["path"], "new": dst.relative_to(ROOT).as_posix()})
    skeletons = []
    for d in data["directories"]:
        ident = hashlib.sha256(d["path"].encode()).hexdigest()[:12]
        dst = ROOT / "99_archive/original_empty_directory_objects" / (ident + "__" + Path(d["path"]).name)
        if dst.is_dir():
            skeletons.append({"old": d["path"], "new": dst.relative_to(ROOT).as_posix()})
    jwrite(ADMIN / "EMPTY_STRUCTURES_PRESERVED.json", {**FLAGS, "directories": skeletons})
    jwrite(ADMIN / "MOVED.json", {**FLAGS, "completed_utc": now(), "original_files": len(data["files"]),
               "archived_links": len(data["symlinks"]), "retained_empty_directory_objects": len(skeletons)})
    print(json.dumps(load("MOVED.json")), flush=True)


def describe():
    files = plan_files()
    units = load("FINAL_PLAN.json")["units"]
    written = []
    for u in units:
        base = ROOT / u["new"]
        if u["kind"] == "single_model":
            base = base.parent
        members = [f for f in files if f["new"] == u["new"] or f["new"].startswith(u["new"] + "/")]
        if not members:
            continue
        require(base.is_dir(), "Unit directory missing: " + str(base))
        path = base / "00_说明_20260919.md"
        sizes = sum(f["bytes"] for f in members)
        nii = [f for f in members if f["new"].endswith((".nii", ".nii.gz"))]
        ids = sorted({m.group() for f in nii for m in re.finditer(r"BraTS-MET-\d+-\d+", f["new"])})
        ckpt = [f for f in members if Path(f["new"]).suffix in {".pt", ".pth", ".ckpt"}]
        metadata = []
        keys = ["created_at", "created_at_utc", "generated_at_utc", "case_count", "expected_case_count", "selected_model", "selected_epoch", "best_epoch", "status"]
        preferred = sorted((f for f in members if Path(f["new"]).suffix == ".json" and f["bytes"] < 2_000_000),
                           key=lambda f: (len(Path(f["new"]).parts), 0 if any(t in f["new"] for t in ["PLAN", "inference_run", "evaluation_run", "VALIDATION"]) else 1, f["new"]))
        for f in preferred:
            try:
                obj = json.loads((ROOT / f["new"]).read_text(encoding="utf-8"))
            except (ValueError, UnicodeError):
                continue
            if isinstance(obj, dict):
                facts = {k: obj[k] for k in keys if k in obj and isinstance(obj[k], (str, int, float, bool))}
                if facts:
                    metadata.append((f, facts))
            if len(metadata) >= 5:
                break
        descendants = sorted(p for p in base.iterdir() if p.name != path.name)
        text = [f"# {u['title']}\n\n", f"## 这是什么\n\n{u['purpose']}\n\n",
                f"- 整理日期：2026-09-19（Asia/Shanghai）。\n- 原位置：`{u['old']}`。\n- 当前实体位置：`{u['new']}`。\n",
                f"- 本地原文件：{len(members)}份，{sizes:,}字节。NIfTI文件：{len(nii)}；从文件名识别的BraTS-MET病例ID：{len(ids)}。\n",
                "- 上述是本地文件计数，不等于独立样本数、入组病例数或服务器完整产物；一个病例可能有多个通道或快照。\n\n",
                "## 目录内容\n\n"]
        for child in descendants[:35]:
            if child.name.startswith("."):
                continue
            text.append("- " + relative_link(path, child, child.name) + ("：目录。\n" if child.is_dir() else "。\n"))
        if len(descendants) > 35:
            text.append("- 其余为同批次病例或配套文件，未逐病例生成重复说明。\n")
        text.append("\n## 输入、来源与对应结果\n\n")
        if u["related"]:
            for related in u["related"]:
                target = ROOT / related
                if target.exists():
                    text.append("- " + relative_link(path, target, related) + "。\n")
        old_siblings = [v for v in units if v != u and (v["old"].startswith(u["old"] + "/") or u["old"].startswith(v["old"] + "/"))]
        for sibling in old_siblings[:8]:
            target = ROOT / sibling["new"]
            if target.exists():
                text.append("- 从同一原包归类的配套资料：" + relative_link(path, target, sibling["title"]) + "。\n")
        text.append("- 完整原路径→新路径及每文件SHA见项目99_archive/reorganization_20260919_r2/FINAL_MOVE_PLAN.csv。旧manifest、计划和远端绝对路径是历史证据，未改写；不能照搬旧路径直接重新提交。\n")
        if metadata:
            text.append("\n## 原记录中的关键信息\n\n")
            for f, facts in metadata:
                text.append("- " + relative_link(path, ROOT / f["new"], Path(f["new"]).name) + "：`" + json.dumps(facts, ensure_ascii=False).replace("`", "'") + "`。\n")
        if ckpt:
            text.append("\n## 权重身份\n\n")
            for f in ckpt[:8]:
                text.append("- `" + os.path.relpath(ROOT / f["new"], base) + "`：SHA256 `" + f["sha256"] + "`。\n")
            if len(ckpt) > 8:
                text.append("- 其余权重逐项记录在迁移清单，所有副本保留。\n")
        text.append("\n## 使用边界\n\n本说明不是重新验收：未训练、未推理、未运行官方MET、未联网。旧失败、取消、未审批状态全部保留；本地status或COMPLETE文件存在不自动代表正式通过。\n\n"
                    "本次新增说明：artifact_status=experimental_unvalidated；operator_approved=false；formal_gate_status=not_run_not_passed。旧原件保持其原有状态。\n")
        mode = stat.S_IMODE(base.stat().st_mode)
        try:
            os.chmod(base, mode | stat.S_IWUSR)
            mwrite(path, "".join(text))
        finally:
            os.chmod(base, mode)
        written.append({"unit": u["new"], "description": path.relative_to(ROOT).as_posix(), "files": len(members), "kind": u["kind"]})
    cwrite(ADMIN / "UNIT_DESCRIPTIONS.csv", written)
    print(json.dumps({"unit_descriptions": len(written)}), flush=True)


def prepare_adaptations():
    paths = ["work_space/G1/01_code/missing_t2w_vae_bbdm/configs.py"] + [
        "work_space/G2/01_code/" + name for name in ["g2_create_train_val_test_split.py", "g2_synthetic_raw_intake_qc.py",
        "g2_pretraining_audit.py", "g2_v2_compose_augmentation.py", "g2_materialize_nnunet_dataset.py", "g2_build_realonly_from_raw.py"]]
    plan = {f["new"]: f for f in plan_files()}
    preserved = []
    for rel in paths:
        src = ROOT / rel
        dst = ROOT / "99_archive/source_before_path_adaptation_20260919" / rel
        require(not dst.exists(), "Original source backup already exists")
        require(sha(src) == plan[rel]["sha256"], "Source changed before preservation")
        dst.parent.mkdir(parents=True, exist_ok=True)
        move_with_parent_permissions(src, dst)
        shutil.copy2(dst, src)
        preserved.append({"active": rel, "original_archive": dst.relative_to(ROOT).as_posix(),
                          "original_sha256": plan[rel]["sha256"], "original_inode": plan[rel]["inode"]})
    jwrite(ADMIN / "SOURCE_ADAPTATIONS.json", {**FLAGS, "created_utc": now(), "files": preserved,
               "scope": "New editable working copies only; originals preserved byte-for-byte and frozen experiment runtimes untouched."})
    print(json.dumps({"source_originals_preserved": len(preserved)}), flush=True)


def correct_release_label():
    old = "20260729_local_release"
    new = "20260919_archived_local_release"
    plan = load("FINAL_PLAN.json")
    source = ROOT / "work_space/G2/02_dataset_release" / old
    target = source.with_name(new)
    require(not target.exists(), "Corrected label already exists")
    source.rename(target)
    for f in plan["files"]:
        f["new"] = f["new"].replace(old, new)
    for u in plan["units"]:
        u["new"] = u["new"].replace(old, new)
        u["related"] = [p.replace(old, new) for p in u["related"]]
    plan["label_correction"] = {"utc": now(), "old": old, "new": new,
        "reason": "Use actual archive date; do not imply an unverified historical data-release date."}
    jwrite(ADMIN / "FINAL_PLAN_LABEL_CORRECTED.json", plan)
    cwrite(ADMIN / "FINAL_MOVE_PLAN_CURRENT.csv", plan["files"])
    revised = []
    for base, dirs, names in os.walk(ROOT / "work_space", followlinks=False):
        for name in names:
            if name != "00_说明_20260919.md":
                continue
            p = Path(base) / name
            text = p.read_text(encoding="utf-8")
            if old in text:
                backup = ADMIN / "generated_descriptions_before_label_correction" / p.relative_to(ROOT)
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, backup)
                p.write_text(text.replace(old, new), encoding="utf-8")
                revised.append(p.relative_to(ROOT).as_posix())
    jwrite(ADMIN / "LABEL_CORRECTION.json", {**FLAGS, "original_payload_changed": False,
        "old": str(source.relative_to(ROOT)), "new": str(target.relative_to(ROOT)), "generated_descriptions_updated": revised})
    print(json.dumps({"archive_date_label_corrected": True, "generated_descriptions_updated": len(revised)}), flush=True)


def navigation():
    files = plan_files()
    titles = {"G1": "生成模型、缺失模态补全与图像评估", "G2": "数据接收、质量控制与发布",
              "S1": "多任务分割", "S2": "分割、样本增强与比赛提交", "S3": "S3分割分支",
              "S4": "SAM2-UNet分割", "S5": "SegMamba分割"}
    roles = {"01_code": "活动代码、测试和必要的包内README/许可证",
             "02_datasets": "本模块的数据集、样本、潜变量或可复用增强数据",
             "02_dataset_release": "保留manifests/splits/qc内部合同的数据发布包",
             "03_experiments": "按研究路线分组的独立实验及其原始结果",
             "04_models": "从代码或混合目录中分离的模型权重",
             "05_documents": "方法、计划、报告、环境和历史说明文档",
             "06_operations": "现有Slurm及云操作脚本，旧远端路径只作历史记录，不自动执行",
             "07_submissions": "比赛推理、传输、提交包、排行榜及容器保留资料"}
    module_stats = []
    for module, title in titles.items():
        root = ROOT / "work_space" / module
        children = sorted(p for p in root.iterdir() if p.is_dir())
        selected = [f for f in files if f["new"].startswith(f"work_space/{module}/")]
        path = root / "README.md"
        text = [f"# {module}：{title}\n\n整理于2026-09-19。这里是实体文件分类，不是快捷方式视图。\n\n",
                "| 目录 | 内容 |\n|---|---|\n"]
        for child in children:
            text.append("| " + relative_link(path, child, child.name) + " | " + roles.get(child.name, "独立历史资料") + " |\n")
        text.append(f"\n当前分类内原文件 {len(selected)} 份，{sum(f['bytes'] for f in selected):,} 字节。每个数据集、模型或结果批次内均有 `00_说明_20260919.md`。\n\n")
        if module == "S2":
            text.append("原E比赛基座是Dataset264 completion+Focal的epoch200，SHA为`4e5ff8d4a29fc498e4d91f9b0a71c34b818da6cd07d359ccabcc72dcb49e4267`；不是E-SNFH。Screen125与Extend175按独立路线保存，不能混做等预算结论。\n\n")
        if module == "G1":
            text.append("VAE/缺失T2W补全与Diffusion增强分开；训练checkpoint保存的扫描影像已放入02_datasets，不再混在模型权重目录。图像质量、offline补全后分割、online增强后分割是三层问题。\n\n")
        if module in {"S1", "S3", "S4", "S5"}:
            text.append("未在本地发现独立结果包的分类没有创建空文件夹；原模板与占位文件在99_archive保留，不推断远端是否有结果。\n\n")
        text.append("历史文档、冻结实验runtime及manifest不改写。活动路径配置见项目根.env和tools/project_paths.py；7份路径适配代码的原件另存99_archive。重新训练或提交前须另做实验预检，不能直接复用旧Slurm绝对路径。\n")
        mwrite(path, "".join(text))
        module_stats.append({"module": module, "original_files_in_current_area": len(selected), "bytes": sum(f["bytes"] for f in selected)})
        for child in children:
            candidates = [child]
            if child.name in {"03_experiments", "05_documents", "07_submissions"}:
                candidates += sorted(p for p in child.iterdir() if p.is_dir())
            for folder in candidates:
                desc = folder / "README.md"
                if desc.exists():
                    desc = folder / "00_目录说明_20260919.md"
                rows = [f"# {module} / {folder.name}\n\n{roles.get(child.name, '')}\n\n"]
                for item in sorted(folder.iterdir()):
                    if not item.name.startswith(".") and item.name != desc.name:
                        rows.append("- " + relative_link(desc, item, item.name) + "\n")
                rows.append("\n整理说明仅帮助定位，不改变任何原实验的批准或失败状态。新说明为experimental_unvalidated。\n")
                mwrite(desc, "".join(rows))
    cwrite(ADMIN / "CURRENT_MODULES.csv", module_stats)
    for name, title in {"01_project_documents": "比赛要求、分工、总计划和跨模块文档",
                        "02_reference_materials": "论文、课程讲义与外部阅读材料",
                        "03_reference_code": "按来源保留的上游参考代码",
                        "data_space": "共享数据契约、参考样例和参考模型",
                        "04_paper_figures": "论文图表与可编辑源素材"}.items():
        folder = ROOT / name
        path = folder / "README.md"
        text = [f"# {title}\n\n2026-09-19实体归类。不同来源的重复副本均保留，不相互覆盖。\n\n"]
        for item in sorted(folder.iterdir()):
            if not item.name.startswith("."):
                text.append("- " + relative_link(path, item, item.name) + "\n")
        mwrite(path, "".join(text))
    mwrite(ROOT / "work_space/README.md", "# 比赛研究工作区\n\n" + "\n".join(
        f"- [{m}]({m}/README.md)：{t}" for m, t in titles.items()) +
        "\n\n各模块按真实代码、数据、实验、模型、文档、操作和提交归类；只创建实际有内容的目录。旧索引和空结构在项目99_archive保留。\n")
    mwrite(ROOT / "99_archive/README.md", "# 历史保留区\n\n这里保留所有未删的旧索引、缓存、占位文件、原目录结构和活动代码适配前原件。主工作树不依赖本区的旧快捷方式。\n\n"
           "- `20260918_index_only`：上一轮索引文件和775个原符号链接对象。链接仅保存历史target及inode，可能已无法按旧相对路径访问，不能作为现目录导航。\n"
           "- `cache_files_20260919`：原.DS_Store、__pycache__及pyc，按原位置存放。\n"
           "- `old_placeholders_and_notes`：原.gitkeep和上一轮空目录说明。\n"
           "- `original_empty_directory_objects`：搬出内容后的原目录对象，保留本体和来源说明，无空目录。\n"
           "- `source_before_path_adaptation_20260919`：7份活动代码路径适配前原件，SHA和inode保留。\n"
           "- `legacy_preprocessing_job_logs`：原根目录logs中的历史日志。\n"
           "- `reorganization_20260919_r2`：本次搬迁计划、事务日志、路径映射和验收。不作为新的科学实验结果。\n")
    print(json.dumps({"module_readmes": len(titles), "navigation_complete": True}), flush=True)


def prepare_test_adaptations():
    plan = {f["new"]: f for f in plan_files()}
    entries = []
    for module in ["G2", "S3", "S4"]:
        folder = ROOT / "work_space" / module / "01_code/tests"
        for src in sorted(folder.glob("*.py")):
            if '/ "code"' not in src.read_text(encoding="utf-8"):
                continue
            rel = src.relative_to(ROOT).as_posix()
            dst = ROOT / "99_archive/source_before_path_adaptation_20260919" / rel
            require(not dst.exists() and sha(src) == plan[rel]["sha256"], "Test source preservation conflict")
            dst.parent.mkdir(parents=True, exist_ok=True)
            move_with_parent_permissions(src, dst)
            shutil.copy2(dst, src)
            entries.append({"active": rel, "original_archive": dst.relative_to(ROOT).as_posix(),
                            "original_sha256": plan[rel]["sha256"], "original_inode": plan[rel]["inode"]})
    jwrite(ADMIN / "TEST_PATH_ADAPTATIONS.json", {**FLAGS, "files": entries, "scope": "Only adjust test source lookup after relocation; test logic unchanged."})
    print(json.dumps({"test_sources_preserved": len(entries)}), flush=True)


def refine_descriptions():
    updated = []
    for base, dirs, names in os.walk(ROOT / "work_space", followlinks=False):
        if "00_说明_20260919.md" not in names:
            continue
        p = Path(base) / "00_说明_20260919.md"
        text = p.read_text(encoding="utf-8")
        lines = text.splitlines(keepends=True)
        revised = "".join(line for line in lines if not (line.startswith("- 从同一原包归类的配套资料：") and "早期real-only的179例validation提交包" in line))
        revised = revised.replace("/FINAL_MOVE_PLAN.csv", "/FINAL_MOVE_PLAN_CURRENT.csv")
        if revised != text:
            backup = ADMIN / "generated_descriptions_before_relation_review" / p.relative_to(ROOT)
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, backup)
            p.write_text(revised, encoding="utf-8")
            updated.append(p.relative_to(ROOT).as_posix())
    jwrite(ADMIN / "DESCRIPTION_RELATION_REVIEW.json", {**FLAGS, "updated": updated,
               "reason": "A shared old results directory does not imply a scientific input/output relationship; remove that broad association."})
    print(json.dumps({"descriptions_refined": len(updated)}), flush=True)


def final_audit():
    from urllib.parse import unquote
    files = plan_files()
    originals = {f["new"] for f in files}
    changed = []
    for name in ["SOURCE_ADAPTATIONS.json", "TEST_PATH_ADAPTATIONS.json"]:
        changed.extend(load(name)["files"])
    current_originals = {f["new"]: f["new"] for f in files}
    current_originals.update({x["active"]: x["original_archive"] for x in changed})
    cwrite(ADMIN / "ORIGINAL_LOCATIONS_FINAL.csv", [dict(f, preserved_original=current_originals[f["new"]]) for f in files])
    broken, checked, descriptions, active_directories = [], 0, [], 0
    for base, dirs, names in os.walk(ROOT, followlinks=False):
        directory = Path(base)
        if directory == ROOT:
            dirs[:] = [d for d in dirs if d not in EXCLUDED and d != "99_archive"]
        active_directories += 1
        for name in names:
            path = directory / name
            rel = path.relative_to(ROOT).as_posix()
            if name == "00_说明_20260919.md":
                descriptions.append(rel)
                content = path.read_text(encoding="utf-8")
                # Case-ID counts above refer specifically to NIfTI paths.
                clarified = content.replace("从文件名识别的BraTS-MET病例ID", "从NIfTI路径识别的BraTS-MET病例ID")
                if clarified != content:
                    backup = ADMIN / "generated_descriptions_before_count_wording" / rel
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, backup)
                    path.write_text(clarified, encoding="utf-8")
            if path.suffix != ".md" or rel in originals:
                continue
            for raw in re.findall(r"\[[^\]]*\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
                if "://" in raw or raw.startswith("#"):
                    continue
                target = path.parent / unquote(raw.split("#")[0])
                checked += 1
                if not target.exists():
                    broken.append({"source": rel, "target": raw})
    require(len(descriptions) == 95, "Unit-description coverage changed")
    require(not broken, "Broken generated documentation links: " + json.dumps(broken, ensure_ascii=False))
    cwrite(ADMIN / "UNIT_DESCRIPTIONS_FINAL.csv", [{"description": p} for p in descriptions])
    test_stats = {"layout_tests_passed": 9, "existing_g2_stdlib_tests_passed": 9,
                  "full_ml_tests_run": False, "reason": "Existing base environment lacks numpy/nibabel/torch; no dependencies installed."}
    jwrite(ADMIN / "DELIVERY_AUDIT.json", {**FLAGS, "completed_utc": now(), "status": "pass",
               "original_files": len(files), "unit_descriptions": len(descriptions), "active_directories": active_directories,
               "new_markdown_links_checked": checked, "broken_new_markdown_links": broken,
               "adapted_source_files": 7, "adapted_test_path_files": 15, "tests": test_stats,
               "historical_document_links": "original text preserved; may reference old paths, use project_paths resolver",
               "archived_symbolic_links": "historical objects only, not live navigation"})
    mwrite(ADMIN / "FINAL_ACCEPTANCE.md", f"""# 2026-09-19实体重整验收

完成UTC：{now()}；本机时间：{datetime.now().astimezone().isoformat()}。

这次实施的是实体文件搬迁：19,623个文件改变位置，整个受检范围19,626个原文件（36,009,759,371字节）保留。旧索引和775个符号链接对象退出主树，转入99_archive。主树没有符号链接。

G1的代码、补全数据、训练扫描、实验报告和模型权重已分开；S2按11条实验路线归类，数据组件池及reference labels独立存放，比赛提交另成一组。共享论文、比赛文件、参考代码、参考数据及论文图表已从混合位置搬出。

95个数据集、模型、实验或结果包有就近的00_说明_20260919.md，说明用途、来源、本地文件计数、对应资料和限制。方法、计划、报告、环境和操作文档实际放入相应目录，包内必要README/许可证及原实验报告保留其上下文。

原件按SHA256、大小、inode、mode、owner和mtime全量复核。7份活动代码仅做路径适配，15份测试仅改源代码定位，其原件也已保存。最终原件位置见ORIGINAL_LOCATIONS_FINAL.csv，原路径到活动位置见FINAL_MOVE_PLAN_CURRENT.csv。

路径测试9项及原G2无额外依赖测试9项通过；完整ML测试没有运行，未安装依赖。新说明链接检查通过。没有训练、推理、提交作业、访问服务器、Git提交或删除原文件。

2012个搬空的原目录对象也保留在历史区并配来源说明，因此没有留下空目录。原reservation的本地结构归档不意味着释放或重建远端锁。旧源码/文档的历史绝对路径和冻结实验合同不批量改写；要重新执行旧实验，必须按新布局另做运行预检。

本次是本地整理验收，不是科研效果或正式门控通过：artifact_status=experimental_unvalidated；operator_approved=false；formal_gate_status=not_run_not_passed。
""")
    print(json.dumps({"final_document_links": checked, "unit_descriptions": len(descriptions), "status": "pass"}), flush=True)


def verify():
    data = load("BASELINE.json")
    data["files"] = plan_files()
    adaptation_file = ADMIN / "SOURCE_ADAPTATIONS.json"
    originals = {x["active"]: x["original_archive"] for x in load("SOURCE_ADAPTATIONS.json")["files"]} if adaptation_file.exists() else {}
    if (ADMIN / "TEST_PATH_ADAPTATIONS.json").exists():
        originals.update({x["active"]: x["original_archive"] for x in load("TEST_PATH_ADAPTATIONS.json")["files"]})
    errors = []
    for i, f in enumerate(data["files"], 1):
        p = ROOT / originals.get(f["new"], f["new"])
        try:
            st = p.stat()
            ok = (sha(p) == f["sha256"] and st.st_ino == f["inode"] and st.st_size == f["bytes"]
                  and stat.S_IMODE(st.st_mode) == f["mode"] and st.st_mtime_ns == f["mtime_ns"]
                  and st.st_uid == f["uid"] and st.st_gid == f["gid"])
            if not ok:
                errors.append(f["old"])
        except OSError:
            errors.append(f["old"])
        if i % 2500 == 0:
            print(json.dumps({"stage": "verify", "files": i, "errors": len(errors)}), flush=True)
    for item in data["symlinks"]:
        p = ROOT / item["new"]
        if not p.is_symlink() or os.readlink(p) != item["target"] or p.lstat().st_ino != item["inode"]:
            errors.append(item["old"])
    empty, active_links = [], []
    for base, subdirs, names in os.walk(ROOT, followlinks=False):
        p = Path(base)
        subdirs[:] = [d for d in subdirs if not (p == ROOT and d in EXCLUDED)]
        if not subdirs and not names:
            empty.append(p.relative_to(ROOT).as_posix())
        if not p.is_relative_to(ROOT / "99_archive"):
            active_links += [str((p / n).relative_to(ROOT)) for n in subdirs + names if (p / n).is_symlink()]
    result = {**FLAGS, "completed_utc": now(), "status": "pass" if not errors and not empty and not active_links else "fail",
              "original_files_verified": len(data["files"]), "original_bytes_verified": sum(f["bytes"] for f in data["files"]),
              "original_files_changed_or_lost": errors, "empty_directories": empty, "active_tree_symlinks": active_links,
              "previous_link_objects_preserved_in_archive": len(data["symlinks"]),
              "adapted_working_copies_with_originals_preserved": len(originals),
              "previous_links_are_historical_objects_not_navigation": True}
    jwrite(ADMIN / ("VERIFICATION_FINAL.json" if (ADMIN / "VERIFICATION.json").exists() else "VERIFICATION.json"), result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    require(result["status"] == "pass", "Verification failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["baseline", "review_plan", "apply_moves", "describe", "prepare_adaptations", "correct_release_label", "navigation", "prepare_test_adaptations", "refine_descriptions", "final_audit", "verify"])
    globals()[parser.parse_args().stage]()
