"""Audited, local-only S1/S2 relocation. Never executes experiment entry points."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
AUDIT = Path(__file__).resolve().parent
LEGACY = ROOT / "99_archive/s1_s2_legacy_20260919"
STATUS = {"artifact_status": "experimental_unvalidated", "operator_approved": False,
          "formal_gate_status": "not_run_not_passed"}
S1 = ROOT / "work_space/S1"
S2 = ROOT / "work_space/S2"


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def relative(path):
    return path.relative_to(ROOT).as_posix()


def write_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def record(path):
    info = path.stat()
    return {"path": relative(path), "sha256": digest(path), "bytes": info.st_size,
            "mode": stat.S_IMODE(info.st_mode), "inode": info.st_ino,
            "mtime_ns": info.st_mtime_ns, "uid": info.st_uid, "gid": info.st_gid}


def script_group(module, name):
    if module == "S1":
        return "01_labels" if name[:2] in {"01", "02", "03", "08"} else "02_audit_split"
    number = name[:2]
    if name == "setup_brats_eval_env.sh":
        return "08_environment"
    if number in {"01", "02", "03", "04", "05", "08", "09"}:
        return "01_data_preparation"
    if number in {"06", "07", "10", "11", "28"} or name in {
        "29_preflight_official_inference_model.py", "30_finalize_official_inference_no_zip.py"
    }:
        return "02_evaluation_submission"
    if number in {"12", "14", "21", "24"}:
        return "03_component_pool_masks"
    if number in {"13", "15", "16", "17", "18", "19", "20", "22", "23"}:
        return "04_route_a"
    if number in {"25", "26", "27"}:
        return "05_original_e_continue"
    if number in {"45", "46", "47"}:
        return "07_fix_v3"
    if number.isdigit() and 29 <= int(number) <= 44:
        return "06_fix_v2"
    raise ValueError(f"Unclassified script: {name}")


def plan():
    moves = []
    def move(source, destination, reason):
        if not source.is_file() or source.is_symlink():
            raise ValueError(f"Expected regular source: {source}")
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(destination)
        moves.append({"old": relative(source), "new": relative(destination), "reason": reason})

    for module in (S1, S2):
        repo = module / "01_code/repository"
        for source in sorted((repo / "scripts").iterdir()):
            if source.name == "02_find_invalid_labels_old.py":
                destination = LEGACY / "01_legacy_source" / module.name / "scripts" / source.name
                reason = "Retain explicitly named old source outside active scripts"
            else:
                destination = source.parent / script_group(module.name, source.name) / source.name
                reason = "Group script by responsibility, keeping original filename"
            move(source, destination, reason)

    for name in ("datasets/brats_multitask_dataset.py.bak", "datasets/brats_multitask_dataset.py.bak2",
                 "trainers/trainer_v1_final_before_dice.py"):
        move(S1 / "01_code/repository" / name, LEGACY / "01_legacy_source/S1" / name,
             "Preserve backup or pre-Dice trainer; not an active entry point")
    for name in ("train.py", "inference.py", "models/README.md", "inference/README.md"):
        source = S2 / "01_code/repository" / name
        assert source.stat().st_size == 0
        move(source, LEGACY / "02_placeholders/S2" / name, "Preserve empty template outside active code")

    # Split files are a data contract, not Python source; their bytes stay frozen.
    for source in sorted((S2 / "01_code/repository/data/splits/completion_warmstart").iterdir()):
        move(source, S2 / "02_datasets/undated_fixed_split_contracts/completion_warmstart" / source.name,
             "Separate original fixed split and audit data from source")
    move(S2 / "01_code/repository/configs/split_info.md", S2 / "05_documents/03_operations/split_info.md",
         "Move split documentation out of code configuration directory")
    move(S2 / "05_documents/01_methods/lesion_pool.csv", S2 / "02_datasets/undated_lesion_pool_table/lesion_pool.csv",
         "Preserve lesion table as a dataset; do not infer source equivalence")
    move(S2 / "05_documents/01_methods/BraTS 2025.pdf", S2 / "05_documents/07_references/BraTS 2025.pdf",
         "Separate reference paper from authored methods")

    s1_docs = {"architecture.md": "01_methods/architecture.md", "data_preparation.md": "03_operations/data_preparation.md",
               "training.md": "03_operations/training.md", "inference.md": "03_operations/inference.md",
               "S1_服务器运行手册.md": "03_operations/S1_服务器运行手册_原代码文档版.md",
               "04_experiment_log.md": "04_reports/04_experiment_log.md", "S1_experiment.txt": "04_reports/S1_experiment.txt"}
    s2_docs = {"TUNING.md": "01_methods/TUNING.md", "MET_AUG_Fix_v1_完整修复方案.md": "02_plans/MET_AUG_Fix_v1_完整修复方案.md",
               "TRAINING.md": "03_operations/TRAINING.md", "INFERENCE.md": "03_operations/INFERENCE.md",
               "DATASET.md": "03_operations/DATASET.md", "S2_服务器运行手册.md": "03_operations/S2_服务器运行手册.md",
               "S2_experiment.txt": "04_reports/S2_experiment.txt"}
    for module, mapping in ((S1, s1_docs), (S2, s2_docs)):
        for name, target in mapping.items():
            move(module / "05_documents/06_code_documentation" / name, module / "05_documents" / target,
                 "Classify documentation by content, preserve original text")
        move(module / "05_documents/06_code_documentation/README.md",
             LEGACY / "04_previous_navigation" / module.name / "06_code_documentation_README.md",
             "Archive prior navigation for emptied category")

    # Historical run bundles remain byte-exact, including environment files and launch records.
    slurm = S2 / "06_operations/slurm"
    for source in sorted(slurm.rglob("*")):
        if not source.is_file():
            continue
        rel = source.relative_to(slurm)
        if rel.parts[0] == "legacy_realonly":
            target = "03_nyu_realonly_legacy_20260919"
            subpath = Path(*rel.parts[1:])
        elif "uhost" in source.name or "h20" in source.name:
            target = "02_h20_legacy_bundle_20260919"
            subpath = rel
        else:
            target = "01_ecnu_legacy_bundle_20260919"
            subpath = rel
        move(source, S2 / "06_operations" / target / subpath, "Keep historical server bundle unmodified")
    for source in sorted((S2 / "06_operations/cloud").iterdir()):
        move(source, S2 / "06_operations/04_cloud_legacy_bundle_20260919" / source.name,
             "Keep cloud code and its launch evidence together, unchanged")

    assert len({m["old"] for m in moves}) == len(moves)
    assert len({m["new"] for m in moves}) == len(moves)
    snapshots = []
    for module in (S1, S2):
        for path in sorted(module.rglob("*")):
            if path.is_file() and not path.is_symlink():
                snapshots.append(record(path))
    for path in [ROOT / "tools/project_paths.py", ROOT / "tools/test_project_paths.py", ROOT / "README.md"]:
        snapshots.append(record(path))
    write_json(AUDIT / "BASELINE.json", {**STATUS, "utc": timestamp(), "files": snapshots})
    write_json(AUDIT / "PLAN.json", {**STATUS, "utc": timestamp(), "moves": moves,
               "preserve_packages": ["custom_nnunet", "S2/03_experiments", "S2/04_models", "S2/07_submissions"],
               "no_remote_execution": True, "no_dependency_installation": True})
    print(json.dumps({"phase": "plan", "moves": len(moves), "baseline_files": len(snapshots)}), flush=True)


def execute_moves():
    value = json.loads((AUDIT / "PLAN.json").read_text())
    baseline = {r["path"]: r for r in json.loads((AUDIT / "BASELINE.json").read_text())["files"]}
    for move in value["moves"]:
        source, target = ROOT / move["old"], ROOT / move["new"]
        assert digest(source) == baseline[move["old"]]["sha256"], source
        assert not target.exists(), target
    with (AUDIT / "MOVES.jsonl").open("x", encoding="utf-8") as journal:
        for move in value["moves"]:
            source, target = ROOT / move["old"], ROOT / move["new"]
            target.parent.mkdir(parents=True, exist_ok=True)
            source.rename(target)
            journal.write(json.dumps(move, ensure_ascii=False) + "\n")
            journal.flush()
    print(json.dumps({"phase": "move", "moved": len(value["moves"])}), flush=True)


def save_updated(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") == content:
            return
        original = record(path)
        backup = LEGACY / "03_before_path_updates" / path.relative_to(ROOT)
        if backup.exists():
            raise FileExistsError(f"Already backed up; review a second edit explicitly: {path}")
        backup.parent.mkdir(parents=True, exist_ok=True)
        path.rename(backup)
        with (AUDIT / "ADAPTATIONS.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps({"active": relative(path), "backup": relative(backup), "original": original}, ensure_ascii=False) + "\n")
        mode = original["mode"]
    else:
        mode = 0o644
    with path.open("x", encoding="utf-8") as handle:
        handle.write(content)
    path.chmod(mode)


def protect_for_manual_edit(path):
    import shutil
    original = record(path)
    backup = LEGACY / "03_before_path_updates" / path.relative_to(ROOT)
    if backup.exists():
        raise FileExistsError(backup)
    backup.parent.mkdir(parents=True, exist_ok=True)
    path.rename(backup)
    shutil.copy2(backup, path)
    with (AUDIT / "ADAPTATIONS.jsonl").open("a", encoding="utf-8") as log:
        log.write(json.dumps({"active": relative(path), "backup": relative(backup), "original": original}, ensure_ascii=False) + "\n")


def adapt():
    planned = json.loads((AUDIT / "PLAN.json").read_text())["moves"]
    changed = []
    for module in (S1, S2):
        repo = module / "01_code/repository"
        scripts = {Path(m["old"]).name: Path(m["new"]).relative_to(repo.relative_to(ROOT)).as_posix()
                   for m in planned if m["old"].startswith(relative(repo / "scripts") + "/")
                   and m["new"].startswith(relative(repo) + "/")}
        for path in sorted(repo.rglob("*")):
            if not path.is_file() or path.suffix not in {".py", ".sh"}:
                continue
            if "custom_nnunet" in path.parts:
                continue  # Checkpoint import identities and frozen contract keys stay byte-exact.
            text = path.read_text(encoding="utf-8")
            original = text
            if path.parent.parent == repo / "scripts" and path.suffix == ".py":
                text = re.sub(r'(REPO(?:SITORY)?_ROOT = Path\(__file__\)\.resolve\(\)\.parents)\[1\]', r'\g<1>[2]', text)
            for name, target in scripts.items():
                group = Path(target).parent.name
                # Only repository-local expressions; frozen runtime contract strings are not rewritten.
                for quote in ('"', "\'"):
                    old = f'{quote}scripts{quote} / {quote}{name}{quote}'
                    new = f'{quote}scripts{quote} / {quote}{group}{quote} / {quote}{name}{quote}'
                    text = text.replace(old, new)
                if path.name in {"train.sh", "setup_brats_eval_env.sh", "28_prepare_official_inference_dual_h20.py",
                                 "30_finalize_official_inference_no_zip.py"}:
                    text = text.replace("scripts/" + name, target)
            if path.name in {"test_met_aug_preparation_scripts.py", "test_met_aug_fix_v2_scripts.py"}:
                text = text.replace('    path = REPOSITORY_ROOT / "scripts" / filename',
                    '    matches = list((REPOSITORY_ROOT / "scripts").glob(f"*/{filename}"))\n'
                    '    if len(matches) != 1:\n'
                    '        raise RuntimeError(f"expected exactly one script: {filename}; got {matches}")\n'
                    '    path = matches[0]')
            if path.name == "setup_brats_eval_env.sh":
                text = text.replace('${SCRIPT_DIR}/.."', '${SCRIPT_DIR}/../.."')
                text = text.replace('${SCRIPT_DIR}/../../../../.."', '${S2_REPO}/../../../.."')
            if module == S2 and path.name in {"train.sh", "infer.sh"}:
                text = text.replace('${REPO_DIR}/data/splits/', '${REPO_DIR}/../../02_datasets/undated_fixed_split_contracts/')
            if text != original:
                save_updated(path, text)
                changed.append(relative(path))
    write_json(AUDIT / "PATH_ADAPTATION_SUMMARY.json", {**STATUS, "utc": timestamp(), "files": changed})
    print(json.dumps({"phase": "adapt", "files": len(changed)}), flush=True)


def preserve_empty_directories():
    records = []
    for module in (S1, S2):
        for path in sorted([p for p in module.rglob("*") if p.is_dir() and not p.is_symlink()],
                           key=lambda p: len(p.parts), reverse=True):
            if not any(path.iterdir()):
                old = relative(path)
                destination = LEGACY / "05_original_directory_objects" / path.relative_to(ROOT)
                # Flat IDs prevent a preserved child from colliding with its old parent object.
                destination = LEGACY / "05_original_directory_objects" / hashlib.sha256(old.encode()).hexdigest()[:16]
                destination.parent.mkdir(parents=True, exist_ok=True)
                info = path.stat()
                path.rename(destination)
                (destination / "ORIGIN.json").write_text(json.dumps({**STATUS, "original_path": old,
                    "inode": info.st_ino, "note": "Emptied original directory object retained; not an active path."},
                    ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                records.append({"old": old, "new": relative(destination)})
    write_json(AUDIT / "EMPTY_DIRECTORY_OBJECTS.json", {**STATUS, "directories": records})


def navigation():
    footer = "\n整理时间：2026-09-19。本地整理标记：experimental_unvalidated；operator_approved=false；formal_gate_status=not_run_not_passed。\n"
    code_note = ("历史源码原件、备份及占位文件保留在项目 `99_archive/s1_s2_legacy_20260919`。\n"
                 "冻结实验runtime和已提交作业证据未改写；重新运行须重新核对数据路径、环境和实验计划。\n")
    titles = {"01_data_preparation": "数据准备与固定划分", "02_evaluation_submission": "官方评估与提交工具",
              "03_component_pool_masks": "组件池、掩膜与坐标检查", "04_route_a": "Route A 门控与检查",
              "05_original_e_continue": "原 E 续训辅助", "06_fix_v2": "FixV2 校准与评审",
              "07_fix_v3": "FixV3 验收与对比", "08_environment": "环境配置工具",
              "01_labels": "标签准备", "02_audit_split": "数据审计与划分"}
    for module in (S1, S2):
        repo = module / "01_code/repository"
        groups = sorted(p for p in (repo / "scripts").iterdir() if p.is_dir())
        links = "\n".join(f"- [{p.name}]({p.name}/README.md)：{titles[p.name]}" for p in groups)
        save_updated(repo / "scripts/README.md", f"# {module.name} 脚本\n\n{links}\n\n" + code_note + footer)
        for group in groups:
            entries = sorted(p for p in group.iterdir() if p.is_file() and p.name != "README.md")
            body = "\n".join(f"- [{p.name}]({p.name})" for p in entries)
            save_updated(group / "README.md", f"# {titles[group.name]}\n\n{body}\n\n" + code_note + footer)
        if module == S1:
            body = "保留当前多任务训练器、模型、数据加载器、损失、指标和推理包；仅将旧版本移出活动目录。\n"
            categories = ["configs", "datasets", "models", "trainers", "inference", "losses", "metrics", "scripts"]
        else:
            body = ("`custom_nnunet` 保持原模块名称和结构，供nnUNet动态加载和checkpoint兼容使用；本次未修改其30个源码文件。\n"
                    "`train.sh`、`infer.sh`、`inference_frozen.py` 保留为真实入口。空的train.py/inference.py已归档。\n"
                    "固定划分与审计已移至 [数据合同](../../02_datasets/undated_fixed_split_contracts/README.md)，BRATS_SPLIT_DIR显式设置优先。\n")
            categories = ["custom_nnunet", "scripts", "tests"]
        links = "\n".join(f"- [{name}]({name}/)" for name in categories)
        save_updated(repo / "README.md", f"# {module.name} 代码\n\n{body}\n{links}\n\n" + code_note + footer)
        save_updated(module / "01_code/README.md", f"# {module.name} 代码\n\n[活动代码包](repository/README.md)\n\n" + code_note + footer)
        save_updated(module / "01_code/00_说明_20260919.md", f"# {module.name} 代码整理说明\n\n[代码结构](repository/README.md)；[脚本分类](repository/scripts/README.md)。\n\n" + code_note + footer)
        docs = module / "05_documents"
        for directory in [docs, *sorted(p for p in docs.iterdir() if p.is_dir())]:
            entries = sorted(p for p in directory.iterdir() if p.name != "README.md")
            body = "\n".join(f"- [{p.name}](<{p.name}{'/' if p.is_dir() else ''}>)" for p in entries)
            save_updated(directory / "README.md", f"# {module.name} 文档：{directory.name}\n\n{body}\n\n"
                         "原计划、原报告和历史手册正文不改写；旧命令中的路径需要按当前布局另做预检。\n" + footer)
        if module == S1:
            body = ("本地保留多任务分割代码、方案、运行手册和实验记录；未发现独立训练结果或模型包，不创建空目录。\n"
                    "- [代码](01_code/repository/README.md)\n- [文档](05_documents/README.md)\n- [运行资料](06_operations/README.md)\n"
                    "\n两份服务器手册内容不同，均保留在03_operations，并标明原代码文档版。\n")
        else:
            body = "\n".join(f"- [{p.name}]({p.name}/README.md)" for p in sorted(module.iterdir()) if p.is_dir())
            body += ("\n\n原 E 比赛基座与 E-SNFH 不同。Screen125等预算、Extend175收敛实验及官方MET评估保持分组，不能混作同一比较。\n"
                     "源数据、模型和实验包未改写。代码脚本按用途拆分；历史服务器操作包仅作复核，不直接执行。\n")
        save_updated(module / "README.md", f"# {module.name}\n\n{body}\n" + footer)
    split = S2 / "02_datasets/undated_fixed_split_contracts"
    save_updated(split / "README.md", "# 固定划分合同\n\n来自原S2代码data/splits目录，5份TXT/JSON原件保持SHA不变，不能当成新划分。\n"
        "本地仅保留completion_warmstart；current/legacy划分未因本次整理而创建。\n\n"
        "[原文件](completion_warmstart/)；[原划分说明](../../05_documents/03_operations/split_info.md)。\n" + footer)
    save_updated(split / "00_说明_20260919.md", "# 数据用途\n\n[固定划分说明](README.md)。本目录是病例清单和审计，不包含MRI影像。\n" + footer)
    pool = S2 / "02_datasets/undated_lesion_pool_table"
    with (pool / "lesion_pool.csv").open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    save_updated(pool / "README.md", f"# 历史病灶池表\n\n[lesion_pool.csv](lesion_pool.csv)：{len(rows)}行。\n"
        "原位于S2方法文档目录，表头含病例、病灶体积、包围框及各类别计数。生成日期和与当前FixV3组件池的对应关系未核实，不能据名称视为同一数据版本。\n" + footer)
    save_updated(pool / "00_说明_20260919.md", "# 数据用途\n\n[历史病灶池表说明](README.md)。只搬迁，不修改CSV或重算统计。\n" + footer)
    datasets = S2 / "02_datasets"
    body = "\n".join(f"- [{p.name}]({p.name}/)" for p in sorted(datasets.iterdir()) if p.is_dir())
    save_updated(datasets / "README.md", "# S2 数据\n\n" + body + footer)
    operations = S2 / "06_operations"
    body = "\n".join(f"- [{p.name}]({p.name}/README.md)" for p in sorted(operations.iterdir()) if p.is_dir())
    save_updated(operations / "README.md", "# S2 历史运行资料\n\n" + body + "\n\n这些是历史运行包，脚本、环境文件、launch记录保持原字节。远端旧路径未自动替换；不是可以直接提交的新计划。\n" + footer)
    for directory in sorted(p for p in operations.iterdir() if p.is_dir()):
        # The existing legacy README is retained and linked, never overwritten here.
        name = "00_说明_20260919.md" if (directory / "README.md").exists() else "README.md"
        body = "\n".join(f"- [{p.name}]({p.name})" for p in sorted(directory.iterdir()) if p.is_file() and p.name != name)
        save_updated(directory / name, f"# {directory.name}\n\n保持同批依赖文件相邻，原脚本及记录不修改、不执行。历史部署路径及平铺scripts布局仅适用于原版本，重用前必须制作新计划并预检。\n\n{body}\n" + footer)
    save_updated(LEGACY / "README.md", "# S1/S2 历史原件\n\n01_legacy_source：备份和旧源码；02_placeholders：空模板；03_before_path_updates：路径适配及导航更新前原件；04_previous_navigation：旧目录导航；05_original_directory_objects：腾空后保留的原目录对象。\n\n"
        "不删除、不去重覆盖。历史版本不作为当前入口。搬迁和内容校验见同级s1_s2_reorganization_20260919_r3。\n" + footer)
    print(json.dumps({"phase": "navigation", "status": "written"}), flush=True)


def run_tests(label):
    import unittest
    repo = S2 / "01_code/repository"
    sys.path.insert(0, str(repo))
    suite = unittest.defaultTestLoader.discover(str(repo / "tests"))
    def flatten(tests):
        for test in tests:
            if isinstance(test, unittest.TestSuite):
                yield from flatten(test)
            else:
                yield test
    omitted, selected = [], []
    for test in flatten(suite):
        if test.id().startswith("test_package_official_submission."):
            omitted.append(test.id())
        else:
            selected.append(test)
    with (AUDIT / f"TESTS_{label}.log").open("x", encoding="utf-8") as log:
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(unittest.TestSuite(selected))
    output = {**STATUS, "utc": timestamp(), "python": sys.executable, "tests_run": result.testsRun,
              "successful": result.wasSuccessful(), "failures": [t.id() for t, _ in result.failures],
              "errors": [t.id() for t, _ in result.errors], "skipped": [(t.id(), reason) for t, reason in result.skipped],
              "omitted_zip_packaging_tests": omitted}
    write_json(AUDIT / f"TESTS_{label}.json", output)
    print(json.dumps(output), flush=True)


def verify(label):
    from urllib.parse import unquote
    import unittest
    moves = {m["old"]: m["new"] for m in json.loads((AUDIT / "PLAN.json").read_text())["moves"]}
    adaptations = [json.loads(line) for line in (AUDIT / "ADAPTATIONS.jsonl").read_text().splitlines()]
    backups = {item["active"]: item["backup"] for item in adaptations}
    errors = []

    def preserved(path):
        current = moves.get(path, path)
        return backups.get(current, current)

    def check(item, path):
        target = ROOT / path
        if not target.is_file() or target.is_symlink():
            errors.append({"path": path, "error": "missing regular original"})
            return
        actual = record(target)
        expected = {"bytes": int(item["bytes"]), "sha256": item["sha256"], "inode": int(item["inode"]),
                    "mode": int(item["mode"]), "mtime_ns": int(item["mtime_ns"]), "uid": int(item["uid"]), "gid": int(item["gid"])}
        bad = {key: {"expected": value, "actual": actual[key]} for key, value in expected.items() if actual[key] != value}
        if bad:
            errors.append({"path": path, "differences": bad})

    baseline = json.loads((AUDIT / "BASELINE.json").read_text())["files"]
    for item in baseline:
        check(item, preserved(item["path"]))
    for item in adaptations:
        check(item["original"], item["backup"])
    with (ROOT / "99_archive/reorganization_20260919_r2/ORIGINAL_LOCATIONS_FINAL.csv").open(encoding="utf-8-sig", newline="") as handle:
        originals = list(csv.DictReader(handle))
    working_originals = set()
    final_locations = []
    for index, item in enumerate(originals, 1):
        location = preserved(item["preserved_original"])
        check(item, location)
        working_originals.add(moves.get(item["new"], item["new"]))
        final_locations.append({**item, "current_working_path": moves.get(item["new"], item["new"]),
                                "current_preserved_original": location})
        if index % 5000 == 0:
            print(json.dumps({"phase": "verify", "originals": index, "errors": len(errors)}), flush=True)
    with (AUDIT / f"ORIGINAL_LOCATIONS_{label}.csv").open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(final_locations[0]))
        writer.writeheader()
        writer.writerows(final_locations)

    empty, links, syntax_errors = [], [], []
    generated_docs = []
    for directory, dirs, files in os.walk(ROOT, followlinks=False):
        path = Path(directory)
        dirs[:] = [name for name in dirs if name not in {".git", ".claude"}]
        if not dirs and not files:
            empty.append(relative(path))
        if "99_archive" not in path.relative_to(ROOT).parts:
            for name in dirs + files:
                candidate = path / name
                if candidate.is_symlink():
                    links.append(relative(candidate))
            for name in files:
                candidate = path / name
                rel = relative(candidate)
                if candidate.suffix == ".md" and (rel not in working_originals or rel in backups):
                    generated_docs.append(candidate)
    broken = []
    link_count = 0
    for path in generated_docs:
        for match in re.finditer(r"\[[^\]]*\]\((<[^>]+>|[^)\n]+)\)", path.read_text(encoding="utf-8")):
            target = match.group(1).strip().strip("<>")
            if target.startswith(("http:", "https:", "mailto:", "#")):
                continue
            target = unquote(target.split("#", 1)[0])
            if not target:
                continue
            link_count += 1
            if not (path.parent / target).exists():
                broken.append({"document": relative(path), "target": target})
    for module in (S1, S2):
        for path in (module / "01_code/repository").rglob("*.py"):
            try:
                ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError as exc:
                syntax_errors.append({"path": relative(path), "error": str(exc)})
    tools_suite = unittest.defaultTestLoader.discover(str(ROOT / "tools"), pattern="test_*.py")
    with (AUDIT / f"LAYOUT_TESTS_{label}.log").open("x", encoding="utf-8") as log:
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(tools_suite)
    training = json.loads((AUDIT / "TESTS_after.json").read_text())
    output = {**STATUS, "utc": timestamp(), "local_time": datetime.now().astimezone().isoformat(),
              "original_files_verified": len(originals), "original_bytes_verified": sum(int(r["bytes"]) for r in originals),
              "this_round_baseline_files_verified": len(baseline), "moved_files": len(moves),
              "adapted_files_with_originals_preserved": len(adaptations), "original_errors": errors,
              "empty_directories": empty, "active_symlinks": links, "syntax_errors": syntax_errors,
              "new_navigation_links_checked": link_count, "broken_navigation_links": broken,
              "layout_tests_run": result.testsRun, "layout_tests_passed": result.wasSuccessful(),
              "s2_tests_run": training["tests_run"], "s2_tests_passed": training["successful"],
              "excluded_tests": training["omitted_zip_packaging_tests"],
              "status": "pass" if not (errors or empty or links or syntax_errors or broken) and result.wasSuccessful() and training["successful"] else "fail"}
    write_json(AUDIT / f"VERIFICATION_{label}.json", output)
    print(json.dumps(output, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["plan", "move", "adapt", "empty_dirs", "navigation", "protect_tools", "tests_before", "tests_after", "verify"])
    parser.add_argument("--label", default="final")
    args = parser.parse_args()
    action = args.action
    if action == "plan":
        plan()
    elif action == "move":
        execute_moves()
    elif action == "adapt":
        adapt()
    elif action == "empty_dirs":
        preserve_empty_directories()
    elif action == "navigation":
        navigation()
    elif action == "protect_tools":
        for name in ("tools/project_paths.py", "tools/test_project_paths.py", "README.md"):
            protect_for_manual_edit(ROOT / name)
    elif action == "verify":
        verify(args.label)
    else:
        run_tests(action.removeprefix("tests_"))
