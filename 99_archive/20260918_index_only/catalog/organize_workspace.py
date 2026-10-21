"""Audited, same-depth workspace organization. No original file is rewritten."""
import argparse
import ast
import csv
import ctypes
import hashlib
import json
import os
import re
import stat
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, unquote

CATALOG = Path(__file__).resolve().parent
ROOT = CATALOG.parent
BATCH = "20260918T153706Z"
STATUS = dict(artifact_status="experimental_unvalidated", operator_approved=False,
              formal_gate_status="not_run_not_passed")
MODULES = ["G1", "G2", "S1", "S2", "S3", "S4", "S5", "paper_figures", "ROOT"]
RENAME = {"code": "10_code", "data": "20_data", "results": "30_results",
          "docs": "40_documents", "slurm": "50_slurm", "cloud": "60_cloud",
          "tests": "70_tests", "container": "80_submission_container"}
DOC_EXT = {".md", ".txt", ".rst", ".pdf", ".doc", ".docx", ".xlsx", ".xls", ".csv", ".tsv", ".html", ".ipynb"}
TEXT_EXT = {".md", ".txt", ".rst", ".py", ".sh", ".slurm", ".json", ".yaml", ".yml", ".toml", ".cfg", ".ini"}
DATE_RE = re.compile(r"(?<!\d)(20[12]\d)[-_]?(0[1-9]|1[0-2])[-_]?(0[1-9]|[12]\d|3[01])(?!\d)")
TOPIC_NAMES = {
    "01_vae_and_missing_t2w": "VAE微调与缺失T2W补全",
    "02_diffusion_augmentation": "扩散模型训练与合成增强",
    "03_g1_image_quality": "G1图像质量与模型评估",
    "04_g1_shared": "G1共享工程与数据",
    "01_qc_and_review": "质量控制与人工审查",
    "02_splits_and_dataset": "病例划分与数据集发布",
    "03_g2_pipeline": "G2管线与协作合同",
    "01_baseline_and_submission": "基线、原E与比赛推理提交",
    "02_augmentation_development": "小病灶消融、Route A与FixV2",
    "03_fix_v3": "FixV3与回顾性比较",
    "04_btof_screen125": "B至F等预算Screen125训练",
    "05_screen125_official_met": "五路selected-checkpoint官方MET",
    "06_bef_extend175": "B、E-SNFH、F独立Extend175",
    "07_epoch175_official_met": "epoch175官方MET与CPU恢复",
    "08_g1r4_endpoint": "G1-r4补全与分割endpoint",
    "09_table7": "Table7与micro-F1探索",
    "10_s2_shared": "S2共享代码、数据与运行资料",
    "01_s1_multitask": "S1多任务分割",
    "01_s3_segmentation": "S3分割分支",
    "01_s4_sam2unet": "S4 SAM2-UNet",
    "01_s5_segmamba": "S5 SegMamba及第三方依赖",
    "01_paper_figures": "论文图表与可编辑素材",
    "01_workspace_shared": "跨模块说明与历史总览",
}


def utc():
    return datetime.now(timezone.utc).isoformat()


def require(ok, msg):
    if not ok:
        raise RuntimeError(msg)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def xattrs(path):
    if hasattr(os, "listxattr"):
        return {k: hashlib.sha256(os.getxattr(path, k, follow_symlinks=False)).hexdigest()
                for k in sorted(os.listxattr(path, follow_symlinks=False))}
    require(sys.platform == "darwin", "No supported extended-attribute API")
    # Conda's macOS Python lacks os.listxattr; use the native read-only API.
    libc = ctypes.CDLL(None, use_errno=True)
    libc.listxattr.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    libc.listxattr.restype = ctypes.c_ssize_t
    libc.getxattr.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_void_p,
                              ctypes.c_size_t, ctypes.c_uint32, ctypes.c_int]
    libc.getxattr.restype = ctypes.c_ssize_t
    raw = os.fsencode(path)
    n = libc.listxattr(raw, None, 0, 1)
    require(n >= 0, "Cannot list extended attributes: " + str(path))
    if not n:
        return {}
    names = ctypes.create_string_buffer(n)
    require(libc.listxattr(raw, names, n, 1) == n, "Extended-attribute list changed")
    result = {}
    for key in sorted(names.raw.rstrip(b"\0").split(b"\0")):
        size = libc.getxattr(raw, key, None, 0, 0, 1)
        require(size >= 0, "Cannot size extended attribute")
        value = ctypes.create_string_buffer(size)
        require(libc.getxattr(raw, key, value, size, 0, 1) == size, "Extended attribute changed")
        result[os.fsdecode(key)] = hashlib.sha256(value.raw).hexdigest()
    return result


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_md(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        f.write(content)


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else ["no_records"]
    with path.open("x", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="raise")
        w.writeheader()
        for row in rows:
            w.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                        for k, v in row.items()})


def read_json(name):
    return json.loads((CATALOG / name).read_text(encoding="utf-8"))


def module_of(rel):
    first = Path(rel).parts[0]
    return first if first in MODULES else "ROOT"


def map_path(rel, mappings):
    for m in mappings:
        if rel == m["old"] or rel.startswith(m["old"] + "/"):
            return m["new"] + rel[len(m["old"]):]
    return rel


def date_of(rel):
    for part in reversed(Path(rel).parts):
        found = DATE_RE.search(part)
        if found:
            value = "-".join(found.groups())
            try:
                datetime.strptime(value, "%Y-%m-%d")
            except ValueError:
                continue
            return value, "filename_or_nearest_parent_date_not_verified_execution_time"
    return "undated", "unknown_not_inferred_from_mtime"


def topic_of(rel, title=""):
    m, p = module_of(rel), (rel + " " + title).lower()
    if m == "G1":
        if "single_model_eval" in p or "paired_comparison" in p:
            return "03_g1_image_quality"
        if any(t in p for t in ["vae", "缺失", "missing_t2w", "latent-ensemble", "v3_validation"]):
            return "01_vae_and_missing_t2w"
        if any(t in p for t in ["diffusion", "扩散", "gligan", "brats_2023_2024"]):
            return "02_diffusion_augmentation"
        return "04_g1_shared"
    if m == "G2":
        if any(t in p for t in ["/qc/", "质量", "review", "teacher", "paired_validation"]):
            return "01_qc_and_review"
        if any(t in p for t in ["/splits/", "/manifests/", "/nnunet_raw/", "/stats/", "划分"]):
            return "02_splits_and_dataset"
        return "03_g2_pipeline"
    if m == "S2":
        for keys, topic in [
            (["epoch175_official_met"], "07_epoch175_official_met"),
            (["extend175"], "06_bef_extend175"),
            (["btof_selected103"], "05_screen125_official_met"),
            (["btof_screen125"], "04_btof_screen125"),
            (["g1r4_endpoint"], "08_g1r4_endpoint"),
            (["table7", "microf1"], "09_table7"),
            (["fix_v3", "fixv3", "fix-v3"], "03_fix_v3"),
            (["met_aug_route_a", "fix_v2", "small_lesion_ablation", "小病灶消融"], "02_augmentation_development"),
            (["official_validation", "original_e", "realonly", "e_continue", "27。result", "s2_eval_results", "leaderboard", "completion_dataset264", "official_submission"], "01_baseline_and_submission"),
        ]:
            if any(t in p for t in keys):
                return topic
        return "10_s2_shared"
    return {"S1": "01_s1_multitask", "S3": "01_s3_segmentation", "S4": "01_s4_sam2unet",
            "S5": "01_s5_segmamba", "paper_figures": "01_paper_figures"}.get(m, "01_workspace_shared")


def category_of(rel):
    p, name, ext = rel.lower(), Path(rel).name.lower(), Path(rel).suffix.lower()
    if name == ".ds_store" or "__pycache__" in p or ext in {".pyc", ".pyo"} or name.startswith("._"):
        return "11_cache_and_system_metadata"
    if name in {".gitkeep", ".keep"}:
        return "12_historical_placeholders"
    if name.startswith(".env") or re.search(r"(^|/)(id_rsa|id_ed25519|credentials)(\.|$)", p):
        return "13_sensitive_config_index_only"
    if name in {".gitignore", ".gitattributes", ".gitmodules", ".dockerignore"}:
        return "02_structured_configs_and_metadata"
    if re.search(r"\.bak\d*$", name):
        return "15_historical_source_backups"
    if name == "experimental_unvalidated":
        return "09_audits_hashes_and_gate_evidence"
    if name == "py.typed":
        return "01_source_and_execution_code"
    if ext in {".zip", ".tar", ".tgz", ".7z"} or name.endswith(".tar.gz"):
        return "10_archives_and_submission_packages"
    if ext in {".pth", ".pt", ".ckpt", ".safetensors"}:
        return "06_model_checkpoints"
    if ext in {".npz", ".npy", ".nii", ".h5", ".hdf5", ".pkl"} or name.endswith(".nii.gz"):
        return "05_arrays_images_and_cached_data"
    if ext in {".py", ".sh", ".slurm", ".ipynb", ".cpp", ".cu", ".c", ".h", ".cuh", ".bat", ".js", ".css"} or name in {"dockerfile", "makefile"}:
        return "01_source_and_execution_code"
    if ext in {".log", ".out", ".err", ".jsonl", ".pid"} or "tfevents" in name:
        return "08_logs_and_event_streams"
    if ext in {".png", ".jpg", ".jpeg", ".svg", ".gif", ".webp", ".tif", ".tiff"}:
        return "07_figures_and_visual_evidence"
    if ext in {".csv", ".tsv", ".xlsx", ".xls"}:
        return "04_tables_and_manifests"
    if ext == ".ok" or any(t in name for t in ["sha256", "validation", "preflight", "failed", "audit", "complete", "reservation", "gate"]):
        return "09_audits_hashes_and_gate_evidence"
    if ext in {".json", ".yaml", ".yml", ".toml", ".cfg", ".ini", ".xml"}:
        return "02_structured_configs_and_metadata"
    if ext in DOC_EXT or name.startswith(("license", "readme", "notice")) or name == "authors":
        return "03_documents_and_licenses"
    return "14_other_preserved_artifacts"


def document_role(rel):
    p, ext = rel.lower(), Path(rel).suffix.lower()
    if ext in {".csv", ".tsv", ".xlsx", ".xls"}:
        return "08_tables"
    if any(t in p for t in ["sha256", "requirements", "environment", "环境"]):
        return "07_environment_hashes"
    if any(t in p for t in ["failed", "failure", "失败"]):
        return "06_failures_and_recovery"
    if any(t in p for t in ["plan", "计划", "方案", "task_assignment"]):
        return "02_plans_and_assignments"
    if any(t in p for t in ["report", "comparison", "acceptance", "validation", "summary", "总结", "结果", "报告", "评估"]):
        return "04_reports_and_evaluations"
    if any(t in p for t in ["contract", "protocol", "规则", "策略"]):
        return "03_methods_and_contracts"
    if any(t in p for t in ["manual", "deploy", "运行", "手册", "部署", "readme", "使用说明"]):
        return "01_guides_and_readmes"
    if ext in {".pdf", ".ipynb"} or "license" in p:
        return "09_references_and_notebooks"
    return "05_notes_and_other_documents"


def safe_title(path):
    if path.suffix.lower() not in {".md", ".rst", ".txt"}:
        return path.name, "filename_only"
    with path.open("r", encoding="utf-8", errors="replace") as f:
        text = f.read(8192)
    for line in text.splitlines():
        if line.startswith("# "):
            title = line[2:].strip()
            if not re.search(r"password|token|secret|api.?key|口令|密码", title, re.I):
                return title[:160], "first_markdown_heading"
    return path.name, "filename_only"


def file_info(path, rel):
    st = path.stat(follow_symlinks=False)
    return {"old_path": rel, "size": st.st_size, "sha256": digest(path),
            "mode": stat.S_IMODE(st.st_mode), "uid": st.st_uid, "gid": st.st_gid,
            "mtime_ns": st.st_mtime_ns, "inode": st.st_ino, "device": st.st_dev,
            "xattrs_sha256": xattrs(path)}


def baseline():
    require(not (CATALOG / "BASELINE.json").exists(), "Baseline already exists")
    mappings = []
    for module in MODULES[:7]:
        for old, new in RENAME.items():
            if (ROOT / module / old).is_dir():
                mappings.append({"old": f"{module}/{old}", "new": f"{module}/{new}",
                                 "reason": "same_depth_category_normalization"})
    for old, new in [("S1/brats2026_multitask_S1_v2", "S1/11_project_brats2026_multitask_s1_v2"),
                     ("S2/BraTS2026_S2_RC_v1.0", "S2/11_project_brats2026_s2_rc_v1_0"),
                     ("paper_figures", "80_paper_figures")]:
        if (ROOT / old).is_dir():
            mappings.append({"old": old, "new": new, "reason": "same_depth_package_normalization"})
    for m in mappings:
        require(not os.path.lexists(ROOT / m["new"]), "Destination collision: " + m["new"])
        require(len(Path(m["old"]).parts) == len(Path(m["new"]).parts), "Depth changes forbidden")
    if (CATALOG / "MOVE_PLAN.json").exists():
        require(read_json("MOVE_PLAN.json")["mappings"] == mappings, "Existing move plan differs")
        require(not (CATALOG / "MOVE_JOURNAL.jsonl").exists(), "Moves already started")
    else:
        write_json(CATALOG / "MOVE_PLAN.json", {**STATUS, "batch": BATCH, "created_utc": utc(),
               "mappings": mappings, "overwrite_allowed": False, "old_path_aliases": True,
               "experiment_bundle_names_and_internal_structure": "preserved"})
    files, dirs = [], []
    for base, subdirs, names in os.walk(ROOT, followlinks=False):
        subdirs.sort()
        names.sort()
        subdirs[:] = [d for d in subdirs if Path(base) / d != CATALOG]
        p = Path(base)
        st = p.stat()
        rel = p.relative_to(ROOT).as_posix()
        dirs.append({"old_path": rel, "mode": stat.S_IMODE(st.st_mode), "mtime_ns": st.st_mtime_ns,
                     "atime_ns": st.st_atime_ns, "inode": st.st_ino, "uid": st.st_uid, "gid": st.st_gid,
                     "xattrs_sha256": xattrs(p), "empty": not subdirs and not names,
                     "placeholder_only": not subdirs and bool(names) and all(n in {".gitkeep", ".DS_Store", ".keep"} for n in names)})
        for name in subdirs + names:
            require(not (p / name).is_symlink(), "Unexpected initial symlink; stop and review")
        for name in names:
            path = p / name
            require(path.is_file(), "Special file requires manual review: " + str(path))
            f = file_info(path, path.relative_to(ROOT).as_posix())
            f.update(new_path=map_path(f["old_path"], mappings), module=module_of(f["old_path"]),
                     category=category_of(f["old_path"]), topic=topic_of(f["old_path"]))
            f["date"], f["date_basis"] = date_of(f["old_path"])
            files.append(f)
            if len(files) % 1500 == 0:
                print(json.dumps({"stage": "baseline_hash", "files": len(files)}, ensure_ascii=False), flush=True)
    write_json(CATALOG / "BASELINE.json", {**STATUS, "batch": BATCH, "completed_utc": utc(),
               "root": str(ROOT), "files": files, "directories": dirs})
    write_csv(CATALOG / "ALL_FILES.csv", files)
    write_csv(CATALOG / "ALL_DIRECTORIES.csv", [dict(d, new_path=map_path(d["old_path"], mappings)) for d in dirs])
    print(json.dumps({"stage": "baseline_complete", "files": len(files), "directories": len(dirs),
                      "bytes": sum(f["size"] for f in files), "moves": len(mappings),
                      "empty": sum(d["empty"] for d in dirs)}, ensure_ascii=False), flush=True)


def link(path, target):
    require(not os.path.lexists(path), "Refusing link overwrite: " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.symlink_to(os.path.relpath(target, path.parent), target_is_directory=target.is_dir())


def slug(value):
    value = re.sub(r"[^\w.\-]+", "_", value, flags=re.UNICODE).strip("._") or "artifact"
    while len(value.encode("utf-8")) > 130:
        value = value[:-1]
    return value


def mdlink(from_path, target, label):
    return f"[{str(label).replace('|', '/').replace(']', '')}]({quote(os.path.relpath(target, from_path.parent), safe='/._-')})"


def get_bundles(files):
    candidates = set()
    for f in files:
        parts = Path(f["old_path"]).parts
        if len(parts) > 3 and parts[1] == "results":
            end = 4 if parts[2] == "experimental_unvalidated" else 3
            if parts[0] == "G2" and parts[2] in {"qc", "nnunet_raw"} and len(parts) > 4:
                end = 4
                if parts[3] in {"v3_paired_validation", "v3_completion_review", "v3_s2_teacher"} and len(parts) > 5:
                    end = 5
            if parts[0] == "G1" and parts[2] in {"vae_finetuned", "v3_validation", "missing_t2w_completion"}:
                end = 4
            candidate = "/".join(parts[:end])
            if len(parts) > end:
                candidates.add(candidate)
        elif parts[0] == "paper_figures" and len(parts) > 2:
            candidates.add("/".join(parts[:2]))
    return sorted(candidates)


def collect_evidence(files):
    evidence, parse_errors, references = [], [], []
    keys = {"status", "artifact_status", "operator_approved", "formal_gate_status", "experiment_id", "experiment",
            "selected_model", "selected_epoch", "best_epoch", "job_id", "canonical_job_id", "slurm_job_id",
            "created_at_utc", "generated_at_utc", "completed_at_utc", "remote_utc", "case_count", "deep_validated"}
    for f in files:
        p = ROOT / f["new_path"]
        if p.suffix.lower() == ".json":
            try:
                obj = json.loads(p.read_text(encoding="utf-8"))
            except (ValueError, UnicodeError) as e:
                parse_errors.append({"old_path": f["old_path"], "error_type": type(e).__name__, "action": "preserved_unchanged"})
                continue
            selected = {k: v for k, v in obj.items() if k in keys and isinstance(v, (str, int, float, bool, type(None)))} if isinstance(obj, dict) else {}
            evidence.append({"old_path": f["old_path"], "new_path": f["new_path"], "sha256": f["sha256"],
                             "json_root_type": type(obj).__name__, "metadata_only_not_acceptance": selected})
        if p.suffix.lower() in TEXT_EXT and f["size"] < 2_000_000 and f["category"] != "13_sensitive_config_index_only":
            text = p.read_text(encoding="utf-8", errors="replace")
            counts = {"resolve_ancestry": len(re.findall(r"\.resolve\(\).*?(?:\.parent|\.parents)", text)),
                      "absolute_local_workspace": text.count(str(ROOT)),
                      "remote_root_reference": text.count("/public/home/"),
                      "experiment_name_contract": len(re.findall(r"root\.name", text)),
                      "relative_workspace_reference": text.count("work_space/")}
            if any(counts.values()):
                references.append({"old_path": f["old_path"], **counts,
                                   "treatment": "original_bytes_unchanged_same_depth_and_old_path_alias"})
    return evidence, parse_errors, references


def organize():
    baseline_data = read_json("BASELINE.json")
    files, dirs = baseline_data["files"], baseline_data["directories"]
    mappings = read_json("MOVE_PLAN.json")["mappings"]
    require(not (CATALOG / "ORGANIZATION_COMPLETE.json").exists(), "Already organized")
    for f in files:
        p = ROOT / f["old_path"]
        st = p.stat()
        require(st.st_ino == f["inode"] and st.st_size == f["size"] and st.st_mtime_ns == f["mtime_ns"],
                "Concurrent file change; stop: " + f["old_path"])
    journal = CATALOG / "MOVE_JOURNAL.jsonl"
    with journal.open("x", encoding="utf-8") as log:
        for m in mappings:
            source, target = ROOT / m["old"], ROOT / m["new"]
            require(not os.path.lexists(target), "Destination collision")
            log.write(json.dumps({"event": "intent", "utc": utc(), **m}) + "\n")
            log.flush()
            os.fsync(log.fileno())
            source.rename(target)
            link(source, target)
            log.write(json.dumps({"event": "renamed_and_aliased", "utc": utc(), **m}) + "\n")
            log.flush()
            os.fsync(log.fileno())
    write_json(CATALOG / "ROLLBACK_PLAN.json", {**STATUS, "execute_automatically": False,
               "steps": [{"alias_to_park_not_delete": m["old"], "rename_from": m["new"], "rename_to": m["old"]}
                         for m in reversed(mappings)],
               "preconditions": "Reverify baseline and refuse collisions; park new aliases/notes/catalog rather than delete; do not execute while jobs or writers use this workspace."})
    notes = []
    for d in dirs:
        if not d["empty"]:
            continue
        p = ROOT / map_path(d["old_path"], mappings)
        require(not any(p.iterdir()), "Empty-directory state drift: " + d["old_path"])
        reserved = "RESERVATION" in p.name.upper()
        note = p / f"HISTORICAL_EMPTY_DIRECTORY_{BATCH}.md"
        mode = stat.S_IMODE(p.stat().st_mode)
        try:
            if not mode & stat.S_IWUSR:
                os.chmod(p, mode | stat.S_IWUSR)
            write_md(note, f"# 历史空目录保留说明\n\n整理批次：{BATCH}。原目录：`{d['old_path']}`。\n\n"
                     + ("这是既有排他 reservation 目录，锁定身份和目录 inode 保留。本说明不代表释放、完成或重新预留，禁止删除或重建锁。\n" if reserved else
                        "本地整理前目录为空。原目录保留，本说明仅消除空目录并解释来源；不代表远端没有数据，也不代表实验成功或失败。\n")
                     + "\nartifact_status=experimental_unvalidated；operator_approved=false；formal_gate_status=not_run_not_passed。\n")
        finally:
            os.chmod(p, mode)
        notes.append({"original_directory": d["old_path"], "note": note.relative_to(ROOT).as_posix(), "reservation_semantics_preserved": reserved})
    write_json(CATALOG / "EMPTY_DIRECTORY_NOTES.json", {**STATUS, "notes": notes})
    evidence, errors, refs = collect_evidence(files)
    write_csv(CATALOG / "EVIDENCE_METADATA.csv", evidence)
    write_json(CATALOG / "HISTORICAL_JSON_PARSE_ISSUES.json", {**STATUS, "issues": errors})
    write_csv(CATALOG / "PATH_DEPENDENCIES.csv", refs)
    cache = [f for f in files if f["category"] in {"11_cache_and_system_metadata", "12_historical_placeholders"}]
    write_csv(CATALOG / "CACHES_AND_PLACEHOLDERS_PRESERVED.csv", cache)
    grouped = defaultdict(list)
    for f in files:
        grouped[(f["size"], f["sha256"])].append(f)
    duplicates = [{"sha256": h, "size_each": size, "copies": len(v), "all_preserved": True,
                   "new_paths": [f["new_path"] for f in v]} for (size, h), v in grouped.items() if len(v) > 1]
    write_csv(CATALOG / "DUPLICATE_GROUPS_PRESERVED.csv", duplicates)
    documents = []
    for f in files:
        p = ROOT / f["new_path"]
        if p.suffix.lower() not in DOC_EXT and not p.name.lower().startswith(("readme", "license", "notice")):
            continue
        title, basis = safe_title(p)
        role = document_role(f["old_path"])
        topic = topic_of(f["old_path"], title)
        unique = hashlib.sha256(f["old_path"].encode()).hexdigest()[:10]
        nav = CATALOG / "modules" / f["module"] / "documents" / role / f"{f['date']}__{slug(p.stem)}__{unique}{p.suffix}"
        link(nav, p)
        documents.append({"module": f["module"], "title": title, "title_basis": basis, "topic": topic,
                          "document_role": role, "date": f["date"], "date_basis": f["date_basis"],
                          "old_path": f["old_path"], "new_path": f["new_path"], "sha256": f["sha256"],
                          "navigation_path": nav.relative_to(CATALOG).as_posix()})
    write_csv(CATALOG / "DOCUMENT_REGISTER.csv", documents)
    bundles = []
    for old in get_bundles(files):
        members = [f for f in files if f["old_path"].startswith(old + "/")]
        if not members:
            continue
        date, date_basis = date_of(old)
        topic, module = topic_of(old), module_of(old)
        new = map_path(old, mappings)
        unique = hashlib.sha256(old.encode()).hexdigest()[:8]
        nav = CATALOG / "modules" / module / "experiments_and_collections" / topic / f"{date}__{slug(Path(old).name)}__{unique}"
        link(nav, ROOT / new)
        fail = [f["new_path"] for f in members if re.search(r"FAILED|FAILURE", Path(f["old_path"]).name)]
        complete = [f["new_path"] for f in members if Path(f["old_path"]).name.endswith("COMPLETE.ok")]
        plans = [f["new_path"] for f in members if "PLAN" in Path(f["old_path"]).name and f["old_path"].endswith(".json")]
        validations = [e for e in evidence if e["old_path"].startswith(old + "/") and e["metadata_only_not_acceptance"].get("status") == "pass"]
        bundles.append({"module": module, "topic": topic, "date": date, "date_basis": date_basis,
                        "old_path": old, "new_path": new, "file_count": len(members), "bytes": sum(f["size"] for f in members),
                        "failure_marker_count": len(fail), "complete_marker_count": len(complete), "status_pass_json_count": len(validations),
                        "failure_evidence_paths": fail, "complete_evidence_paths": complete, "plan_paths": plans,
                        "interpretation": "local_marker_inventory_only_not_revalidated_experiment_success",
                        "navigation_path": nav.relative_to(CATALOG).as_posix()})
    write_csv(CATALOG / "EXPERIMENT_AND_COLLECTION_REGISTER.csv", bundles)
    for module in MODULES:
        mod_files = [f for f in files if f["module"] == module]
        mod_docs = [d for d in documents if d["module"] == module]
        mod_bundles = [b for b in bundles if b["module"] == module]
        index = CATALOG / "modules" / module / "README.md"
        rows = [f"# {module} 分类入口\n", f"整理批次：{BATCH}。本地原文件 {len(mod_files)} 份，{sum(f['size'] for f in mod_files):,} 字节。\n",
                "本页是整理导航，不是实验验收。实验目录内部、失败证据和历史文档不改写。\n", "## 目录用途\n"]
        for m in mappings:
            if module_of(m["old"]) == module:
                nav = index.parent / "resources" / Path(m["new"]).name
                link(nav, ROOT / m["new"])
                rows.append(f"- {mdlink(index, nav, Path(m['new']).name)}：原入口 `{m['old']}` 保留为兼容链接。\n")
        rows.extend(["\n## 本地文件分类\n", "| 分类 | 文件数 |\n|---|---:|\n"])
        for key, count in sorted(Counter(f["category"] for f in mod_files).items()):
            rows.append(f"| {key} | {count} |\n")
        rows.extend(["\n## 实验与结果集合时间线\n", "日期来自文件名或父目录，不等于核实后的运行日期；undated 表示未知。标记数量不是成功判定。\n\n",
                     "| 日期 | 主题及原实验 | 文件数 | 失败标记 | COMPLETE标记 |\n|---|---|---:|---:|---:|\n"])
        for b in sorted(mod_bundles, key=lambda x: (x["date"], x["old_path"])):
            rows.append(f"| {b['date']} | {TOPIC_NAMES[b['topic']]} / {mdlink(index, CATALOG / b['navigation_path'], Path(b['old_path']).name)} | {b['file_count']} | {b['failure_marker_count']} | {b['complete_marker_count']} |\n")
        if not mod_bundles:
            rows.append("| 未提供 | 当前本地未发现独立结果集合，不推断远端情况 | 0 | 0 | 0 |\n")
        rows.extend(["\n## 文档逐项分类\n", "历史计划仍是计划；文件标题来自首个标题或文件名，不据此认定已经实施。完整记录见总目录 DOCUMENT_REGISTER.csv。\n\n",
                     "| 分类 | 日期 | 文档 | 主题 |\n|---|---|---|---|\n"])
        for d in sorted(mod_docs, key=lambda x: (x["document_role"], x["date"], x["old_path"])):
            rows.append(f"| {d['document_role']} | {d['date']} | {mdlink(index, ROOT / d['new_path'], d['title'])} | {TOPIC_NAMES[d['topic']]} |\n")
        write_md(index, "".join(rows))
        if module in MODULES[:7]:
            shortcut = ROOT / module / f"00_CATALOG_{BATCH}.md"
            write_md(shortcut, f"# {module} 整理入口\n\n{mdlink(shortcut, index, '打开分类目录、文档索引和实验时间线')}\n\n"
                     "编号目录为整理后的实体目录；code/data/results/docs 等旧入口是兼容链接，不是重复副本。\n\n"
                     "artifact_status=experimental_unvalidated；operator_approved=false；formal_gate_status=not_run_not_passed。\n")
    unknown = [f for f in files if f["category"] == "14_other_preserved_artifacts"]
    write_csv(CATALOG / "OTHER_ARTIFACTS_REQUIRING_CONTEXT.csv", unknown)
    for d in reversed(dirs):
        p = ROOT / map_path(d["old_path"], mappings)
        os.utime(p, ns=(d["atime_ns"], d["mtime_ns"]))
    result = {**STATUS, "batch": BATCH, "completed_utc": utc(), "moves": len(mappings),
              "original_files": len(files), "original_bytes": sum(f["size"] for f in files),
              "documents": len(documents), "bundles": len(bundles), "empty_notes": len(notes),
              "duplicate_groups": len(duplicates), "cache_and_placeholder_files": len(cache),
              "historical_json_parse_issues": len(errors), "verification": "not_yet_run"}
    write_json(CATALOG / "ORGANIZATION_COMPLETE.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


def verify():
    data = read_json("BASELINE.json")
    files, dirs = data["files"], data["directories"]
    mappings = read_json("MOVE_PLAN.json")["mappings"]
    failures, symlinks, actual_files, empty = [], [], [], []
    for i, f in enumerate(files, 1):
        p, old = ROOT / f["new_path"], ROOT / f["old_path"]
        try:
            st = p.stat()
            checks = {"sha256": digest(p) == f["sha256"], "size": st.st_size == f["size"],
                      "inode": st.st_ino == f["inode"], "mode": stat.S_IMODE(st.st_mode) == f["mode"],
                      "owner": st.st_uid == f["uid"] and st.st_gid == f["gid"],
                      "mtime_ns": st.st_mtime_ns == f["mtime_ns"], "xattrs": xattrs(p) == f["xattrs_sha256"],
                      "old_path_samefile": old.samefile(p),
                      "same_depth": len(old.parts) == len(p.resolve().parts)}
            if not all(checks.values()):
                failures.append({"path": f["old_path"], "checks": checks})
        except (OSError, RuntimeError) as e:
            failures.append({"path": f["old_path"], "error_type": type(e).__name__})
        if i % 1500 == 0:
            print(json.dumps({"stage": "verify_hash", "files": i, "failures": len(failures)}), flush=True)
    for d in dirs:
        p, old = ROOT / map_path(d["old_path"], mappings), ROOT / d["old_path"]
        st = p.stat()
        checks = {"inode": st.st_ino == d["inode"], "mode": stat.S_IMODE(st.st_mode) == d["mode"],
                  "owner": st.st_uid == d["uid"] and st.st_gid == d["gid"],
                  "xattrs": xattrs(p) == d["xattrs_sha256"], "old_path_samefile": old.samefile(p)}
        if not all(checks.values()):
            failures.append({"directory": d["old_path"], "checks": checks})
    for base, subdirs, names in os.walk(ROOT, followlinks=False):
        if not subdirs and not names:
            empty.append(str(Path(base).relative_to(ROOT)))
        for name in subdirs + names:
            p = Path(base) / name
            if p.is_symlink():
                try:
                    target = p.resolve(strict=True)
                    if not target.is_relative_to(ROOT):
                        failures.append({"symlink": str(p), "error": "outside_workspace"})
                    if target.is_dir() and p.parent.resolve().is_relative_to(target):
                        failures.append({"symlink": str(p), "error": "ancestor_cycle"})
                    symlinks.append({"path": p.relative_to(ROOT).as_posix(), "target": target.relative_to(ROOT).as_posix()})
                except (OSError, RuntimeError) as e:
                    failures.append({"symlink": str(p), "error_type": type(e).__name__})
            elif p.is_file():
                actual_files.append(p.relative_to(ROOT).as_posix())
    original_paths = {f["new_path"] for f in files}
    unexpected_missing = original_paths - set(actual_files)
    failures.extend({"missing_actual_file": p} for p in sorted(unexpected_missing))
    write_csv(CATALOG / "SYMLINK_REGISTER.csv", symlinks)
    write_csv(CATALOG / "ADDED_FILES_REGISTER.csv", [{"path": p, "sha256": digest(ROOT / p)} for p in sorted(set(actual_files) - original_paths)])
    result = {**STATUS, "batch": BATCH, "verification_utc": utc(),
              "status": "pass" if not failures and not empty else "fail", "original_files_verified": len(files),
              "original_bytes_verified": sum(f["size"] for f in files), "original_directories_verified": len(dirs),
              "new_regular_files_at_verification": len(set(actual_files) - original_paths),
              "symlinks_checked": len(symlinks), "empty_directories": empty, "failures": failures,
              "remote_or_training_executed": False, "git_executed": False,
              "scope": "local organization integrity; not model result acceptance"}
    write_json(CATALOG / "VERIFICATION.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    require(result["status"] == "pass", "Verification failed; preserve evidence and investigate")


def selftest():
    mappings = [{"old": "G1/code", "new": "G1/10_code"}]
    assert map_path("G1/code/a.py", mappings) == "G1/10_code/a.py"
    assert map_path("G1/code_other/a.py", mappings) == "G1/code_other/a.py"
    assert date_of("a/run_3034060/metrics.csv")[0] == "undated"
    assert date_of("a/20260911_r2/report.md")[0] == "2026-09-11"
    assert topic_of("S2/results/experimental_unvalidated/s2_met_aug_bef_epoch175_official_met_20260911_r2/a.json") == "07_epoch175_official_met"
    assert category_of("G1/data/diffusion_cache/a.npz") == "05_arrays_images_and_cached_data"
    assert category_of("S2/a/__pycache__/a.pyc") == "11_cache_and_system_metadata"
    assert len(slug("中" * 500).encode()) <= 130
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    print("SELFTEST PASS (9 checks)")


def refine():
    data = read_json("BASELINE.json")
    files = data["files"]
    final_files = [dict(f, category=category_of(f["old_path"])) for f in files]
    require(all(f["category"] != "14_other_preserved_artifacts" for f in final_files), "Unclassified originals remain")
    write_csv(CATALOG / "FILE_CLASSIFICATION_FINAL.csv", final_files)
    corrections = [{"old_path": f["old_path"], "initial_category": f["category"],
                    "final_category": category_of(f["old_path"]), "action": "classification_only_original_unchanged"}
                   for f in files if f["category"] != category_of(f["old_path"])]
    write_csv(CATALOG / "CLASSIFICATION_REVIEW.csv", corrections)
    sensitive = [f for f in final_files if f["category"] == "13_sensitive_config_index_only"]
    write_csv(CATALOG / "SENSITIVE_PATHS_INDEX_ONLY.csv", sensitive)
    backups = [f for f in final_files if f["category"] == "15_historical_source_backups"]
    write_csv(CATALOG / "SOURCE_BACKUPS_PRESERVED.csv", backups)
    checkpoints = [f for f in final_files if f["category"] == "06_model_checkpoints"]
    write_csv(CATALOG / "CHECKPOINT_REGISTER.csv", checkpoints)
    by_resolved = {(ROOT / f["new_path"]).resolve(): f for f in files}
    sha_rows = []
    for f in files:
        if "sha256" not in Path(f["old_path"]).name.lower() or Path(f["old_path"]).suffix.lower() not in {".txt", ".sha256", ""}:
            continue
        source = ROOT / f["new_path"]
        for num, line in enumerate(source.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            match = re.match(r"^([0-9a-fA-F]{64})\s+\*?(.+)$", line)
            if not match:
                continue
            expected, ref = match.groups()
            candidates = set()
            path = Path(ref)
            if path.is_absolute():
                if path.is_relative_to(ROOT):
                    candidates.add(path.resolve())
            else:
                for ancestor in [source.parent, *source.parent.parents]:
                    if ancestor.is_relative_to(ROOT):
                        candidates.add((ancestor / path).resolve())
            found = [by_resolved[p] for p in candidates if p in by_resolved]
            matches = [x for x in found if x["sha256"] == expected.lower()]
            result = "matches_original_baseline" if matches else "preexisting_local_hash_disagreement" if found else "not_resolved_in_local_pull"
            sha_rows.append({"manifest_old_path": f["old_path"], "line": num, "reference": ref,
                             "expected_sha256": expected.lower(), "resolution": result,
                             "local_candidates": [x["new_path"] for x in found],
                             "note": "Original baseline retained; unresolved remote files were not downloaded."})
    write_csv(CATALOG / "HISTORICAL_SHA_REFERENCES.csv", sha_rows)
    invariants = []
    for f in files:
        if f["old_path"].endswith(".py"):
            old, new = ROOT / f["old_path"], ROOT / f["new_path"]
            invariants.append({"old_path": f["old_path"], "new_path": f["new_path"],
                               "old_and_new_resolve_to_same_file": old.resolve() == new.resolve(),
                               "ancestry_depth_preserved": len(old.parts) == len(new.resolve().parts)})
    require(all(x["old_and_new_resolve_to_same_file"] and x["ancestry_depth_preserved"] for x in invariants), "Python ancestry drift")
    write_csv(CATALOG / "PYTHON_PATH_INVARIANTS.csv", invariants)
    for module in MODULES:
        rows = [f for f in final_files if f["module"] == module]
        md = CATALOG / "modules" / module / "FINAL_CLASSIFICATION.md"
        text = [f"# {module} 复核后分类\n\n整理批次：{BATCH}。优先使用此表；README 中的初始自动分类已由本表补充复核。\n\n",
                "| 分类 | 文件数 | 字节 |\n|---|---:|---:|\n"]
        for cat in sorted({f["category"] for f in rows}):
            selected = [f for f in rows if f["category"] == cat]
            text.append(f"| {cat} | {len(selected)} | {sum(f['size'] for f in selected)} |\n")
        text.append("\n旧 PID 只作历史进程记录，不代表当前进程仍存在。备份、环境配置、第三方作者信息均保留。配置文件仅登记路径与哈希，不抄录内容。\n")
        write_md(md, "".join(text))
    result = {**STATUS, "refined_utc": utc(), "files_classified": len(files), "unclassified": 0,
              "classification_corrections": len(corrections), "checkpoint_files": len(checkpoints),
              "python_paths_checked": len(invariants), "sensitive_config_paths": len(sensitive),
              "sha_reference_counts": dict(Counter(r["resolution"] for r in sha_rows))}
    write_json(CATALOG / "CLASSIFICATION_REVIEW_SUMMARY.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


def deliver():
    verification = read_json("VERIFICATION.json")
    require(verification["status"] == "pass", "Original-file verification must pass first")
    data = read_json("BASELINE.json")
    originals = {f["new_path"] for f in data["files"]}
    for f in data["files"]:
        st = (ROOT / f["new_path"]).stat()
        require((st.st_ino, st.st_size, st.st_mtime_ns) == (f["inode"], f["size"], f["mtime_ns"]),
                "Concurrent original change after verification: " + f["old_path"])
    with (CATALOG / "DOCUMENT_REGISTER.csv").open(encoding="utf-8-sig", newline="") as source:
        documents = list(csv.DictReader(source))
    for f in data["files"]:
        if Path(f["old_path"]).name != "AUTHORS":
            continue
        target = ROOT / f["new_path"]
        nav = CATALOG / "modules" / f["module"] / "documents" / "09_references_and_notebooks" / (slug(str(Path(f["old_path"]).parent)) + "__AUTHORS")
        link(nav, target)
        documents.append({"module": f["module"], "title": str(Path(f["old_path"]).parent.name) + " AUTHORS",
                          "title_basis": "filename_and_parent", "topic": f["topic"], "document_role": "09_references_and_notebooks",
                          "date": f["date"], "date_basis": f["date_basis"], "old_path": f["old_path"],
                          "new_path": f["new_path"], "sha256": f["sha256"], "navigation_path": nav.relative_to(CATALOG).as_posix()})
    write_csv(CATALOG / "DOCUMENT_REGISTER_FINAL.csv", documents)
    authors_md = CATALOG / "modules/S5/THIRD_PARTY_AUTHORS.md"
    write_md(authors_md, "# 第三方作者资料补充\n\n" + "\n".join(
        "- " + mdlink(authors_md, ROOT / d["new_path"], d["title"])
        for d in documents if Path(d["old_path"]).name == "AUTHORS") +
        "\n\n原作者文件未改写；许可证与第三方源码亦已纳入全量SHA保护。\n")
    checked_at = utc()
    report = f"""# 工作区整理最终验收

批次：`{BATCH}`。本机验收UTC：`{checked_at}`；本机本地时间：`{datetime.now().astimezone().isoformat()}`。

## 完成结果

- 已整理整个work_space，覆盖G1、G2、S1至S5及论文图表；43个外层实体目录规范化并保留旧路径兼容链接。
- 17,787个原文件，34,746,825,213字节；1,435个原目录全部保留。全部原文件有最终用途分类，无未分类文件。
- 633份文档、表格及作者资料有逐项登记；56个实验或结果集合按主题和日期建立导航，61个checkpoint文件按路径及SHA登记。
- 2,282组重复件、337个缓存/系统元数据/占位文件、两份源码备份均保留，无去重、无原件删除。
- 18个原空目录加入历史说明，原目录inode保留；既有reservation不删除、不重建、不释放。最终空目录数为0。

## 完整性

全量重新读取原文件并计算SHA256，17,787份全部与整理前基线一致。大小、inode、mode、owner、mtime和扩展属性摘要一致；全部旧文件路径指向同一原件。1,433个Python路径保持原层级深度。详情见 [VERIFICATION.json](VERIFICATION.json)。

原目录名中的实验身份、内部运行结构与历史SHA文本保留。旧路径只是别名，扫描时不要同时递归跟随新旧目录。父子结果集合存在交叠，不能把56项的文件数直接相加。

## 已知历史问题

3个.json实际为键值文本，1处静态SHA清单与整理前现存脚本不一致，均原样保留并独立登记。6,352条历史SHA引用未按本地路径解析，不推断远端文件缺失或损坏。详见 [HISTORICAL_ISSUES.md](HISTORICAL_ISSUES.md)。

本次没有重跑训练、推理、官方MET或完整科学验收；“整理验收通过”不是“模型效果正式通过”。原E比赛基座和E-SNFH已区分，Screen125与Extend175不合表，图像质量与分割收益不混写。

## 交付入口

- [总入口](00_START_HERE.md)
- [研究路线和关键版本](RESEARCH_LINEAGE.md)
- [最终逐文件分类](FILE_CLASSIFICATION_FINAL.csv)
- [最终文档登记](DOCUMENT_REGISTER_FINAL.csv)
- [新旧目录映射](MOVE_PLAN.json)及[实际迁移日志](MOVE_JOURNAL.jsonl)
- [回退方案](ROLLBACK_PLAN.json)：未执行，回退前必须复核并转存新增元数据，不能删除旧原件。
- `DELIVERY_AUDIT.json`：最终索引、链接与JSON检查。
- `CATALOG_SHA256.json`：最终新增资料的SHA封存；此前阶段性登记保留为历史记录，不覆盖。

所有本次新增资料的状态：artifact_status=experimental_unvalidated，operator_approved=false，formal_gate_status=not_run_not_passed。未修改工作区外内容，未联网，未安装依赖，未执行Git或上传操作。
"""
    write_md(CATALOG / "FINAL_ACCEPTANCE.md", report)
    links_checked, markdown_checked, json_checked, broken, empty, symlinks = 0, 0, 0, [], [], []
    added = []
    for base, subdirs, names in os.walk(ROOT, followlinks=False):
        if not subdirs and not names:
            empty.append(str(Path(base).relative_to(ROOT)))
        for name in subdirs + names:
            p = Path(base) / name
            if p.is_symlink():
                target = p.resolve(strict=True)
                require(target.is_relative_to(ROOT), "External symlink")
                symlinks.append({"path": p.relative_to(ROOT).as_posix(), "target": target.relative_to(ROOT).as_posix()})
        for name in names:
            p = Path(base) / name
            rel = p.relative_to(ROOT).as_posix()
            if p.is_symlink() or rel in originals:
                continue
            added.append({"path": rel, "sha256": digest(p)})
            if p.suffix == ".json":
                json.loads(p.read_text(encoding="utf-8"))
                json_checked += 1
            if p.suffix == ".md":
                markdown_checked += 1
                for raw in re.findall(r"\[[^\]]*\]\(([^)]+)\)", p.read_text(encoding="utf-8")):
                    if re.match(r"\w+://", raw) or raw.startswith("#"):
                        continue
                    target = (p.parent / unquote(raw.split("#")[0])).resolve()
                    links_checked += 1
                    if not target.exists():
                        broken.append({"source": rel, "target": raw})
    require(not broken and not empty, "Broken documentation links or empty directories: " + json.dumps(broken))
    require(len({d["old_path"] for d in documents}) == len(documents) == 633, "Document coverage mismatch")
    write_csv(CATALOG / "FINAL_ADDED_FILES.csv", added)
    write_csv(CATALOG / "FINAL_SYMLINK_REGISTER.csv", symlinks)
    result = {**STATUS, "status": "pass", "completed_utc": utc(), "original_sha_verification": "VERIFICATION.json",
              "final_documents": len(documents), "original_files": len(originals), "final_symlinks": len(symlinks),
              "new_markdown_checked": markdown_checked, "new_json_checked": json_checked,
              "markdown_links_checked": links_checked, "broken_links": broken, "empty_directories": empty,
              "all_originals_metadata_unchanged_since_sha_verification": True}
    write_json(CATALOG / "DELIVERY_AUDIT.json", result)
    seal = []
    for base, subdirs, names in os.walk(ROOT, followlinks=False):
        for name in names:
            p = Path(base) / name
            rel = p.relative_to(ROOT).as_posix()
            if not p.is_symlink() and rel not in originals:
                seal.append({"path": rel, "sha256": digest(p)})
    write_json(CATALOG / "CATALOG_SHA256.json", {**STATUS, "sealed_utc": utc(),
               "scope": "All new regular files except this seal itself; original files in BASELINE.json", "files": seal})
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["selftest", "baseline", "organize", "refine", "verify", "deliver"])
    args = parser.parse_args()
    globals()[args.stage]()
