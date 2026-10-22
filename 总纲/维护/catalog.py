"""Build a local, evidence-linked project catalog without running project code."""
import argparse
import ast
import collections
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import warnings
from urllib.parse import quote, unquote

ROOT = Path(__file__).resolve().parents[2]
HOME = ROOT / "总纲"
STATUS = dict(artifact_status="experimental_unvalidated", operator_approved=False,
              formal_gate_status="not_run_not_passed")
FIELDS = ["path", "absolute_path", "kind", "module", "category", "unit", "description",
          "basis", "structure", "context_document", "size_bytes", "mtime_ns", "mode",
          "sha256", "inspection", "link_target"]
CATEGORY = {
    "01_code": "活动代码", "02_datasets": "数据与数据契约", "02_dataset_release": "G2数据发布合同",
    "03_experiments": "实验结果与冻结证据", "04_models": "独立模型权重",
    "05_documents": "模块文档", "06_operations": "历史运行与服务器资料",
    "07_submissions": "比赛推理与提交材料", "01_project_documents": "项目要求与跨模块文档",
    "02_reference_materials": "外部参考材料", "03_reference_code": "上游参考代码",
    "04_paper_figures": "论文图表与绘图源码", "data_space": "共享数据与参考权重",
    "99_archive": "不可删的历史保留区", "tools": "本地路径和结构工具", "总纲": "项目总纲与索引",
}
ROLE_PATTERNS = [
    (r"FAILED|FAILURE", "失败证据；保留失败语义，不因后续成功而改写"),
    (r"VALIDATION|PREFLIGHT", "验收或预检记录；须结合具体status与同批次证据解释"),
    (r"COMPLETE.*\.ok$", "该批次的完成标记；单独存在不等于科学结论通过"),
    (r"STARTED", "启动时运行合同记录；不等于完成"),
    (r"LAUNCH|RESERVATION", "排他提交或作业启动审计；不应重复执行或删除"),
    (r"SHA256|\.sha256$", "原文件校验清单；历史路径保持原样"),
    (r"PLAN|计划", "计划与预注册约束；计划不是执行结果"),
    (r"PAIR|BOOTSTRAP", "同病例配对比较或重采样结果；核对方向、n、CI和p"),
    (r"RANK|COMPARISON|对比", "模型或指标比较表；比较口径以所属实验为准"),
    (r"selection|选择", "checkpoint或模型选择证据；以冻结规则为准"),
    (r"manifest|mapping|membership|split", "数据成员、映射或划分合同；不是影像本体"),
    (r"metrics|summary", "指标、统计或汇总数据；不可跨不同口径直接比较"),
    (r"00_说明", "该单元的归档说明；含来源、用途及本地内容"),
    (r"README", "所在目录的说明入口；历史说明可能滞后于最终验收"),
]


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def restricted(rel):
    parts = Path(rel).parts
    name = Path(rel).name.lower()
    return (".git" in parts or ".claude" in parts or ".ssh" in parts
            or name.startswith(".env") or name.startswith("id_rsa")
            or name.startswith("id_ed25519") or name.endswith((".pem", ".key", ".p12"))
            or bool(re.search(r"credential|secret|token|password|登陆|登录|vpn|ssh.config", name)))


def group(rel):
    p = Path(rel).parts
    module = (p[1] if p[1] in {"G1", "G2", "S1", "S2", "S3", "S4", "S5"} else "work_space") if len(p) > 1 and p[0] == "work_space" else p[0] if len(p) > 1 else "root"
    category = next((CATEGORY[x] for x in p if x in CATEGORY), "根配置或其他文件")
    if p[0] == "work_space" and len(p) >= 3:
        category = CATEGORY.get(p[2], "模块根说明")
    return module, category


def entries(root):
    records = []
    def onerror(error):
        raise error
    for base, dirs, files in os.walk(root, followlinks=False, onerror=onerror):
        # Git metadata is repository state rather than project material.  It
        # changes after every commit and must not invalidate a catalog check.
        dirs[:] = [name for name in dirs if name != ".git"]
        for name in sorted(dirs + files):
            path = Path(base) / name
            s = path.lstat()
            kind = "symlink" if stat.S_ISLNK(s.st_mode) else "directory" if path.is_dir() else "file"
            records.append((path.relative_to(root).as_posix(), kind, s))
    return sorted(records)


def match_unit(rel, units):
    return next((u for u in units if rel == u["path"] or rel.startswith(u["path"] + "/")), None)


def role(rel, kind):
    p = Path(rel)
    low = rel.lower()
    if restricted(rel):
        return "敏感配置或版本控制内部对象；仅登记路径与文件元数据，不读取内容"
    if kind == "symlink":
        return "符号链接对象；只登记，不跟随到项目外，也不当成新增数据"
    if "总纲/90_索引快照/" in rel or rel in {"总纲/当前索引.json", "总纲/当前索引.md"}:
        return "总纲生成的路径目录或覆盖验收；路径纳入索引，不作自引用内容校验"
    if p.name == ".DS_Store":
        return "macOS Finder目录显示元数据；保留，不作为实验数据"
    if kind == "directory":
        return CATEGORY.get(p.name, "子目录；用途由模块、所属单元和下级文件确定")
    for pattern, description in ROLE_PATTERNS:
        if re.search(pattern, p.name, re.I):
            return description
    if low.endswith((".nii", ".nii.gz")):
        if re.search(r"label|reference|seg\.nii", low):
            return "NIfTI标签或参考体积；具体类别和来源以同批manifest为准"
        if "mask" in low:
            return "NIfTI有效区或空间掩膜；不是分割性能结果"
        if "prediction" in low or "/validation/" in low:
            return "NIfTI预测体积；用途按实验上下文推断，未逐体素验收"
        return "NIfTI影像体积；病例、模态及合成来源由路径和同批manifest定位"
    if p.suffix.lower() in {".pt", ".pth", ".ckpt", ".safetensors"}:
        return "模型checkpoint或张量权重；未反序列化，模型身份须查SHA与训练证据"
    if p.suffix == ".py":
        return "Python测试" if "test" in p.name or "/tests/" in low else "Python实现或运行工具；只做AST静态阅读，不导入执行"
    suffix_roles = {".sh": "Shell运行脚本；不执行", ".slurm": "历史Slurm作业脚本；不提交",
                    ".sbatch": "历史Slurm提交脚本；不提交", ".json": "结构化配置、审计或数据记录",
                    ".jsonl": "逐事件或逐病例记录；未将原始行复制进总纲",
                    ".csv": "表格数据或清单", ".tsv": "制表符表格", ".yaml": "参数配置", ".yml": "参数配置",
                    ".md": "Markdown说明、方法或历史记录", ".txt": "文本说明或清单",
                    ".log": "运行日志；存在日志不证明成功", ".out": "作业标准输出", ".err": "作业标准错误",
                    ".pdf": "PDF资料或报告；本轮仅登记，不作逐页复核", ".docx": "可编辑Word文档；本轮仅登记",
                    ".xlsx": "工作簿；本轮仅登记，不改变单元格", ".png": "图片、QC拼图或图表；未逐张视觉验收",
                    ".jpg": "图像资料；未逐张视觉验收", ".svg": "矢量图形或图表源码",
                    ".npz": "数组、潜变量或增强中间产物；未解码", ".npy": "数组或潜变量；未解码",
                    ".zip": "已有压缩包；原样保留，不解包也不新建压缩包", ".pyc": "历史Python字节码缓存；不执行"}
    return suffix_roles.get(p.suffix.lower(), "其他原始文件；按所属目录定位，内容格式尚待专项确认")


def inspect_structure(path, rel, size):
    if restricted(rel) or "/90_索引快照/" in rel:
        return "", "metadata_only"
    if size > 2 * 1024 * 1024:
        return "", "large_file_metadata_only"
    try:
        if path.suffix == ".py":
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", SyntaxWarning)
                tree = ast.parse(path.read_text(encoding="utf-8-sig"))
            names = [x.name for x in tree.body if isinstance(x, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))]
            return "顶层定义: " + ", ".join(names[:18]) + ("; 语法警告=" + str(len(caught)) if caught else ""), "ast_only_warnings" if caught else "ast_only"
        if path.suffix == ".md" and "06_operations" not in rel and "99_archive" not in rel:
            headings = re.findall(r"^#{1,4}\s+(.+)$", path.read_text(encoding="utf-8-sig"), re.M)
            headings = [h for h in headings if not re.search(r"password|secret|token|credential|密钥|密码", h, re.I)]
            return "标题: " + " / ".join(h[:160] for h in headings[:4]), "document_headings_only"
        if path.suffix == ".json":
            obj = json.loads(path.read_text(encoding="utf-8-sig"))
            shape = "顶层字段: " + ", ".join(map(str, list(obj)[:18])) if isinstance(obj, dict) else type(obj).__name__ + " 长度=" + str(len(obj)) if isinstance(obj, list) else type(obj).__name__
            return shape, "schema_only"
        if path.suffix in {".csv", ".tsv"}:
            with path.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.reader(stream, delimiter="\t" if path.suffix == ".tsv" else ",")
                header = next(reader, [])
                count = sum(1 for _ in reader)
            if any(re.search(r"password|secret|token|credential", h, re.I) for h in header):
                return "表格含敏感字段名；不提取", "metadata_only"
            return "数据行=" + str(count) + "; 列: " + ", ".join(header[:16]), "schema_and_row_count"
    except (ValueError, SyntaxError, UnicodeError, csv.Error) as error:
        return "静态解析未完成: " + type(error).__name__, "parse_error"
    return "", "metadata_and_path"


def row_for(root, rel, kind, s, units):
    module, category = group(rel)
    unit = match_unit(rel, units)
    path = root / rel
    generated = rel.startswith("总纲/90_索引快照/") or rel in {"总纲/当前索引.json", "总纲/当前索引.md"}
    structure, inspection = inspect_structure(path, rel, s.st_size) if kind == "file" else ("", "metadata_only")
    context = unit["description"] if unit else category
    r = dict(zip(FIELDS, [""] * len(FIELDS)))
    r.update(path=rel, absolute_path=str(path), kind=kind, module=module, category=category,
             unit=unit["path"] if unit else "", description=context + "；" + role(rel, kind),
             basis="curated_context+" + inspection if unit else "path_rule+" + inspection,
             structure=structure, context_document=(unit.get("evidence") or [""])[0] if unit else "",
             size_bytes=s.st_size if kind != "directory" and not generated else "",
             mtime_ns=s.st_mtime_ns if kind != "directory" and not generated else "",
             mode=oct(stat.S_IMODE(s.st_mode)), inspection=inspection,
             link_target=os.readlink(path) if kind == "symlink" else "")
    if kind == "file" and s.st_size <= 2 * 1024 * 1024 and not restricted(rel) and not generated:
        r["sha256"] = digest(path)
    return r


def link(label, target, from_dir):
    return "[" + label.replace("|", "\\|").replace("[", "\\[").replace("]", "\\]") + "](" + quote(os.path.relpath(target, from_dir), safe="/._-") + ")"


def cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def write_csv(path, rows, fields=FIELDS):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def check_links(root, home, allow_initial_pointer=False):
    broken = []
    count = 0
    for p in home.rglob("*.md"):
        if "90_索引快照" in p.parts or "历史原件" in p.parts:
            continue
        for url in re.findall(r"\]\(([^)]+)\)", p.read_text(encoding="utf-8")):
            if url.startswith(("https://", "http://", "#", "mailto:")):
                continue
            target = (p.parent / unquote(url.split("#", 1)[0].strip("<>"))).resolve()
            count += 1
            expected = allow_initial_pointer and target in {home / "当前索引.json", home / "当前索引.md"}
            if not target.exists() and not expected:
                broken.append({"source": p.relative_to(root).as_posix(), "target": url})
    return count, broken


def source_hashes(root, config):
    result = {}
    for unit in config["units"]:
        if not (root / unit["path"]).exists():
            raise ValueError("Missing curated unit: " + unit["path"])
        for rel in unit.get("evidence", []):
            if not (root / rel).is_file():
                raise ValueError("Missing curated evidence: " + rel)
            if restricted(rel):
                raise ValueError("Restricted evidence must not be extracted: " + rel)
            result[rel] = digest(root / rel)
    home = root / "总纲"
    for p in home.glob("*.md"):
        if p.name == "当前索引.md":
            continue
        for url in re.findall(r"\]\(([^)]+)\)", p.read_text(encoding="utf-8")):
            if url.startswith(("https://", "http://", "#", "mailto:")):
                continue
            target = (p.parent / unquote(url.split("#", 1)[0].strip("<>"))).resolve()
            if target.is_file() and target.is_relative_to(root) and not target.is_relative_to(home):
                rel = target.relative_to(root).as_posix()
                if not restricted(rel):
                    result[rel] = digest(target)
    return result


def build(root=ROOT, home=HOME, reason="manual refresh"):
    root, home = root.resolve(), home.resolve()
    config = json.loads((home / "维护/知识登记.json").read_text(encoding="utf-8"))
    units = sorted(config["units"], key=lambda x: len(x["path"]), reverse=True)
    evidence_hashes = source_hashes(root, config)
    checked, broken = check_links(root, home, allow_initial_pointer=True)
    if broken:
        raise ValueError("Broken curated links: " + json.dumps(broken, ensure_ascii=False))
    old_pointer = home / "当前索引.json"
    previous = json.loads(old_pointer.read_text(encoding="utf-8")) if old_pointer.exists() else None
    if previous:
        old = json.loads((root / previous["snapshot"] / "覆盖验收.json").read_text(encoding="utf-8"))
        changed = [p for p, h in old["curated_source_sha256"].items() if evidence_hashes.get(p) != h]
        if changed and config.get("reviewed_at", "") <= old["knowledge_reviewed_at"]:
            raise ValueError("Curated evidence changed; review conclusions and advance reviewed_at first: " + str(changed))
    now = dt.datetime.now(dt.timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%S%fZ")
    snap = home / "90_索引快照" / run_id
    snap.mkdir(parents=True, exist_ok=False)
    (snap / "逐文件").mkdir()
    scanned = entries(root)
    rows = [row_for(root, r, k, s, units) for r, k, s in scanned]
    files = [r for r in rows if r["kind"] != "directory"]
    directories = [r for r in rows if r["kind"] == "directory"]
    group_rows = collections.defaultdict(list)
    for r in files:
        group_rows[r["module"]].append(r)
    pages = []
    for group_id, values in sorted(group_rows.items()):
        for start in range(0, len(values), 400):
            page_name = re.sub(r"[^A-Za-z0-9_-]", "_", group_id) or "other"
            page_name += "_" + hashlib.sha256(group_id.encode()).hexdigest()[:6] + "_" + f"{start // 400 + 1:03d}.md"
            pages.append((snap / "逐文件" / page_name, group_id, values[start:start + 400]))
    out_names = ["README.md", "全部文件.csv", "全部目录.csv", "完整文件树.txt", "实验登记.md", "数据与模型实体.md", "覆盖验收.json", "更新差异.json"]
    outputs = [snap / n for n in out_names] + [p[0] for p in pages] + [snap / "逐文件/总纲生成文件.md", old_pointer, home / "当前索引.md"]
    actual = {r["path"] for r in files}
    generated_rows = []
    for p in outputs:
        rel = p.relative_to(root).as_posix()
        if rel not in actual:
            r = dict(zip(FIELDS, [""] * len(FIELDS)))
            r.update(path=rel, absolute_path=str(p), kind="file", module="总纲", category=CATEGORY["总纲"],
                     description="本次总纲索引生成文件；不读取自引用大小、mtime或SHA", basis="generated_path_contract",
                     inspection="generated_path_only", mode="0o644")
            files.append(r)
            generated_rows.append(r)
    files.sort(key=lambda r: r["path"])
    old_rows = {}
    if previous:
        with (root / previous["snapshot"] / "全部文件.csv").open(encoding="utf-8-sig", newline="") as stream:
            old_rows = {r["path"]: r for r in csv.DictReader(stream)}
    current = {r["path"]: r for r in files}
    diff = {"added": sorted(set(current) - set(old_rows)), "removed": sorted(set(old_rows) - set(current)),
            "changed": [p for p in sorted(set(current) & set(old_rows))
                        if any(str(current[p][k]) != old_rows[p][k] for k in ("size_bytes", "mtime_ns", "sha256", "link_target"))]}
    dump(snap / "更新差异.json", {**STATUS, "reason": reason, "previous": previous, **diff})
    for page, group_id, values in pages + [(snap / "逐文件/总纲生成文件.md", "本次生成", generated_rows)]:
        lines = ["# " + group_id + " 逐文件目录", "", "路径全覆盖不等于逐个二进制内容验收。说明依据见最后一列。", "",
                 "| 文件 | 用途与上下文 | 静态结构 | 依据 |", "|---|---|---|---|"]
        for r in values:
            lines.append("| " + link(r["path"], root / r["path"], page.parent) + " | " + cell(r["description"]) + " | " + cell(r["structure"]) + " | " + r["basis"] + " |")
        page.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_csv(snap / "全部文件.csv", files)
    write_csv(snap / "全部目录.csv", directories)
    tree = [str(root) + "/", "目录用途见全部目录.csv；每个文件用途见全部文件.csv和逐文件目录。", ""]
    for r in sorted(files + directories, key=lambda r: r["path"]):
        depth = len(Path(r["path"]).parts) - 1
        tree.append("|   " * depth + "|-- " + Path(r["path"]).name + ("/" if r["kind"] == "directory" else "")
                    + (" -> " + r["link_target"] if r["kind"] == "symlink" else ""))
    (snap / "完整文件树.txt").write_text("\n".join(tree) + "\n", encoding="utf-8")
    exp = ["# 实验与QC批次登记", "", "状态由人工核对证据登记，不从目录名中的pass/complete自动推定。所有历史attempt独立保留。", "",
           "| 路径 | 已核实内容/边界 | 状态 | 证据 |", "|---|---|---|---|"]
    for unit in config["units"]:
        if unit.get("kind") == "experiment":
            refs = " · ".join(link(Path(e).name, root / e, snap) for e in unit.get("evidence", []))
            exp.append("| " + link(unit["path"], root / unit["path"], snap) + " | " + cell(unit["description"]) + " | " + cell(unit.get("state", "未专项确认")) + " | " + refs + " |")
    (snap / "实验登记.md").write_text("\n".join(exp) + "\n", encoding="utf-8")
    entities = ["# 本地数据与模型实体", "", "只统计当前物理文件，不把远端路径当成本地模型；同一权重的重复文件不代表不同模型。", "",
                "| 类型 | 路径 | 说明 |", "|---|---|---|"]
    for r in directories:
        if "/02_datasets/" in r["path"] and len(Path(r["path"]).parts) == 4 or "/02_dataset_release/" in r["path"] and len(Path(r["path"]).parts) == 4:
            count = sum(f["path"].startswith(r["path"] + "/") for f in files)
            entities.append("| 数据单元 | " + link(r["path"], root / r["path"], snap) + " | " + str(count) + "个本地文件；" + cell(r["description"]) + " |")
    for r in files:
        if Path(r["path"]).suffix.lower() in {".pth", ".pt", ".ckpt", ".safetensors"}:
            entities.append("| 权重/张量 | " + link(r["path"], root / r["path"], snap) + " | " + str(r["size_bytes"]) + " bytes；未加载 |")
    (snap / "数据与模型实体.md").write_text("\n".join(entities) + "\n", encoding="utf-8")
    index = ["# 全量索引 " + run_id, "", "[返回总纲](../../README.md)", "",
             "- [完整文件树](完整文件树.txt)", "- [全部文件及用途CSV](全部文件.csv)", "- [全部目录CSV](全部目录.csv)",
             "- [实验批次登记](实验登记.md)", "- [数据和权重实体](数据与模型实体.md)", "- [覆盖验收](覆盖验收.json)", "", "## 按模块逐文件阅读", ""]
    for page, group_id, values in pages:
        index.append("- " + link(group_id + " / " + page.stem + " (" + str(len(values)) + ")", page, snap))
    index.append("- [本次生成文件](逐文件/总纲生成文件.md)")
    (snap / "README.md").write_text("\n".join(index) + "\n", encoding="utf-8")
    receipt = {**STATUS, "snapshot_utc": now.isoformat(), "snapshot_local": now.astimezone().isoformat(),
               "knowledge_reviewed_at": config["reviewed_at"], "root": str(root), "reason": reason,
               "files": len(files), "directories": len(directories), "curated_units": len(units),
               "curated_source_sha256": evidence_hashes, "curated_links_checked": checked,
               "source_contents_unmodified_by_catalog": True,
               "hash_policy": "non-sensitive files <=2MiB hashed; large files metadata-only; index snapshots path-only",
               "description_basis_counts": dict(collections.Counter(r["basis"] for r in files)),
               "parse_error_paths": [r["path"] for r in files if r["inspection"] == "parse_error"],
               "syntax_warning_paths": [r["path"] for r in files if r["inspection"] == "ast_only_warnings"],
               "scientific_approval_not_implied": True, "coverage_status": "pending_postwrite_check"}
    dump(snap / "覆盖验收.json", receipt)
    dump(old_pointer, {**STATUS, "snapshot": snap.relative_to(root).as_posix(), "updated_at_utc": now.isoformat()})
    (home / "当前索引.md").write_text("# 当前全量索引\n\n更新UTC：" + now.isoformat() + "\n\n"
        + link("打开全量索引与分页目录", snap / "README.md", home) + "\n\n"
        + link("完整文件树", snap / "完整文件树.txt", home) + " · "
        + link("全部文件及用途", snap / "全部文件.csv", home) + " · "
        + link("全部目录及用途", snap / "全部目录.csv", home) + " · "
        + link("实验批次登记", snap / "实验登记.md", home) + " · "
        + link("数据与模型实体", snap / "数据与模型实体.md", home) + " · "
        + link("覆盖验收", snap / "覆盖验收.json", home) + "\n", encoding="utf-8")
    result = verify(root, home)
    receipt["coverage_status"] = result["status"]
    receipt["verification"] = result
    dump(snap / "覆盖验收.json", receipt)
    print(json.dumps({"snapshot": str(snap), **result}, ensure_ascii=False, indent=2))
    return result


def verify(root=ROOT, home=HOME):
    root, home = root.resolve(), home.resolve()
    pointer = json.loads((home / "当前索引.json").read_text(encoding="utf-8"))
    snap = root / pointer["snapshot"]
    with (snap / "全部文件.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    with (snap / "全部目录.csv").open(encoding="utf-8-sig", newline="") as stream:
        dirs = list(csv.DictReader(stream))
    actual = entries(root)
    file_set = {r for r, k, s in actual if k != "directory"}
    dir_set = {r for r, k, s in actual if k == "directory"}
    indexed = {r["path"] for r in rows}
    changed = []
    for r in rows:
        p = root / r["path"]
        if r["path"] not in file_set:
            continue
        s = p.lstat()
        if (r["size_bytes"] and int(r["size_bytes"]) != s.st_size
                or r["mtime_ns"] and int(r["mtime_ns"]) != s.st_mtime_ns
                or r["sha256"] and digest(p) != r["sha256"]
                or r["kind"] == "symlink" and os.readlink(p) != r["link_target"]):
            changed.append(r["path"])
    count, broken = check_links(root, home)
    config = json.loads((home / "维护/知识登记.json").read_text(encoding="utf-8"))
    registered = {u["path"] for u in config["units"] if u.get("kind") == "experiment"}
    unregistered = [r for r in dir_set if len(Path(r).parts) == 5
                    and Path(r).parts[:3] in {("work_space", "G1", "03_experiments"), ("work_space", "S2", "03_experiments")}
                    and r not in registered]
    stored = json.loads((snap / "覆盖验收.json").read_text(encoding="utf-8"))["curated_source_sha256"]
    source_drift = [p for p, h in stored.items() if not (root / p).is_file() or digest(root / p) != h]
    result = {"files": len(file_set), "directories": len(dir_set), "missing_from_index": sorted(file_set - indexed),
              "missing_on_disk": sorted(indexed - file_set), "changed_since_snapshot": changed,
              "directory_difference": sorted(dir_set ^ {r["path"] for r in dirs}),
              "duplicate_file_rows": len(rows) - len(indexed), "undescribed_files": [r["path"] for r in rows if not r["description"]],
              "broken_curated_links": broken, "curated_links_checked": count, "curated_source_drift": source_drift,
              "unregistered_G1_S2_experiment_units": sorted(unregistered),
              "units_missing": [u["path"] for u in config["units"] if not (root / u["path"]).exists()]}
    result["status"] = "pass" if not any(v for k, v in result.items() if k not in {"files", "directories", "curated_links_checked"}) else "fail"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["build", "check", "find"])
    parser.add_argument("query", nargs="?", default="")
    parser.add_argument("--reason", default="manual refresh")
    args = parser.parse_args()
    if args.action == "build":
        result = build(reason=args.reason)
    elif args.action == "check":
        result = verify()
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        pointer = json.loads((HOME / "当前索引.json").read_text(encoding="utf-8"))
        with (ROOT / pointer["snapshot"] / "全部文件.csv").open(encoding="utf-8-sig", newline="") as stream:
            for r in csv.DictReader(stream):
                if args.query.lower() in (r["path"] + " " + r["description"] + " " + r["structure"]).lower():
                    print(r["absolute_path"] + "\t" + r["description"])
        return
    sys.exit(0 if result["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
