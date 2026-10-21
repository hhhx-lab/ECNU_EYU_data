"""Resolve the reorganized local layout without rewriting historical manifests."""
import argparse
import json
import os
import shlex
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = {
    "G1_CODE_ROOT": "work_space/G1/01_code/missing_t2w_vae_bbdm",
    "G1_DATA_ROOT": "work_space/G1/02_datasets/undated_vae_evaluation_inputs",
    "G1_WEIGHTS_ROOT": "work_space/G1/04_models/undated_pretrained_vae",
    "G1_VAE_WEIGHTS": "work_space/G1/04_models/undated_pretrained_vae/vae/autoencoder_epoch273.pt",
    "G1_TRAINING_ROOT": "work_space/G1/03_experiments/new_training_runs",
    "G2_RELEASE_ROOT": "work_space/G2/02_dataset_release/20260919_archived_local_release",
    "S1_CODE_ROOT": "work_space/S1/01_code/repository",
    "S2_CODE_ROOT": "work_space/S2/01_code/repository",
    "OFFICIAL_MET_CODE_ROOT": "03_reference_code/04_task1_evaluation_and_trainers/BraTS_evaluation",
}


@lru_cache(maxsize=1)
def local_settings():
    settings = {}
    path = ROOT / ".env"
    if path.exists():
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            tokens = shlex.split(line, comments=True, posix=True)
            if not tokens:
                continue
            if len(tokens) != 1 or "=" not in tokens[0]:
                raise ValueError(f"Unsupported .env assignment on line {line_number}")
            name, value = tokens[0].split("=", 1)
            settings[name] = value
    return settings


def get_path(name, default=None):
    value = os.environ.get(name) or local_settings().get(name) or default or DEFAULTS.get(name)
    if value is None:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


@lru_cache(maxsize=1)
def relocation_tables():
    directory = ROOT / "99_archive/reorganization_20260919_r2"
    final = directory / "FINAL_PLAN_LABEL_CORRECTED.json"
    plan = json.loads((final if final.exists() else directory / "FINAL_PLAN.json").read_text(encoding="utf-8"))
    previous = json.loads((ROOT / "99_archive/20260918_index_only/catalog/MOVE_PLAN.json").read_text(encoding="utf-8"))
    files = {f["old"]: f["new"] for f in plan["files"]}
    units = sorted(((u["old"], u["new"]) for u in plan["units"]), key=lambda p: len(p[0]), reverse=True)
    refinement = ROOT / "99_archive/s1_s2_reorganization_20260919_r3/PLAN.json"
    if refinement.is_file():
        changes = {m["old"]: m["new"] for m in json.loads(refinement.read_text(encoding="utf-8"))["moves"]}
        files = {old: changes.get(new, new) for old, new in files.items()}
        files.update(changes)
    return files, units, previous["mappings"]


def resolve_legacy_path(value, anchor=None):
    path = Path(value).expanduser()
    if path.is_absolute():
        try:
            rel = path.relative_to(ROOT).as_posix()
        except ValueError:
            return path  # A server path or another project is not a local relocation.
    else:
        rel = path.as_posix()
    current = ROOT / rel
    files, units, previous = relocation_tables()
    if rel in files:
        return ROOT / files[rel]
    for move in previous:
        if rel == "work_space/" + move["old"] or rel.startswith("work_space/" + move["old"] + "/"):
            old = "work_space/" + move["old"]
            rel = "work_space/" + move["new"] + rel[len(old):]
            break
    if rel in files:
        return ROOT / files[rel]
    if current.exists():
        return current
    if anchor is not None and not path.is_absolute():
        candidate = Path(anchor) / path
        if candidate.exists():
            return candidate.resolve()
    for old, new in units:
        if rel == old or rel.startswith(old + "/"):
            candidate = ROOT / (new + rel[len(old):])
            if candidate.exists():
                return candidate
    return (Path(anchor) / path).resolve() if anchor is not None and not path.is_absolute() else current


def main():
    parser = argparse.ArgumentParser(description="Read-only local path configuration and legacy-path resolver")
    parser.add_argument("path", nargs="?")
    args = parser.parse_args()
    if args.path:
        path = resolve_legacy_path(args.path)
        print(json.dumps({"resolved": str(path), "exists": path.exists()}, ensure_ascii=False))
    else:
        print(json.dumps({k: {"path": str(get_path(k)), "exists": get_path(k).exists()} for k in DEFAULTS}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
