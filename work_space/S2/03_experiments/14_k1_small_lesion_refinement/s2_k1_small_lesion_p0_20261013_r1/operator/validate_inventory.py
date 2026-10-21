"""Validate one saved inventory against frozen per-file identity, not modality order."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime"))
from audit_existing import META, require, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    args = parser.parse_args()
    path = args.snapshot / "REMOTE_INVENTORY.json"
    data = json.loads(path.read_text())
    source = ROOT.parents[2] / "03_experiments/12_k1245_rc_aware/s2_met_aug_k1_selected103_official_met_20261008_r1/manifests/INPUT_REFERENCE_SHA256.json"
    expected = json.loads(source.read_text())["files"]
    checks = data["input_reference_checks"]
    require(len(checks) == len(expected) == 515, "515 input contract mismatch")
    actual = {item["path"]: item["sha256"] for item in checks}
    require(len(actual) == 515 and actual == expected, "frozen per-file input/reference SHA drift")
    frozen = ROOT.parents[2] / "03_experiments/12_k1245_rc_aware/s2_met_aug_k1_175_resume_20261002_r1/evidence/REMOTE_FINAL_PREFLIGHT.json"
    dependencies = json.loads(frozen.read_text())["dependency_sha256"]
    observed = {r["path"]: r["sha256"] for r in data["frozen_files"]}
    require(observed == dependencies, "dependency SHA differs from local frozen contract")
    require(all(r["exists"] and r["sha256"] == r["expected_sha256"] for r in data["frozen_files"]), "remote dependency contract drift")
    require(all(r["sha256"] == r["expected_sha256"] for r in checks), "modality-to-file audit SHA mismatch")
    require(data["cohort_counts"] == dict(input=412, reference=103), "input count drift")
    require(data["input_audit_sha256"] == "7335aee3869fdc10fe776b62465a09600eb0f510f71f397102bbe45d6a63446e", "input audit SHA drift")
    require(not any("error" in v for v in data["software"].values()), "nnU-Net source inspection error")
    require(data["official_evaluator"]["versions"] == {"panoptica": "2.1.0", "BraTS-evaluation": "0.0.8", "numpy": "1.26.4"}, "official MET version drift")
    checkpoints = {Path(r["path"]).name: r for r in data["checkpoints"]}
    require(checkpoints["checkpoint_epoch_0175.pth"]["sha256"] == "28e81950b675cd0f5ca26cc1509f9271b2fa827b524f379024b417c0a5402204", "K1-175 SHA drift")
    require("checkpoint_epoch_0150.pth" in checkpoints, "K1-150 absent")
    for name, count in (("train_fixed.txt", 1035), ("val_fixed.txt", 103)):
        ids = data["split_lists"][name]
        require(len(ids) == len(set(ids)) == count, "split count/duplicate drift")
    require(not set(data["split_lists"]["train_fixed.txt"]) & set(data["split_lists"]["val_fixed.txt"]), "train/development overlap")
    destination = args.snapshot / "installed_source"
    destination.mkdir(exist_ok=False)
    for name, record in data["software"].items():
        target = destination / (name.replace(".", "_") + ".py")
        content = record["source"].encode()
        require(hashlib.sha256(content).hexdigest() == record["sha256"], "source content SHA mismatch")
        with target.open("xb") as handle:
            handle.write(content)
    for package, key in (("panoptica", "python_sources"), ("brats_evaluation", "brats_sources")):
        for relative, content in data["official_evaluator"][key].items():
            rel = Path(relative)
            require(not rel.is_absolute() and ".." not in rel.parts, "source traversal")
            target = destination / package / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("x") as handle:
                handle.write(content)
    write_json(args.snapshot / "INVENTORY_VALIDATION.json", dict(**META, status="pass",
        input_reference_sha_checked=515, dependency_sha_checked=len(data["frozen_files"]),
        original_inventory_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        remote_utc=data["remote_utc"], selected175_sha256=checkpoints["checkpoint_epoch_0175.pth"]["sha256"],
        selected150_sha256=checkpoints["checkpoint_epoch_0150.pth"]["sha256"],
        inventory_modality_note="Frozen input order t1n/t1c/t2w/t2f/seg and all515 per-file hashes verified; prior failed snapshot produced no validated inventory.",
        runtime_import_validated=False, inspection="installed_source_only",
        training_submitted=False, diagnosis_jobs_submitted=False))
    print(json.dumps(dict(status="pass", inputs=515, snapshot=str(args.snapshot))))


if __name__ == "__main__":
    main()
