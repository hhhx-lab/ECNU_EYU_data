"""Local unsubmitted candidate; preserve the failed frozen normalizer."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_SHA256 = "8c8ba3ac4daad96a4f6a8fe043757e4d102f7e913a0a2a88ec28960005258205"


def candidate_source():
    data = (ROOT / "runtime/normalize_oci_hardlinks.py").read_bytes()
    if hashlib.sha256(data).hexdigest() != ORIGINAL_SHA256:
        raise RuntimeError("Frozen failed normalizer drift; no candidate generation")
    source = data.decode()
    replacements = [
        ("        for original in source:\n",
         "        derived_members = iter(derived)\n        for original in source:\n"),
        ("            emitted = derived.next()\n",
         "            emitted = next(derived_members, None)\n"),
        ("        require(derived.next() is None, 'normalized archive added header')\n",
         "        require(next(derived_members, None) is None, 'normalized archive added header')\n"),
    ]
    for old, new in replacements:
        if source.count(old) != 1:
            raise RuntimeError("Candidate source replacement is not unique")
        source = source.replace(old, new, 1)
    compile(source, "unsubmitted_empty_layer_candidate", "exec")
    return source
