"""Record bounded, anonymous read-only requests for availability evidence."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("destination")
    parser.add_argument("--round-list", action="store_true")
    args = parser.parse_args()
    path = (ROOT / args.destination).resolve()
    if not path.is_relative_to(ROOT) or path.exists():
        raise ValueError("destination must be new and within this evidence batch")
    if not args.url.startswith("https://"):
        raise ValueError("HTTPS only")
    extra = []
    if args.round_list:
        if not re.fullmatch(r"https://repo-prod\.prod\.sagebase\.org/repo/v1/evaluation/[0-9]+/round/list", args.url):
            raise ValueError("POST is restricted to the documented read-only round/list API")
        extra = ["--header", "Content-Type: application/json", "--data", "{}"]
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".partial")
    receipt = path.with_name(path.name + ".source.json")
    if part.exists() or receipt.exists():
        raise ValueError("refusing to overwrite an earlier acquisition")
    started = datetime.now(timezone.utc).isoformat()
    result = subprocess.run(
        ["curl", "--location", "--silent", "--show-error", "--max-time", "45",
         "--max-filesize", str(8 * 1024**2), "--output", str(part),
         "--write-out", "%{json}", *extra, args.url],
        capture_output=True, text=True, timeout=60,
    )
    try:
        raw = json.loads(result.stdout)
        network = {k: raw[k] for k in ("http_code", "content_type", "url_effective",
                   "size_download", "time_total", "ssl_verify_result") if k in raw}
    except json.JSONDecodeError:
        network = {"parse_failed": True}
    success = result.returncode == 0 and network.get("http_code") == 200
    saved = path if success else path.with_name(path.name + ".failed_response")
    if part.exists():
        part.rename(saved)
    record = dict(url=args.url, destination=args.destination, started_utc=started,
                  received_utc=datetime.now(timezone.utc).isoformat(), success=success,
                  returncode=result.returncode, network=network, stderr=result.stderr,
                  response_path=str(saved.relative_to(ROOT)) if saved.exists() else None,
                  bytes=saved.stat().st_size if saved.exists() else None,
                  sha256=hashlib.sha256(saved.read_bytes()).hexdigest() if saved.exists() else None,
                  method="POST" if args.round_list else "GET", read_only_request=True,
                  request_body={} if args.round_list else None, anonymous=True,
                  untrusted_external_reference=True,
                  project_scientific_validation_pass=False)
    receipt.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: record[k] for k in ("destination", "success", "bytes", "sha256")}, ensure_ascii=False))
    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
