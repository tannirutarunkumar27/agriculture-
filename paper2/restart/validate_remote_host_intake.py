"""Validate non-secret remote-host intake before any remote launcher is used."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

LOOPBACK = {"localhost", "127.0.0.1", "::1"}
RESERVED_SUFFIXES = (".example", ".example.com", ".example.net", ".example.org", ".invalid", ".test")


def evaluate(config: dict, *, key_exists: bool | None = None) -> dict:
    errors = []
    approval = str(config.get("status", "")).strip()
    host = str(config.get("ssh_host", "")).strip()
    remote_root = str(config.get("remote_root", "")).strip()
    key = str(config.get("ssh_key_path", "")).strip()
    try:
        gpu_count = int(config.get("gpu_count", 0))
        workers = int(config.get("workers", 0))
    except (TypeError, ValueError):
        gpu_count = workers = 0
        errors.append("gpu_count_and_workers_must_be_integers")
    try:
        ssh_port = int(config.get("ssh_port", 22))
    except (TypeError, ValueError):
        ssh_port = 0
        errors.append("ssh_port_must_be_an_integer")
    host_name = host.rsplit("@", 1)[-1].strip("[]").lower()
    if approval != "APPROVED_FOR_PREFLIGHT":
        errors.append("explicit_preflight_approval_required")
    if (not host or "<" in host or ">" in host or host_name in LOOPBACK
            or host_name.endswith(RESERVED_SUFFIXES)
            or not re.fullmatch(r"[A-Za-z0-9._-]+@[A-Za-z0-9.-]+", host)):
        errors.append("real_nonloopback_ssh_host_required")
    root_parts = Path(remote_root).parts
    if (not remote_root.startswith("/") or "<" in remote_root or ">" in remote_root
            or not re.fullmatch(r"/[A-Za-z0-9._/-]+", remote_root)
            or ".." in root_parts):
        errors.append("absolute_posix_remote_root_required")
    if not key or "<" in key or ">" in key:
        errors.append("concrete_ssh_key_path_required")
    if gpu_count < 1 or gpu_count > 8:
        errors.append("gpu_count_must_be_between_1_and_8")
    if workers < 1 or workers > gpu_count:
        errors.append("workers_must_be_between_1_and_gpu_count")
    if ssh_port < 1 or ssh_port > 65535:
        errors.append("ssh_port_must_be_between_1_and_65535")
    if key_exists is False:
        errors.append("ssh_private_key_file_not_found")
    return {
        "status": "REMOTE_HOST_INTAKE_READY_FOR_PREFLIGHT" if not errors else "REMOTE_HOST_INTAKE_BLOCKED",
        "preflight_allowed": not errors,
        "errors": sorted(set(errors)),
        "remote_execution_performed": False,
        "model_training": False,
        "model_inference": False,
    }


def main(config_path: Path, output: Path | None = None) -> int:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    key = Path(str(config.get("ssh_key_path", ""))).expanduser()
    report = evaluate(config, key_exists=key.is_file())
    if output is not None:
        output = Path(output).resolve()
        results = Path(__file__).resolve().parents[1] / "results"
        if output.exists() or results not in output.parents:
            raise ValueError("Output must be a new Paper 2 results directory")
        output.mkdir(parents=True)
        (output / "intake.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["preflight_allowed"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    raise SystemExit(main(args.config, args.output))
