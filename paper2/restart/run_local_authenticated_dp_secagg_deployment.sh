#!/usr/bin/env bash
# Execute only from Ubuntu WSL. A single-laptop deployment-runtime study is
# useful operational evidence but is not a multi-host or privacy certification.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <new-output-directory>" >&2; exit 2
fi
OUTPUT="$1"
[[ ! -e "$OUTPUT" ]] || { echo "Output already exists: $OUTPUT" >&2; exit 2; }
REPO_ROOT="/mnt/d/Final Project/implementation_for _crop recommendation"
APP_SOURCE="$REPO_ROOT/paper2/paper2-dp-secagg-deploy"
DATA_ROOT="$REPO_ROOT/paper2/results/restart_public_data_v1"
VENV="${PAPER2_FLOWER_VENV:-$HOME/.venvs/paper2-clean-v1}"
[[ -f "$VENV/bin/activate" ]] || { echo "Missing Flower environment: $VENV" >&2; exit 2; }
# shellcheck disable=SC1090
source "$VENV/bin/activate"
for command in openssl ssh-keygen flower-superlink flower-supernode flwr timeout python; do
  command -v "$command" >/dev/null || { echo "Missing command: $command" >&2; exit 2; }
done
[[ "$(python -c 'import flwr; print(flwr.__version__)')" == "1.42.0" ]] || { echo "Flower 1.42.0 required" >&2; exit 2; }

# Refuse to execute an adapter whose source binding or frozen semantics changed.
# v2 binds the packaging correction (official CPU index and the compatible
# Python range) before any successful deployment run.  Earlier bindings stay
# immutable evidence for their earlier source snapshot.
PYTHONPATH="$REPO_ROOT" python -c 'import json; from pathlib import Path; from paper2.restart.prepare_authenticated_dp_secagg_adapter import validate; report=validate(); binding=json.loads(Path("'"$REPO_ROOT"'/paper2/results/restart_authenticated_dp_secagg_adapter_v2/adapter_binding.json").read_text()); assert report["status"] == binding["status"] == "PASS_AUTHENTICATED_DEPLOYMENT_ADAPTER_SEMANTIC_BINDING"; assert report["app_sources_sha256"] == binding["app_sources_sha256"]; assert report["frozen_study_manifest_sha256"] == binding["frozen_study_manifest_sha256"]'

mkdir -p "$OUTPUT"
WORK="$(mktemp -d)"; declare -a PIDS=(); FINISHED=false
record_interruption() {
  local status="$1"
  [[ -f "$OUTPUT/receipt.json" ]] && return
  mkdir -p "$OUTPUT/diagnostics"
  for file in "$WORK/superlink.log" "$WORK/nodes_before.json" "$WORK/nodes_after.json" "$WORK/run_stdout.log" "$WORK/run_stderr.log"; do
    [[ -f "$file" ]] && cp "$file" "$OUTPUT/diagnostics/"
  done
  python - "$OUTPUT" "$status" <<'PY'
import json, sys
from pathlib import Path
output, status = Path(sys.argv[1]), sys.argv[2]
(output / "receipt.json").write_text(json.dumps({
  "status": "ABORTED_LOCAL_AUTHENTICATED_DP_SECAGG_DEPLOYMENT_NO_AUTOMATIC_RETRY",
  "execution_scope": "single_laptop_wsl_loopback", "reason": f"runner_exit_{status}",
  "tls_enabled": True, "cli_managed_supernode_auth": True, "secagg_completed": False,
  "central_dp_completed": False, "network_deployment_gate_passed": False,
  "privacy_claim_allowed": False,
  "interpretation": "The local deployment did not complete; this is failure evidence only."}, indent=2) + "\n")
PY
}
cleanup() {
  local status=$?
  trap - EXIT INT TERM
  [[ "$FINISHED" == true ]] || record_interruption "$status"
  for pid in "${PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done
  for pid in "${PIDS[@]:-}"; do wait "$pid" 2>/dev/null || true; done
  rm -rf "$WORK"
  exit "$status"
}
trap cleanup EXIT INT TERM
umask 077
mkdir -p "$WORK/keys" "$WORK/flwr" "$WORK/app"
cp -R "$APP_SOURCE/." "$WORK/app/"
openssl req -x509 -newkey rsa:2048 -nodes -sha256 -days 1 \
  -keyout "$WORK/server.key" -out "$WORK/ca.crt" -subj "/CN=127.0.0.1" \
  -addext "subjectAltName=IP:127.0.0.1,DNS:localhost" >/dev/null 2>&1
cat > "$WORK/flwr/config.toml" <<EOF
[superlink]
default = "wsl-local"
[superlink.wsl-local]
address = "127.0.0.1:18100"
root-certificates = "$WORK/ca.crt"
EOF
export FLWR_HOME="$WORK/flwr"
flower-superlink --host 127.0.0.1 --port 18100 \
  --ssl-ca-certfile "$WORK/ca.crt" --ssl-certfile "$WORK/ca.crt" --ssl-keyfile "$WORK/server.key" \
  --enable-supernode-auth > "$WORK/superlink.log" 2>&1 &
PIDS+=("$!"); sleep 3; kill -0 "${PIDS[0]}" || { cat "$WORK/superlink.log" >&2; exit 1; }

ROSTER=(Auzeville Copenhagen+Taastrup Grignon Jyndevad Kassel Reading Sanmarcoargentano)
for index in $(seq 0 6); do
  ssh-keygen -q -t ecdsa -b 384 -N '' -f "$WORK/keys/node-$index" -C "paper2-dp-deploy-$index"
  flwr supernode register "$WORK/keys/node-$index.pub" wsl-local --name "wsl-dp-node-$index" --format json > "$WORK/register-$index.json"
done
for index in $(seq 0 6); do
  flower-supernode --root-certificates "$WORK/ca.crt" --auth-supernode-private-key "$WORK/keys/node-$index" \
    --superlink 127.0.0.1:18100 --host 127.0.0.1 --port "$((19100 + index))" \
    --node-config "partition-id=$index num-partitions=7 client-id='${ROSTER[$index]}' data-root='$DATA_ROOT' fold-id='outer_0'" \
    --ssl-ca-certfile "$WORK/ca.crt" --ssl-certfile "$WORK/ca.crt" --ssl-keyfile "$WORK/server.key" \
    --isolation subprocess --max-wait-time 30 > "$WORK/supernode-$index.log" 2>&1 &
  PIDS+=("$!")
done
sleep 8
for pid in "${PIDS[@]}"; do kill -0 "$pid" || { cat "$WORK/superlink.log" >&2; exit 1; }; done
flwr supernode list wsl-local --format json > "$WORK/nodes_before.json"

set +e
timeout 1800 flwr run "$WORK/app" wsl-local --stream > "$WORK/run_stdout.log" 2> "$WORK/run_stderr.log"
RUN_CODE=$?
set -e
flwr supernode list wsl-local --format json > "$WORK/nodes_after.json" || true
printf '%s\n' "$RUN_CODE" > "$WORK/run_returncode.txt"

python - "$WORK" "$OUTPUT" <<'PY'
import hashlib, json, os, shutil, sys
from pathlib import Path
work, output = map(Path, sys.argv[1:])
logs = output / "sanitized_logs"; logs.mkdir()
files = [work / "superlink.log", work / "nodes_before.json", work / "nodes_after.json", work / "run_stdout.log", work / "run_stderr.log", work / "run_returncode.txt"]
files += [work / f"register-{i}.json" for i in range(7)] + [work / f"supernode-{i}.log" for i in range(7)]
for source in files:
    if source.is_file(): shutil.copyfile(source, logs / source.name)
before = json.loads((work / "nodes_before.json").read_text())
nodes = before.get("nodes", []) if isinstance(before, dict) else []
online = [item for item in nodes if isinstance(item, dict) and item.get("status") == "online"]
ids = [item.get("node-id") for item in nodes if isinstance(item, dict)]
returncode = int((work / "run_returncode.txt").read_text().strip())
run_log = "\n".join(
    (work / name).read_text(encoding="utf-8", errors="replace")
    for name in ("run_stdout.log", "run_stderr.log")
    if (work / name).is_file()
)
failure_patterns = {
    "serverapp_exception": r"ServerApp raised an exception",
    "clientapp_exception": r"ClientApp raised an exception",
    "runtime_dependency_installation": r"RuntimeDependencyInstallationError",
    "runtime_dependency_failure": r"Runtime installation of app dependencies failed",
    "flower_nonzero_exit": r"ERROR\s*:\s*Exit Code:\s*[1-9]",
}
matched_failures = [name for name, pattern in failure_patterns.items() if __import__("re").search(pattern, run_log)]
semantic_passed = returncode == 0 and not matched_failures
semantic_log_audit = {
  "passed": semantic_passed,
  "matched_failure_signatures": matched_failures,
  "evidence": "run_stdout.log and run_stderr.log were inspected; CLI return code alone is not accepted as a completion signal.",
}
(work / "run_semantic_status.txt").write_text("PASS\n" if semantic_passed else "FAIL\n", encoding="utf-8")
shutil.copyfile(work / "run_semantic_status.txt", logs / "run_semantic_status.txt")
receipt = {
  "status": "PASS_LOCAL_AUTHENTICATED_DP_SECAGG_DEPLOYMENT_RUNTIME_NOT_CERTIFIED" if semantic_passed and len(online) == len(set(ids)) == 7 else "FAIL_LOCAL_AUTHENTICATED_DP_SECAGG_DEPLOYMENT_RUNTIME",
  "execution_scope": "single_laptop_wsl_loopback", "flower_version": "1.42.0", "rounds_requested": 20,
  "run_returncode": returncode, "online_nodes_before_run": len(online), "unique_node_ids_before_run": len(set(ids)),
  "semantic_log_audit": semantic_log_audit,
  "clientapp_isolation": "subprocess",
  "tls_enabled": True, "cli_managed_supernode_auth": True, "private_keys_exported": False,
  "private_keys_retained_after_run": False, "outer_test_inference": False, "network_deployment_gate_passed": False,
  "privacy_claim_allowed": False, "independent_security_review": "PENDING",
  "log_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(logs.iterdir())},
  "interpretation": "Execution result is local deployment-runtime evidence only. It is not an external multi-host deployment, a privacy certification, or an independent audit.",
}
(output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt, indent=2))
PY
FINISHED=true
[[ "$(<"$WORK/run_semantic_status.txt")" == "PASS" ]] || exit 1
