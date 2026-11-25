#!/usr/bin/env bash
# Containerized seven-process local deployment.  It creates one SuperLink and
# seven authenticated SuperNode containers on one Docker host.  The evidence is
# deliberately restricted to local container/process/network isolation.
set -euo pipefail
if [[ $# -ne 1 ]]; then echo "Usage: $0 <new-output-directory>" >&2; exit 2; fi
OUTPUT="$(realpath -m "$1")"
[[ ! -e "$OUTPUT" ]] || { echo "Output already exists: $OUTPUT" >&2; exit 2; }
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
COMPOSE_DIR="$ROOT/paper2/docker"
DATA="$ROOT/paper2/results/restart_public_data_v1"
[[ -d "$DATA" ]] || { echo "Missing frozen public deployment data: $DATA" >&2; exit 2; }
command -v docker >/dev/null || { echo 'Docker CLI is required' >&2; exit 2; }
docker info >/dev/null || { echo 'Docker engine is unavailable' >&2; exit 2; }
mkdir -p "$OUTPUT/container_logs"
export PAPER2_DOCKER_DATA_DIR="$DATA"
export COMPOSE_PROJECT_NAME="paper2docker$RANDOM$RANDOM"
cleanup() { (cd "$COMPOSE_DIR" && docker compose down --volumes --remove-orphans) >/dev/null 2>&1 || true; }
trap cleanup EXIT INT TERM
set +e
(cd "$COMPOSE_DIR" && docker compose up --abort-on-container-exit --exit-code-from coordinator) 2>&1 | tee "$OUTPUT/container_logs/compose.log"
code=${PIPESTATUS[0]}
set -e
coordinator="$(cd "$COMPOSE_DIR" && docker compose ps -aq coordinator || true)"
if [[ -n "$coordinator" ]]; then
  for name in nodes_before.json nodes_after.json run_stdout.log run_stderr.log run_returncode.txt register-0.json register-1.json register-2.json register-3.json register-4.json register-5.json register-6.json; do
    docker cp "$coordinator:/runtime/$name" "$OUTPUT/container_logs/$name" 2>/dev/null || true
  done
fi
python3 - "$OUTPUT" "$code" <<'PY'
import hashlib, json, re, sys
from pathlib import Path
out, code = Path(sys.argv[1]), int(sys.argv[2]); logs = out / 'container_logs'
text = '\n'.join((path.read_text(errors='replace') for path in logs.glob('*.log')))
bad = [label for label, pattern in {
    'serverapp_exception': r'ServerApp raised an exception',
    'clientapp_exception': r'ClientApp raised an exception',
    'runtime_dependency_failure': r'Runtime (installation of app )?dependencies failed',
}.items() if re.search(pattern, text)]
receipt = {
  'status': 'PASS_LOCAL_DOCKER_AUTHENTICATED_DP_SECAGG_RUNTIME_NOT_CERTIFIED' if code == 0 and not bad else 'FAIL_LOCAL_DOCKER_AUTHENTICATED_DP_SECAGG_RUNTIME',
  'execution_scope': 'single_laptop_docker_bridge_network', 'containerized_supernodes': 7,
  'containerized_superlink': 1, 'rounds_requested': 20, 'run_returncode': code,
  'tls_enabled': True, 'cli_managed_supernode_auth': True, 'clientapp_isolation': 'subprocess',
  'network_deployment_gate_passed': False, 'privacy_claim_allowed': False,
  'independent_security_review': 'PENDING', 'key_custody_claim_allowed': False,
  'semantic_log_audit': {'passed': code == 0 and not bad, 'matched_failure_signatures': bad},
  'log_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(logs.iterdir()) if p.is_file()},
  'interpretation': 'This is one Docker host with bridged containers. It is evidence of local container/process/network isolation only, not independent hosts, farm deployment, or privacy/security certification.'
}
(out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, indent=2))
PY
[[ "$code" -eq 0 ]]
