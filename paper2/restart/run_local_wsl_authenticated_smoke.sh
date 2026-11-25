#!/usr/bin/env bash
# Execute only from Ubuntu WSL.  This is a free, loopback-only transport and
# identity smoke test; it is intentionally not evidence of a farm deployment.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <new-output-directory>" >&2
  exit 2
fi

OUTPUT="$1"
if [[ -e "$OUTPUT" ]]; then
  echo "Output must not already exist: $OUTPUT" >&2
  exit 2
fi
# The clean Linux reproduction environment is the default free local runtime.
# An operator may override it (without changing the script) through
# PAPER2_FLOWER_VENV.  Activating it only adjusts this process's PATH.
if ! command -v flower-superlink >/dev/null; then
  FLOWER_VENV="${PAPER2_FLOWER_VENV:-$HOME/.venvs/paper2-clean-v1}"
  [[ -f "$FLOWER_VENV/bin/activate" ]] || {
    echo "Flower runtime not found; set PAPER2_FLOWER_VENV to a venv containing Flower 1.42.0." >&2
    exit 2
  }
  # shellcheck disable=SC1090
  source "$FLOWER_VENV/bin/activate"
fi
for command in openssl ssh-keygen flower-superlink flower-supernode flwr python; do
  command -v "$command" >/dev/null || { echo "Missing command: $command" >&2; exit 2; }
done
FLOWER_VERSION="$(python -c 'import flwr; print(flwr.__version__)')"
[[ "$FLOWER_VERSION" == "1.42.0" ]] || {
  echo "Expected frozen Flower 1.42.0, found $FLOWER_VERSION" >&2
  exit 2
}
export PAPER2_FLOWER_VERSION="$FLOWER_VERSION"

mkdir -p "$OUTPUT"
WORK="$(mktemp -d)"
declare -a PIDS=()
cleanup() {
  local status=$?
  for pid in "${PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done
  for pid in "${PIDS[@]:-}"; do wait "$pid" 2>/dev/null || true; done
  rm -rf "$WORK"
  exit "$status"
}
trap cleanup EXIT INT TERM

umask 077
mkdir -p "$WORK/keys" "$WORK/flwr"
openssl req -x509 -newkey rsa:2048 -nodes -sha256 -days 1 \
  -keyout "$WORK/server.key" -out "$WORK/ca.crt" \
  -subj "/CN=127.0.0.1" \
  -addext "subjectAltName=IP:127.0.0.1,DNS:localhost" >/dev/null 2>&1

cat > "$WORK/flwr/config.toml" <<EOF
[superlink]
default = "wsl-local"

[superlink.wsl-local]
address = "127.0.0.1:18000"
root-certificates = "$WORK/ca.crt"
EOF

export FLWR_HOME="$WORK/flwr"
flower-superlink --host 127.0.0.1 --port 18000 \
  --ssl-ca-certfile "$WORK/ca.crt" --ssl-certfile "$WORK/ca.crt" --ssl-keyfile "$WORK/server.key" \
  --enable-supernode-auth --disable-runtime-dependency-installation \
  > "$WORK/superlink.log" 2>&1 &
PIDS+=("$!")
sleep 3
kill -0 "${PIDS[0]}" || { cat "$WORK/superlink.log" >&2; exit 1; }

for index in $(seq 0 6); do
  # Flower 1.42 expects an ECDSA (not RSA/Ed25519) OpenSSH auth key.
  ssh-keygen -q -t ecdsa -b 384 -N '' -f "$WORK/keys/node-$index" -C "paper2-wsl-smoke-$index"
  flwr supernode register "$WORK/keys/node-$index.pub" wsl-local --name "wsl-local-node-$index" --format json \
    > "$WORK/register-$index.json"
done

for index in $(seq 0 6); do
  flower-supernode --root-certificates "$WORK/ca.crt" \
    --auth-supernode-private-key "$WORK/keys/node-$index" \
    --superlink 127.0.0.1:18000 --host 127.0.0.1 --port "$((19000 + index))" \
    --node-config "partition-id=$index num-partitions=7" \
    --ssl-ca-certfile "$WORK/ca.crt" --ssl-certfile "$WORK/ca.crt" --ssl-keyfile "$WORK/server.key" \
    --isolation process --max-wait-time 20 \
    > "$WORK/supernode-$index.log" 2>&1 &
  PIDS+=("$!")
done
sleep 8

for pid in "${PIDS[@]}"; do kill -0 "$pid" || { cat "$WORK/superlink.log" >&2; exit 1; }; done
flwr supernode list wsl-local --format json > "$WORK/nodes.json"

python - "$WORK" "$OUTPUT" <<'PY'
import hashlib
import json
import shutil
import sys
from pathlib import Path

work, output = map(Path, sys.argv[1:])
logs = output / "sanitized_logs"
logs.mkdir()
for file in [work / "superlink.log", work / "nodes.json", *(work / f"register-{i}.json" for i in range(7)), *(work / f"supernode-{i}.log" for i in range(7))]:
    shutil.copyfile(file, logs / file.name)
hashes = {file.name: hashlib.sha256(file.read_bytes()).hexdigest() for file in sorted(logs.iterdir())}
node_listing = json.loads((work / "nodes.json").read_text())
nodes = node_listing.get("nodes") if isinstance(node_listing, dict) else None
if node_listing.get("success") is not True or not isinstance(nodes, list):
    raise SystemExit("Flower node-list response is not a successful node collection")
node_ids = [node.get("node-id") for node in nodes if isinstance(node, dict)]
online_nodes = [node for node in nodes if isinstance(node, dict) and node.get("status") == "online"]
if len(nodes) != 7 or len(node_ids) != 7 or len(set(node_ids)) != 7 or len(online_nodes) != 7:
    raise SystemExit("Expected exactly seven distinct online SuperNodes")
receipt = {
    "status": "PASS_LOCAL_WSL_LOOPBACK_TLS_AUTH_SMOKE_NOT_DISTRIBUTED_DEPLOYMENT",
    "environment": "Ubuntu WSL2 on one physical laptop",
    "flower_version": __import__("os").environ["PAPER2_FLOWER_VERSION"],
    "transport": "TLS with ephemeral self-signed local CA",
    "supernode_authentication": "Flower CLI allowlist with seven ephemeral ECDSA keys",
    "separate_processes": 8,
    "registered_nodes_expected": 7,
    "online_nodes_observed_from_listing": len(online_nodes),
    "unique_node_ids_observed": len(set(node_ids)),
    "loopback_only": True,
    "private_keys_exported": False,
    "private_keys_retained_after_run": False,
    "secagg_executed": False,
    "central_dp_executed": False,
    "network_deployment_gate_passed": False,
    "claim_allowed": False,
    "sanitized_log_sha256": hashes,
    "interpretation": "This validates local process, TLS, and authenticated-node plumbing only. It is not a multi-host or farm deployment, an end-to-end SecAgg/DP execution, or a security certification.",
}
(output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
PY

echo "Local-only smoke receipt written to: $OUTPUT/receipt.json"
