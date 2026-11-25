#!/usr/bin/env sh
set -eu
export FLWR_HOME=/runtime/flwr
deadline=$(( $(date +%s) + 300 ))
while [ "$(find /runtime/public -name 'node-*.pub' -type f 2>/dev/null | wc -l)" -ne 7 ]; do
  [ "$(date +%s)" -lt "$deadline" ] || { echo 'public-key collection timeout' >&2; exit 1; }
  sleep 1
done
for index in 0 1 2 3 4 5 6; do
  flwr supernode register "/runtime/public/node-${index}.pub" docker-local --name "docker-dp-node-${index}" --format json > "/runtime/register-${index}.json"
done
touch /runtime/registered
for attempt in $(seq 1 90); do
  flwr supernode list docker-local --format json > /runtime/nodes_before.json || true
  if python - <<'PY'
import json
try:
    nodes = json.load(open('/runtime/nodes_before.json')).get('nodes', [])
    raise SystemExit(0 if len([node for node in nodes if node.get('status') == 'online']) == 7 else 1)
except Exception:
    raise SystemExit(1)
PY
  then break; fi
  [ "$attempt" -lt 90 ] || { echo 'online-node timeout' >&2; exit 1; }
  sleep 2
done
set +e
timeout 1800 flwr run /opt/paper2-app docker-local --stream > /runtime/run_stdout.log 2> /runtime/run_stderr.log
status=$?
set -e
printf '%s\n' "$status" > /runtime/run_returncode.txt
flwr supernode list docker-local --format json > /runtime/nodes_after.json || true
cat /runtime/run_stdout.log
cat /runtime/run_stderr.log >&2
exit "$status"
