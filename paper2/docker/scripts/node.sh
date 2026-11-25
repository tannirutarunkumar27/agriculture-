#!/usr/bin/env sh
set -eu
: "${PARTITION_ID:?}" "${CLIENT_ID:?}"
umask 077
mkdir -p /runtime/public
if [ ! -f /node-key/node ]; then
  ssh-keygen -q -t ecdsa -b 384 -N '' -f /node-key/node -C "paper2-docker-node-${PARTITION_ID}"
fi
cp /node-key/node.pub "/runtime/public/node-${PARTITION_ID}.pub"
deadline=$(( $(date +%s) + 300 ))
while [ ! -f /runtime/registered ]; do
  [ "$(date +%s)" -lt "$deadline" ] || { echo 'registration timeout' >&2; exit 1; }
  sleep 1
done
exec flower-supernode --root-certificates /runtime/ca.crt --auth-supernode-private-key /node-key/node \
  --superlink superlink:18100 --host 0.0.0.0 --port 19100 \
  --node-config "partition-id=${PARTITION_ID} num-partitions=7 client-id='${CLIENT_ID}' data-root='/data' fold-id='outer_0'" \
  --ssl-ca-certfile /runtime/ca.crt --ssl-certfile /runtime/ca.crt --ssl-keyfile /runtime/server.key \
  --isolation subprocess --max-wait-time 30
