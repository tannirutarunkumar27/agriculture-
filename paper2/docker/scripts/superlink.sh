#!/usr/bin/env sh
set -eu
test -f /runtime/bootstrap-ready
exec flower-superlink --host 0.0.0.0 --port 18100 \
  --ssl-ca-certfile /runtime/ca.crt --ssl-certfile /runtime/ca.crt --ssl-keyfile /runtime/server.key \
  --enable-supernode-auth
