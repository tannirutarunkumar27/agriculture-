#!/usr/bin/env sh
set -eu
umask 077
mkdir -p /runtime/flwr /runtime/public
openssl req -x509 -newkey rsa:2048 -nodes -sha256 -days 1 \
  -keyout /runtime/server.key -out /runtime/ca.crt -subj /CN=superlink \
  -addext 'subjectAltName=DNS:superlink,DNS:localhost' >/dev/null 2>&1
cat > /runtime/flwr/config.toml <<'EOF'
[superlink]
default = "docker-local"
[superlink.docker-local]
address = "superlink:18100"
root-certificates = "/runtime/ca.crt"
EOF
touch /runtime/bootstrap-ready
echo 'bootstrap complete'
