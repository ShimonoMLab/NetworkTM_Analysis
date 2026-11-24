#!/usr/bin/env bash
# Disposable Linux user/network namespaces. No host network configuration is changed.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
python_bin=${PYTHON_BIN:-python3}
output=$(realpath -m "${1:-$here/virtual_results.csv}")
export python_bin output here
unshare -Urn bash -euc '
  unshare -n sleep 300 & peer=$!
  trap "kill $peer 2>/dev/null || true; wait $peer 2>/dev/null || true" EXIT
  for i in $(seq 1 100); do [ "$(readlink /proc/$peer/ns/net)" != "$(readlink /proc/self/ns/net)" ] && break; sleep .02; done
  test "$(readlink /proc/$peer/ns/net)" != "$(readlink /proc/self/ns/net)"
  ip link add veth_tx type veth peer name veth_rx
  ip link set veth_rx netns "$peer"
  ip addr add 10.240.0.1/30 dev veth_tx; ip link set veth_tx up
  nsenter -t "$peer" -n ip addr add 10.240.0.2/30 dev veth_rx
  nsenter -t "$peer" -n ip link set veth_rx up
  "$python_bin" "$here/virtual_link.py" --peer "$peer" --output "$output"
'
