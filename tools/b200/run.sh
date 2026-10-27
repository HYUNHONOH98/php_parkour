#!/NHNHOME/haecheolpark/php_parkour/.runtime/b200/host-tools/bash
set -euo pipefail
PHP_ROOT="$(cd "${BASH_SOURCE[0]%/*}/../.." && pwd)"
PHP_TOOLS="$PHP_ROOT/.runtime/b200/host-tools"
exec "$PHP_TOOLS/unshare" --user --map-root-user --mount --ipc --fork --kill-child "$PHP_TOOLS/bash" "$PHP_ROOT/tools/b200/namespace.sh" "$PHP_ROOT" "$@"
