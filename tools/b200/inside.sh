#!/NHNHOME/haecheolpark/php_parkour/.runtime/b200/host-tools/bash
set -euo pipefail
PHP_ROOT=$1
shift
source "$PHP_ROOT/tools/b200/activate.sh"
export PHP_FILESYSTEM_ISOLATED=1
if [[ ${1:-} == bootstrap-python ]]; then
    shift
    exec "$PHP_ROOT/.runtime/b200/host-tools/confine" "$PHP_ROOT/.runtime/conda-bootstrap/bin/python" "$@"
fi
if [[ ${1:-} == shell ]]; then
    shift
    exec "$PHP_ROOT/.runtime/b200/host-tools/confine" "$PHP_ROOT/.runtime/b200/host-tools/bash" "$@"
fi
if [[ ${1:-} == python ]]; then
    shift
    exec "$PHP_ROOT/.runtime/b200/host-tools/confine" "$PHP_ROOT/.runtime/b200/env/bin/python" "$@"
fi
exec "$PHP_ROOT/.runtime/b200/host-tools/confine" "$PHP_ROOT/.runtime/b200/env/bin/python" "$PHP_ROOT/tools/b200/train.py" "$@"
