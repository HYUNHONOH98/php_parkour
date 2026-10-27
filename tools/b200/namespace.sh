#!/NHNHOME/haecheolpark/php_parkour/.runtime/b200/host-tools/bash
set -euo pipefail
PHP_ROOT=$1
shift
PHP_RUNTIME="$PHP_ROOT/.runtime/b200"
PHP_FS="$PHP_RUNTIME/rootfs"
PHP_TOOLS="$PHP_RUNTIME/host-tools"
export PATH="$PHP_TOOLS"
"$PHP_TOOLS/mount" --make-rprivate /
# Preserve the absolute installation prefix within the private filesystem.
"$PHP_TOOLS/mount" --bind "$PHP_ROOT" "$PHP_FS$PHP_ROOT"
# This container masks parts of /proc and rejects a fresh procfs mount. Bind
# its existing kernel view; Landlock below prevents proc-root symlink escapes.
"$PHP_TOOLS/mount" --rbind /proc "$PHP_FS/proc"
"$PHP_TOOLS/mount" --rbind /sys "$PHP_FS/sys"
"$PHP_TOOLS/mount" -o remount,bind,ro "$PHP_FS/sys"
"$PHP_TOOLS/mount" -t tmpfs -o mode=1777,nosuid,nodev tmpfs "$PHP_FS/dev/shm"
"$PHP_TOOLS/mount" --bind "$PHP_RUNTIME/tmp" "$PHP_FS/tmp"
# Only OS device interfaces are shared; no host directory or executable tree.
for device in /dev/null /dev/zero /dev/random /dev/urandom /dev/nvidia[0-9]* /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools /dev/nvidia-caps/*; do
    if [[ -c "$device" ]]; then
        "$PHP_TOOLS/mkdir" -p "${PHP_FS}${device%/*}"
        "$PHP_TOOLS/touch" "${PHP_FS}${device}"
        "$PHP_TOOLS/mount" --bind "$device" "${PHP_FS}${device}"
    fi
done
cd "$PHP_FS"
# Dropping capabilities after chroot prevents an application from undoing it.
exec "$PHP_TOOLS/chroot" "$PHP_FS" "$PHP_TOOLS/setpriv" --bounding-set=-all --inh-caps=-all --ambient-caps=-all --no-new-privs "$PHP_TOOLS/bash" "$PHP_ROOT/tools/b200/inside.sh" "$PHP_ROOT" "$@"
