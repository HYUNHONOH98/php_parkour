#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <linux/landlock.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#ifndef LANDLOCK_ACCESS_FS_REFER
#define LANDLOCK_ACCESS_FS_REFER (1ULL << 13)
#endif
#ifndef LANDLOCK_ACCESS_FS_TRUNCATE
#define LANDLOCK_ACCESS_FS_TRUNCATE (1ULL << 14)
#endif

/* Run only after chroot. The sole allowed persistent filesystem is the
 * private project tree; resolved /proc/PID/root escapes are not beneath it.
 * Kernel and GPU interfaces mounted in the private tree remain available.
 */
int main(int argc, char **argv) {
    if (argc < 2) return 2;
    int abi = syscall(SYS_landlock_create_ruleset, NULL, 0, LANDLOCK_CREATE_RULESET_VERSION);
    if (abi < 1) { perror("Landlock unavailable; refusing unconfined launch"); return 125; }
    __u64 access = LANDLOCK_ACCESS_FS_EXECUTE | LANDLOCK_ACCESS_FS_WRITE_FILE |
        LANDLOCK_ACCESS_FS_READ_FILE | LANDLOCK_ACCESS_FS_READ_DIR |
        LANDLOCK_ACCESS_FS_REMOVE_DIR | LANDLOCK_ACCESS_FS_REMOVE_FILE |
        LANDLOCK_ACCESS_FS_MAKE_CHAR | LANDLOCK_ACCESS_FS_MAKE_DIR |
        LANDLOCK_ACCESS_FS_MAKE_REG | LANDLOCK_ACCESS_FS_MAKE_SOCK |
        LANDLOCK_ACCESS_FS_MAKE_FIFO | LANDLOCK_ACCESS_FS_MAKE_BLOCK |
        LANDLOCK_ACCESS_FS_MAKE_SYM;
    if (abi >= 2) access |= LANDLOCK_ACCESS_FS_REFER;
    if (abi >= 3) access |= LANDLOCK_ACCESS_FS_TRUNCATE;
    struct landlock_ruleset_attr rules = {.handled_access_fs=access};
    int rs = syscall(SYS_landlock_create_ruleset, &rules, sizeof(rules), 0);
    if (rs < 0) { perror("landlock ruleset"); return 125; }
    int root = open("/", O_PATH | O_CLOEXEC);
    struct landlock_path_beneath_attr path = {.allowed_access=access,.parent_fd=root};
    if (root < 0 || syscall(SYS_landlock_add_rule,rs,LANDLOCK_RULE_PATH_BENEATH,&path,0)) {
        perror("landlock private root rule"); return 125;
    }
    close(root);
    if (prctl(PR_SET_NO_NEW_PRIVS,1,0,0,0) || syscall(SYS_landlock_restrict_self,rs,0)) {
        perror("landlock restrict"); return 125;
    }
    close(rs);
    setenv("PHP_LANDLOCK_CONFINED","1",1);
    execvp(argv[1],&argv[1]);
    perror("confined exec"); return 126;
}
