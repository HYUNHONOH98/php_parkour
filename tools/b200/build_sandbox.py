"""Copy the small OS runtime used by the project into its private filesystem.

Experiment processes see this filesystem plus their own project and GPU/kernel
interfaces. They cannot see MinReward_server or any other host project.
"""
from pathlib import Path
import hashlib,json,os,shutil,stat

ROOT=Path(__file__).resolve().parents[2]
RUNTIME=ROOT/'.runtime/b200';FS=RUNTIME/'rootfs';TOOLS=RUNTIME/'host-tools'
FS.mkdir(parents=True,exist_ok=True);TOOLS.mkdir(exist_ok=True)
inventory=[]
def copy(source,dest):
    source=Path(source);dest=Path(dest)
    if not source.is_file():return
    dest.parent.mkdir(parents=True,exist_ok=True)
    # Dereference host symlinks: no private library may escape to a host path.
    if not dest.exists() or dest.stat().st_size!=source.stat().st_size:
        shutil.copy2(source.resolve(),dest)
    inventory.append({'source':str(source),'private':str(dest.relative_to(ROOT)),'bytes':dest.stat().st_size})

def library(source,dest):
    """Keep driver aliases on one inode, with every target vendored locally.

    Flattening libcuda.so and libcuda.so.1 into distinct copies lets Torch and
    Warp load two CUDA drivers into one process, breaking the second cuInit.
    """
    source=Path(source);dest=Path(dest)
    if not source.is_file():return
    target=source.resolve()
    if source.name==target.name:
        copy(source,dest)
        return
    canonical=dest.parent/target.name
    copy(target,canonical)
    if dest.is_symlink() or dest.exists():dest.unlink()
    dest.symlink_to(canonical.name)
    inventory.append({'source':str(source),'private':str(dest.relative_to(ROOT)),'symlink':canonical.name})

names=['bash','sh','env','unshare','mount','chroot','setpriv','nvidia-smi','which','readlink','realpath','dirname','mkdir','touch','cat','grep','sed','uname','id','getconf','date','sleep','tee','head','tail','wc','ls','ps','find','xargs','sort','tr','test','true','false','ldd','lscpu']
for name in names:
    source=shutil.which(name)
    if source:
        copy(source,TOOLS/name);copy(source,FS/'usr/bin'/name)
for name in ['bash','sh']:
    copy(TOOLS/name,FS/'bin'/name)
for source in Path('/usr/lib/x86_64-linux-gnu').glob('*.so*'):
    if source.is_file():library(source,FS/'usr/lib/x86_64-linux-gnu'/source.name)
for source in Path('/lib64').glob('*'):
    if source.is_file():copy(source,FS/'lib64'/source.name)
for source in Path('/usr/local/nvidia/lib64').glob('*.so*'):
    if source.is_file():library(source,FS/'usr/local/nvidia/lib64'/source.name)
if not (FS/'lib').exists():(FS/'lib').symlink_to('usr/lib')
for name in ['os-release','hosts','resolv.conf','nsswitch.conf','localtime','ld.so.cache','services','protocols']:
    copy(Path('/etc')/name,FS/'etc'/name)
certs=Path('/etc/ssl/certs')
if certs.is_dir():
    for source in certs.iterdir():
        if source.is_file():copy(source,FS/'etc/ssl/certs'/source.name)
for source in Path('/usr/share/vulkan/icd.d').glob('*'):copy(source,FS/'usr/share/vulkan/icd.d'/source.name)
for directory in [FS/str(ROOT).lstrip('/'),FS/'proc',FS/'sys',FS/'dev/shm',FS/'tmp',FS/'run',FS/'home/beomjoon',FS/'etc']:
    directory.mkdir(parents=True,exist_ok=True)
for name,target in [('fd','/proc/self/fd'),('stdin','/proc/self/fd/0'),('stdout','/proc/self/fd/1'),('stderr','/proc/self/fd/2')]:
    path=FS/'dev'/name
    if not path.exists() and not path.is_symlink():path.symlink_to(target)
(FS/'etc/passwd').write_text('beomjoon:x:0:0:Project runtime:/home/beomjoon:/bin/bash\n')
(FS/'etc/group').write_text('beomjoon:x:0:\n')
(RUNTIME/'sandbox_inventory.json').write_text(json.dumps(inventory,indent=2))
print(json.dumps({'rootfs':str(FS),'copied_files':len(inventory)},indent=2))
