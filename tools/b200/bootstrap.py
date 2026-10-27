"""Install the captured Linux/x86_64 runtime entirely inside this checkout.

Run with a project-local Conda bootstrap Python; never clones another project.
"""
from pathlib import Path
import argparse,hashlib,json,os,shutil,subprocess,sys,tarfile
ROOT=Path(__file__).resolve().parents[2];RUNTIME=ROOT/'.runtime/b200';BUNDLE=ROOT/'repro/b200_20261027/runtime'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--accept-isaac-eula',action='store_true',help='Confirm you have read and accept the NVIDIA Isaac Sim EULA')
    p.add_argument('--plan',action='store_true',help='Print steps without creating/installing a runtime')
    args=p.parse_args()
    prefix=RUNTIME/'env';bootstrap=ROOT/'.runtime/conda-bootstrap/bin/conda'
    local=['isaaclab','isaaclab_assets','isaaclab_tasks','isaaclab_rl']
    commands=[
      [str(bootstrap),'create','--yes','--prefix',str(prefix),'--file',str(BUNDLE/'conda-explicit-linux-64.txt')],
      [str(prefix/'bin/python'),'-m','pip','install','--no-deps','--index-url','https://pypi.org/simple','--extra-index-url','https://pypi.nvidia.com','--extra-index-url','https://download.pytorch.org/whl/cu128','-r',str(BUNDLE/'requirements-linux-64.txt')],
      *[[str(prefix/'bin/python'),'-m','pip','install','--no-deps','--no-build-isolation','-e',str(RUNTIME/'IsaacLab/source'/s)]for s in local],
      [str(prefix/'bin/python'),'-m','pip','install','--no-deps','--no-build-isolation','-e',str(ROOT/'thirdparty/holosoma/src/holosoma'),'-e',str(ROOT)],
      [sys.executable,str(ROOT/'tools/b200/build_sandbox.py')],
    ]
    if args.plan:
        print(json.dumps(dict(commands=commands,python='3.11.17',isaaclab_source='Bundled exact source snapshot',needs_project_local_bootstrap=True),indent=2));return
    assert Path(sys.executable).resolve().is_relative_to(ROOT),'Use .runtime/conda-bootstrap/bin/python, not an external project environment'
    assert bootstrap.is_file(),'Install a Conda bootstrap under .runtime/conda-bootstrap first'
    assert not prefix.exists(),'Refusing to overwrite an existing runtime (including a live training environment)'
    assert args.accept_isaac_eula,'Explicit --accept-isaac-eula is required for a fresh Isaac Sim installation'
    for name in ['cache','config','tmp','run','secrets']: (RUNTIME/name).mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,CONDA_PKGS_DIRS=str(ROOT/'.runtime/conda-pkgs'),CONDA_ENVS_PATH=str(ROOT/'.runtime/envs'),
      CONDARC=str(RUNTIME/'condarc'),PIP_CACHE_DIR=str(RUNTIME/'cache/pip'),TMPDIR=str(RUNTIME/'tmp'),
      XDG_CACHE_HOME=str(RUNTIME/'cache'),XDG_CONFIG_HOME=str(RUNTIME/'config'),PYTHONNOUSERSITE='1',OMNI_KIT_ACCEPT_EULA='YES')
    (RUNTIME/'condarc').write_text('auto_activate_base: false\n')
    manifest=json.loads((ROOT/'repro/b200_20261027/manifest.json').read_text())
    for relative,item in manifest['assets'].items():
        f=ROOT/relative;assert f.is_file()and hashlib.sha256(f.read_bytes()).hexdigest()==item['sha256'],relative
    lab=RUNTIME/'IsaacLab';lab.mkdir()
    with tarfile.open(BUNDLE/'isaaclab-source.tar.gz')as tar:
        for member in tar.getmembers():
            assert not member.issym()and not member.islnk()and(lab/member.name).resolve().is_relative_to(lab)
        tar.extractall(lab,filter='data')
    for command in commands:subprocess.run(command,cwd=ROOT,env=env,check=True)
    shutil.copy2(ROOT/'tools/b200/bin/confine-linux-x86_64',RUNTIME/'host-tools/confine')
    (RUNTIME/'host-tools/confine').chmod(0o755)
    for name in ['run.sh','namespace.sh','inside.sh']:
        f=ROOT/'tools/b200'/name;lines=f.read_text().splitlines()
        lines[0]='#!'+str(RUNTIME/'host-tools/bash');f.write_text('\n'.join(lines)+'\n');f.chmod(0o755)
    (RUNTIME/'isaac-sim-eula.accepted').write_text('ACCEPT\n')
    print('Installed project-local runtime; run tools/b200/run.sh python tools/b200/doctor.py',flush=True)

if __name__=='__main__':main()
