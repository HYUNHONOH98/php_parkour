"""Check the independent runtime without requiring W&B credentials."""
from pathlib import Path
import importlib.metadata as metadata,importlib.util,json,os,subprocess,sys
ROOT=Path(__file__).resolve().parents[2]
assert os.environ.get('PHP_FILESYSTEM_ISOLATED')=='1'and os.environ.get('PHP_LANDLOCK_CONFINED')=='1'
assert Path(sys.executable).is_relative_to(ROOT)
modules={name:importlib.util.find_spec(name).origin for name in ['torch','holosoma','wbt_training','isaacsim','isaaclab']}
assert all(Path(p).is_relative_to(ROOT)for p in modules.values())
check=subprocess.run([sys.executable,'-m','pip','check'],text=True,capture_output=True)
expected='isaacsim-kernel 5.1.0.0 has requirement typing_extensions==4.12.2, but you have typing-extensions 4.15.0.'
assert check.returncode==0 or check.returncode==1 and check.stdout.strip()==expected,check.stdout
import torch
assert torch.cuda.is_available()
report=dict(python=sys.executable,modules=modules,gpu=torch.cuda.get_device_name(0),
 packages={n:metadata.version(n)for n in ['torch','isaacsim','isaaclab','numpy','warp-lang','wandb','tyro','typing_extensions']},
 filesystem_isolated=True,landlock_confined=True,pip_check_returncode=check.returncode,pip_check=check.stdout.strip())
(ROOT/'outputs').mkdir(exist_ok=True);(ROOT/'outputs/runtime_doctor.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
