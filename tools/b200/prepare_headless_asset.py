"""Remove two dangling visual-only USD references; keep all physics unchanged."""
from pathlib import Path
import hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[2]
source=ROOT/'data/b200_robot';target=ROOT/'data/b200_robot_headless'
if not target.exists():shutil.copytree(source,target)
from isaaclab.app import AppLauncher
original=AppLauncher._resolve_device_settings
def device(self,args):
    original(self,args);args.update(active_gpu=0,physics_gpu=0,multi_gpu=False,max_gpu_count=1)
AppLauncher._resolve_device_settings=device
launcher=AppLauncher(headless=True,device='cuda:0',kit_args='--allow-root --/app/settings/loadUserConfig=false --/app/settings/persistent=false')
from pxr import Sdf,Usd,UsdPhysics
name='main_mesh_arm_collision_halfspherehand'
def signature(folder):
    stage=Usd.Stage.Open(str(folder/(name+'.usd')))
    return {key:sorted(str(p.GetPath()) for p in stage.Traverse(Usd.TraverseInstanceProxies()) if p.HasAPI(api)) for key,api in [('rigid_bodies',UsdPhysics.RigidBodyAPI),('collisions',UsdPhysics.CollisionAPI)]} | {'joints':sorted(str(p.GetPath()) for p in stage.Traverse() if p.IsA(UsdPhysics.Joint))}
before=signature(source)
base=Sdf.Layer.FindOrOpen(str(target/'configuration'/(name+'_base.usd')))
removed=[]
for foot in ['LL_FOOT','LR_FOOT']:
    spec=base.GetPrimAtPath(f'/g1/{foot}/visuals')
    refs=list(spec.referenceList.GetAppliedItems())
    assert not refs or len(refs)==1 and str(refs[0].primPath)==f'/visuals/{foot}',refs
    spec.referenceList.ClearEdits()
    assert not spec.referenceList.GetAppliedItems()
    removed.append(str(spec.path))
base.Save()
after=signature(target);assert before==after,'Physical USD articulation/collisions changed'
hashes={str(path.relative_to(source)):hashlib.sha256(path.read_bytes()).hexdigest() for path in source.rglob('*physics.usd')}
assert hashes and all(hashlib.sha256((target/path).read_bytes()).hexdigest()==digest for path,digest in hashes.items())
report={'removed_visual_references':removed,'physics_identical':True,'physics_layer_sha256':hashes,'counts':{key:len(value) for key,value in before.items()},'usd':str(target/(name+'.usd'))}
(ROOT/'outputs/task_suite_20261026/headless_asset.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2));launcher.app.close()
