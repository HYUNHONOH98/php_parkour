"""Explicit resume budgets and evidence for the B200 launcher's opt-in revision."""
from pathlib import Path
import hashlib,json

def budget(start_checkpoint_iteration,additional,total):
    """A checkpoint's iteration already completed its optimizer update."""
    if (additional is None)==(total is None):raise ValueError('Choose additional or total iterations')
    if total is not None:
        start=0 if start_checkpoint_iteration is None else start_checkpoint_iteration+1
        if total<=start:raise ValueError(f'Target {total} already reached at next iteration {start}')
        return start,total-start,total
    if additional<=0:raise ValueError('Iterations must be positive')
    # Legacy --iterations retains native load/learn indexing for old commands.
    start=0 if start_checkpoint_iteration is None else start_checkpoint_iteration
    return start,additional,start+additional

def fingerprint(value):
    """Compare tensor values and optimizer state independent of CUDA placement."""
    import torch
    digest=hashlib.sha256()
    def add(v):
        if torch.is_tensor(v):
            t=v.detach().cpu().contiguous();digest.update(str(t.dtype).encode());digest.update(str(tuple(t.shape)).encode());digest.update(t.numpy().tobytes())
        elif isinstance(v,dict):
            for k in sorted(v,key=lambda x:(type(x).__name__,str(x))):digest.update(repr(k).encode());add(v[k])
        elif isinstance(v,(list,tuple)):
            digest.update(type(v).__name__.encode())
            for x in v:add(x)
        else:digest.update(repr(v).encode())
    add(value);return digest.hexdigest()

def verify_resume(algo,checkpoint,out,next_iteration):
    import torch
    source=torch.load(checkpoint,map_location='cpu',weights_only=False)
    pairs={'actor':(source['actor_model_state_dict'],algo.actor.state_dict()),'critic':(source['critic_model_state_dict'],algo.critic.state_dict()),'actor_optimizer':(source['actor_optimizer_state_dict'],algo.actor_optimizer.state_dict()),'critic_optimizer':(source['critic_optimizer_state_dict'],algo.critic_optimizer.state_dict())}
    result={name:dict(source=fingerprint(a),restored=fingerprint(b)) for name,(a,b) in pairs.items()}
    assert all(v['source']==v['restored'] for v in result.values()),result
    source_iteration=int(source['iter']);assert algo.current_learning_iteration==source_iteration
    result.update(checkpoint=str(checkpoint),checkpoint_iteration=source_iteration,next_iteration=next_iteration,source_run=source.get('wandb_run_path'),checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),weights_and_optimizer_exact=True)
    Path(out,'resume_verification.json').write_text(json.dumps(result,indent=2));return result

def articulation_self_collision(env,requested):
    """Read the spawned USD articulation attributes, beyond the config flag."""
    import omni.usd
    from pxr import PhysxSchema,Usd,UsdPhysics
    stage=omni.usd.get_context().get_stage();root=stage.GetPrimAtPath('/World/envs/env_0/Robot');values={}
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):values[str(prim.GetPath())]=PhysxSchema.PhysxArticulationAPI(prim).GetEnabledSelfCollisionsAttr().Get()
    assert values,'No native articulation root found'
    if requested is not None:assert all(v is requested for v in values.values()),(requested,values)
    return values
