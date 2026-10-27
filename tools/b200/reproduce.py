"""Run the packaged four-expert experiment without historical controller state."""
from pathlib import Path
import argparse,copy,fcntl,hashlib,json,os,subprocess,sys
ROOT=Path(__file__).resolve().parents[2]
PACKAGE=ROOT/'repro/b200_20261027'
TASKS=['hurdle','box_up','box_down','crouch']


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def check_assets():
    manifest=json.loads((PACKAGE/'manifest.json').read_text())
    for name,item in manifest['assets'].items():
        p=ROOT/name
        if not p.is_file()or sha(p)!=item['sha256']:raise ValueError('Missing or changed reproduction asset: '+name)
    return manifest


def prepare_experts(manifest):
    """Rebase config paths in copies; preserve every model/optimizer tensor."""
    import torch
    sys.path.insert(0,str(ROOT/'tools/b200'))
    from train_support import fingerprint
    dest=ROOT/'.runtime/b200/reproduction-inputs';dest.mkdir(parents=True,exist_ok=True)
    result={};proof={}
    def rebase(value):
        if isinstance(value,str):return value.replace(manifest['source_root'],str(ROOT))
        if isinstance(value,dict):return {k:rebase(v)for k,v in value.items()}
        if isinstance(value,list):return [rebase(v)for v in value]
        if isinstance(value,tuple):return tuple(rebase(v)for v in value)
        return value
    for task in TASKS:
        item=manifest['experts'][task];source=ROOT/item['path'];target=dest/(task+'.pt')
        original=torch.load(source,map_location='cpu',weights_only=False)
        assert original['iter']==original['iteration']==item['iteration']
        assert original['experiment_config']['robot']['asset']['enable_self_collisions']is False
        payload=copy.copy(original);payload['experiment_config']=rebase(original['experiment_config'])
        registry=(ROOT/manifest['registries'][task]).as_uri()
        payload['experiment_config']['training']['registry_name']=registry
        tensor_keys=[k for k in original if k.endswith('_state_dict')]
        fingerprints={k:fingerprint(original[k])for k in tensor_keys}
        valid=False
        if target.is_file():
            current=torch.load(target,map_location='cpu',weights_only=False)
            valid=current['experiment_config']==payload['experiment_config']and all(fingerprint(current[k])==v for k,v in fingerprints.items())
        if not valid:
            temp=target.with_suffix('.tmp');torch.save(payload,temp);temp.replace(target)
        current=torch.load(target,map_location='cpu',weights_only=False)
        assert all(fingerprint(current[k])==v for k,v in fingerprints.items())
        proof[task]=dict(source_sha256=item['sha256'],rebased_sha256=sha(target),iteration=item['iteration'],
            model_and_optimizer_tensors_unchanged=True,registry=registry,fingerprints=fingerprints)
        result[task]=target
    (dest/'provenance.json').write_text(json.dumps(proof,indent=2))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['student','teacher','prepare'])
    parser.add_argument('--perception',choices=['heightmap','depth'],default='heightmap')
    parser.add_argument('--task',choices=TASKS)
    parser.add_argument('--from-scratch',action='store_true',help='New teacher baseline; not a replay of the historical teacher lineage')
    parser.add_argument('--num-envs',type=int)
    parser.add_argument('--total-iterations',type=int)
    parser.add_argument('--save-interval',type=int,default=500)
    parser.add_argument('--logger',choices=['disabled','offline','online'],default='online')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args()
    assert os.environ.get('PHP_FILESYSTEM_ISOLATED')=='1'and os.environ.get('PHP_LANDLOCK_CONFINED')=='1','Use tools/b200/run.sh'
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
    manifest=check_assets();experts=prepare_experts(manifest)
    if args.stage=='prepare':
        print(json.dumps(dict(assets_verified=True,experts={t:str(p)for t,p in experts.items()}),indent=2));return
    if args.stage=='teacher'and not args.task:parser.error('teacher requires --task')
    if args.stage=='student'and args.from_scratch:parser.error('--from-scratch is only a teacher option; students are always new')
    tasks=TASKS if args.stage=='student'else[args.task]
    output=(args.output or ROOT/'outputs/reproductions'/('student_'+args.perception+'_nodr'if args.stage=='student'else'teacher_'+args.task)).resolve()
    assert output.is_relative_to(ROOT/'outputs'),'Output must remain inside this checkout/outputs'
    output.mkdir(parents=True,exist_ok=True)
    with(output/'reproduce.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if(output/'launch_command.json').exists():raise RuntimeError('Output already launched; choose a new --output. No automatic restart.')
        count=args.total_iterations or(30000 if args.stage=='student'else 10000)
        cmd=[sys.executable,str(ROOT/'tools/b200/train.py'),args.stage,'--total-iterations',str(count),
            '--self-collision','off','--num-envs',str(args.num_envs or(32768 if args.stage=='student'else 4096)),
            '--terrain-copies','1'if args.stage=='student'else'64','--save-interval',str(args.save_interval),
            '--registry',','.join((ROOT/manifest['registries'][t]).as_uri()for t in tasks),
            '--output',str(output),'--logger',args.logger,'--wandb-group','reproduce_b200_20261027','--seed','42']
        if args.stage=='student':
            cmd+=['--no-domain-randomization','--student-perception',args.perception]
            for task in TASKS:cmd+=['--teacher-checkpoint',str(experts[task])]
        elif not args.from_scratch:cmd+=['--checkpoint',str(experts[args.task])]
        if args.validate_only:cmd+=['--validate']
        else:(output/'launch_command.json').write_text(json.dumps(dict(command=cmd,manifest_sha256=sha(PACKAGE/'manifest.json')),indent=2))
        print(json.dumps(dict(command=cmd),indent=2),flush=True)
        result=subprocess.run(cmd,cwd=ROOT)
        if result.returncode:raise SystemExit(result.returncode)
        if not args.validate_only:
            marker=output/'finished.json'
            assert marker.exists()and json.loads(marker.read_text()).get('completed'),'Native exit alone does not confirm successful training'

if __name__=='__main__':main()
