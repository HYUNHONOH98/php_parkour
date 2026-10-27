# B200 task-separated PHP experiments

The current experiment is a **heightmap-only, no-DR Student**, trained from four
frozen task experts. This branch includes the actual training code, processed
motion/terrain inputs, headless robot USD, and full expert checkpoints used by
that run. It is a custom four-task experiment, not a claim to reproduce every
result in the original PHP paper.

- [Experiment configuration and history](TASK_SUITE_B200.md)
- [Asset/checkpoint hashes and provenance](repro/b200_20261027/manifest.json)
- [Recorded validation evidence](repro/b200_20261027/evidence)

## What is included

| Item | Location |
|---|---|
| Hurdle, descent, crouch inputs; original historical ascent | `data/task_teachers_20261026/` |
| **Corrected active ascent**: 611 frames, authored approach geometry | `data/box_up_original_geometry_20261026/box_up/` |
| Exact headless robot and collision USD | `data/b200_robot_headless/` |
| Full expert checkpoints, including actor/critic and both Adam states | `repro/b200_20261027/experts/` |
| Native clock fix, B200 buffer/material-command adapter, exact-mesh tiling | `wbt_training/utils/`, `wbt_training/simulators/b200.py` |
| Heightmap actor and no-DR configuration | `wbt_training/config_values/heightmap_student.py`, `no_domain_randomization.py` |
| Flat-MLP DAgger+PPO adapter and ONNX export | `wbt_training/utils/heightmap_distillation.py` |
| Stateless reproduction launcher | `tools/b200/reproduce.py` |
| Runtime lock and exact Isaac Lab source snapshot, including apps | `repro/b200_20261027/runtime/` |

No W&B API key, SSH key/password, installed Conda environment, host driver
library, cache, or multi-gigabyte training output is committed. The original
processed inputs are bundled; training does not require another MinReward
checkout, access to the original motion registry, or running motion matching.

## Runtime installation

Target: Ubuntu 24.04 / Linux x86_64, a single B200 with a working NVIDIA driver,
and user/mount/IPC namespaces plus Landlock available. CUDA device 0 is used.
The `server_2` GPU 4/5 constraint is specific to that other host.

```bash
git clone --branch codex/php-b200-repro-20261027 \
  --recurse-submodules https://github.com/HYUNHONOH98/php_parkour.git
cd php_parkour
```

Provision a **project-local Conda bootstrap** at `.runtime/conda-bootstrap`
first, for example with the [official Miniforge installer](https://github.com/conda-forge/miniforge#install). Use its noninteractive `-b -p` install option and this checkout’s absolute `.runtime/conda-bootstrap` path; do not run `conda init` or change another project’s environment. The installer intentionally requires its Python to be under this
checkout and refuses to clone an environment from another project or overwrite
an existing `.runtime/b200/env`. On an already provisioned B200 checkout, keep
its independent runtime and skip installation.

```bash
.runtime/conda-bootstrap/bin/python tools/b200/bootstrap.py --plan
# After reviewing and accepting NVIDIA's Isaac Sim EULA:
.runtime/conda-bootstrap/bin/python tools/b200/bootstrap.py --accept-isaac-eula
./tools/b200/run.sh python tools/b200/doctor.py
```

Bootstrap installs the captured Conda package URLs and Python versions into
`.runtime/b200/env`, restores the exact Isaac Lab source, installs the pinned
Holosoma/PHP source locally, copies the OS/GPU interfaces into the private
runtime filesystem, and rewrites launcher shebangs for this checkout path.
The small Landlock helper has both source (`confine.c`) and a supplied
Linux/x86_64 executable. It can be rebuilt with a project-local C compiler.

The validated versions include Python 3.11.17, Torch 2.7.0+cu128, Isaac Sim
5.1.0.0, Isaac Lab 0.47.2, NumPy 1.26.0, Warp 1.10.0, W&B 0.22.0, Tyro 1.0.7,
and typing_extensions 4.15.0. The exact package inventory is captured alongside
the lock files. The `unitree_sdk2` and `booster_robotics_sdk` hardware SDKs are
recorded in the inventory but excluded from the training install lock; they
are not training dependencies.

The installed environment has one known `pip check` conflict: Isaac Sim kernel
pins typing_extensions 4.12.2 whereas Tyro needs 4.15.0. The recorded runtime
uses 4.15.0 and passed real training. The lock is installed with `--no-deps` to
preserve that tested combination, not to claim that dependency metadata is
conflict-free. A fresh full dependency download/reinstallation has not been
repeated for this branch; relocation and native learning are tested using the
recorded independent B200 runtime.

Experiment entry points always use `tools/b200/run.sh`. It applies a private
mount/IPC namespace, chroot, capability dropping, and Landlock. Application
code, executables, libraries, credentials, caches, temporary files and outputs
stay under this checkout; kernel/GPU device interfaces are infrastructure.
The OS/runtime provisioning step copies host interface libraries; it never
uses another project's Python environment. Refresh the private driver copies
when the host driver changes.

## Reproduce the current heightmap Student

First check inputs and rebase checkpoint metadata for the current checkout.
Original packaged checkpoints remain unchanged; only copies under `.runtime/`
are rewritten. Every model and optimizer tensor is fingerprinted before/after
rebasing.

```bash
./tools/b200/run.sh python tools/b200/reproduce.py prepare
./tools/b200/run.sh python tools/b200/reproduce.py student \
  --perception heightmap --validate-only --logger disabled \
  --output outputs/reproductions/heightmap_preflight

# Actual smoke run: 128 envs, four optimizer updates, all four experts.
./tools/b200/run.sh python tools/b200/reproduce.py student \
  --perception heightmap --num-envs 128 --total-iterations 4 \
  --save-interval 1 --logger disabled \
  --output outputs/reproductions/heightmap_smoke

# Full run; detached execution can use your own job scheduler.
./tools/b200/run.sh python tools/b200/reproduce.py student \
  --perception heightmap --num-envs 32768 --total-iterations 30000 \
  --logger online --output outputs/reproductions/heightmap_32768_30k
```

Online logging uses `humanoid-parkour/php-parkour`. Supply your own W&B key in
`.runtime/b200/secrets/wandb-api-key` (directory 0700, file 0600), or use
`--logger offline` / `--logger disabled`. No credential is part of this branch.

Do **not** invoke historical production queue/stop scripts. The reproduction
launcher needs no historical `status.json`, PID, completed-teacher marker, or
external registry. It locks its output and refuses to reuse an output that
already launched. Its default student command always includes all four experts
in hurdle/ascent/descent/crouch order and all DR OFF.

The branch was also unpacked into a separate B200 path and its public reproduction launcher completed the 128-env/four-update test after checkpoint metadata rebasing. The packaged original checkpoint hashes and every model/optimizer tensor remained unchanged; the existing production Student kept running. Evidence: `repro/b200_20261027/evidence/fresh_checkout_verification.json`.

A successful process exit alone is insufficient. Check `finished.json`,
`runtime_probe.json`, `student_update_verification.json`, finite loss rows in
`metrics.jsonl`, and a saved checkpoint. The smoke proof requires a changed
Student, all 44 teacher tensors unchanged, an actual 397-wide actor input, and
no depth sensor/CNN. The exported `model_*.onnx` is one 397-input / 29-output
MLP; there is no depth-backbone model.

## Other controlled experiments

The previous depth/CNN Student uses the same four experts and no-DR physics:

```bash
./tools/b200/run.sh python tools/b200/reproduce.py student \
  --perception depth --num-envs 32768 --total-iterations 30000 \
  --logger disabled --output outputs/reproductions/depth_nodr
```

To continue one packaged Teacher with its full optimizer state to **total**
10,000 updates, in a new output directory:

```bash
./tools/b200/run.sh python tools/b200/reproduce.py teacher --task box_up \
  --total-iterations 10000 --logger disabled \
  --output outputs/reproductions/box_up_continue
```

A separate `--from-scratch` Teacher baseline is supported, but does not recreate
the historical expert lineage. Teacher randomization remains its own original
preset; the request to disable all DR applies to the Student. Nothing in the
reproduction launcher automatically starts or restarts a Teacher. The live
production Teachers were explicitly stopped and must stay stopped.

The original checkpoints fix the teacher targets for Student reproduction.
GPU PhysX, random sampling and parallel execution do not promise bitwise
identical learning trajectories across fresh runs or changed hardware.
