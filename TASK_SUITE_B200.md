# Four-task PHP experiment record — 2026-10-27

## Current run

[Heightmap Student m730hwnl](https://forge.coreweave.com/wandb/humanoid-parkour/php-parkour/runs/m730hwnl)
is the current run. [Reproduction commands](README_B200.md) use packaged expert
checkpoints and do not depend on the historical production controller.

| Setting | Current value |
|---|---|
| Student actor input | 108 proprioception/command + 289 height samples, exactly that order |
| Student MLP | 397 → 2048 → 1024 → 512 → 256 → 128 → 29, ELU |
| Perception | 17×17 height scanner, 1.6×1.6 m, 0.1 m spacing, yaw aligned, 50 Hz |
| Depth / CNN / RNN | None |
| Algorithm | Native DAgger+PPO schedule and residual-based PPO gate |
| Parallel environments / budget | 32,768 / 30,000 updates |
| Rollout length / control / physics | 24 control steps per update / 50 Hz / 200 Hz |
| Student learning rate | 3e-4; original adaptive schedule and loss coefficients retained |
| Student DR | OFF: observation, physical, reset, delay, push and spawn perturbations |
| Fixed materials | Robot and terrain static/dynamic friction 1, restitution 0 |
| Self-collision | OFF |
| Sampling retained | Uniform expert/frame dataset sampling and PPO action exploration |
| Curriculum retained | Deterministic termination-threshold relaxation |
| Training-only observations | Teacher 450 dims including routing; critic 575 dims |
| W&B | `humanoid-parkour/php-parkour` |
| Seed | 42 |

The original 108 features, sorted by observation term name, are previous action
(29), base angular velocity (3), joint position (29), joint velocity (29),
projected gravity (3), velocity-command one-hot (15). `zz_height_scan` sorts
last and appends 289 values. No motion-reference state is added to this actor.
Teachers still receive their original privileged input, including height_scan;
the routing column is removed before their 449-input forward pass.

The four courses share one exact-mesh layout, with matching rigid transforms
applied to motions and terrain. Each reset samples a task expert and a motion
frame. This is not one episode that must execute all four obstacles in sequence.
One fixed copy per task is used. `add_onpath_obstacle=True` is the internal
exact-mesh tiling switch in `apply_task_preprocess`; this suite does not add
randomized obstacles through it. The 40 s episode cap and reference-end
resampling semantics remain those of the captured training configuration.

## Frozen experts and exact input bundles

| Index | Task | Frozen checkpoint update | Frames | Duration at 50 Hz | Active input directory |
|---:|---|---:|---:|---:|---|
| 0 | Walk–Hurdle–Walk | 6100 | 382 | 7.64 s | `data/task_teachers_20261026/hurdle` |
| 1 | Walk–Box climb up–Walk | 6100 | 611 | 12.22 s | `data/box_up_original_geometry_20261026/box_up` |
| 2 | Walk–Box climb down–Walk | 6100 | 642 | 12.84 s | `data/task_teachers_20261026/box_down` |
| 3 | Walk–Crouch–Walk | 6200 | 253 | 5.06 s | `data/task_teachers_20261026/crouch` |

The corrected ascent preserves the original authored `6ujlv349` reference,
including its original relative box placement. Entry root-to-front-plane
separation is 0.4096321063 m; blending affects surrounding walking frames.
The old 603-frame ascent remains in the historical task folder but is **not**
selected by the reproduction manifest.

The checkpoint files are full snapshots (actor, critic, both optimizers,
configuration and metadata), not newly trained replacement teachers. SHA256
hashes and source W&B identities are in `repro/b200_20261027/manifest.json`.
All four teacher training processes remain stopped.

## History and validation

1. Four separate 4,096-env teachers were originally scheduled for 20k updates.
   The ascent reference/terrain alignment was corrected separately.
2. Self-collision was disabled and existing weights/optimizer states were
   continued. A later plan targeted total 10k; those teachers did **not** all
   finish 10k before the user requested Student training.
3. The no-DR depth/CNN Student
   [b3892i64](https://forge.coreweave.com/wandb/humanoid-parkour/php-parkour/runs/b3892i64)
   used the packaged 6100/6100/6100/6200 experts. It was stopped around update
   176 on request; its recorded artifacts were preserved.
4. The current heightmap-only Student starts a new 397-input MLP. Its first
   128-env test completed four updates. The 32,768-env run completed updates
   0–9 by 15:34:17 KST with finite losses, exact 108+289 layout, no camera/CNN,
   actual no-DR material/reset/observation settings, and native self-collision
   OFF. Student tensors changed; all 44 frozen-teacher tensors remained fixed
   and matched their immutable source weights. There was no PhysX overflow.
   The single ONNX MLP matched PyTorch within 5.97e-7 on a three-sample test.

B200 implementation fixes preserve the MDP: the physics clock remains 200 Hz
under 50 Hz control; PhysX rigid-patch capacity is at least `32*num_envs`
(1,048,576 at 32,768 envs); the adapter registers and fully qualifies the
**original native NVIDIA PreviewSurface command**. Headless USD removes two
invalid foot *visual* references while preserving collision/physics assets.

Recorded success-rate evidence is an earlier teacher evaluation at checkpoints
5700/5800, 64 first-clip rollouts per task: hurdle 61/64, ascent 62/64,
descent 61/64, crouch 64/64. These are not Student success rates and are not
an evaluation of the exact later frozen checkpoints in this branch.
