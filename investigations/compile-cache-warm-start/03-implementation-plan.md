# 03 — Implementation plan

## Phase 0: measurement baseline (before any code)

Run the ultra pipeclean recipe twice on the same allocation size with the
current launcher and record, per role, the time from `NRL_RAY_READY_EPOCH` to
the first completed step, and the vLLM engine construction time from the
worker logs. This is the number the rest of the work is judged against. Keep
the numbers in this directory as `baseline.md`.

## Phase 1: bundle store and agent, no training-loop changes

Files:

- `nemo_rl/utils/compile_cache/__init__.py`
- `nemo_rl/utils/compile_cache/config.py`: `CompileCacheConfig` BaseModel and
  env-var overrides.
- `nemo_rl/utils/compile_cache/bundle.py`: `compute_key`, `resolve_latest`,
  `manifest`, `pack`, `unpack`, `publish` (temp plus rename, pointer update,
  prune), marker read and write, `preseed` helper used by the CLI.
- `nemo_rl/utils/compile_cache/agent.py`: `CompileCacheAgent` Ray actor.
- `nemo_rl/utils/compile_cache/manager.py`: `CompileCacheManager`.
- `nemo_rl/utils/compile_cache/__main__.py`: `status`, `promote`, `init`
  (creates tier dirs and applies striping), `preseed`.
- `nemo_rl/distributed/virtual_cluster.py`: add `compile_cache` to
  `ClusterConfig` and a helper returning unique node ids per placement group.

Tests (`tests/unit/utils/test_compile_cache.py`, no GPU):

- key stability and sensitivity to each factor
- `publish` is atomic: a concurrent reader that resolved `LATEST` before a
  publish still opens a valid file; pruning never removes the pointed-to file
- manifest hash ignores temp and lock files; export returns `unchanged` when
  nothing changed
- `unpack` skips when the marker matches and wipes on mismatch
- refusal of a shared `write_tier` unless explicitly allowed
- agent seed and export on a local Ray cluster with two fake nodes using
  custom resources

## Phase 2: worker plumbing

- `vllm_worker.configure_worker`: set the generation env vars from the local
  layout when the feature is enabled; keep the `_{seed}` suffix.
- `vllm_worker._load_model`: hardlink fan-out from the seeded vLLM dir into
  `VLLM_CACHE_ROOT` before `_create_engine`; DP-rank directory aliasing when
  `VLLM_DP_SIZE > 1`; `get_cache_dirs()` for the manager.
- `megatron_policy_worker.configure_worker` and a new
  `dtensor_policy_worker_v2.configure_worker`: set policy env vars.
- Policy workers: `get_compile_counters()` returning dynamo and inductor
  counter deltas.

Functional test (`tests/functional`): tiny model, one node, run twice with a
tmp tier; assert the second run logs a seed hit for both roles and that vLLM
engine construction is faster than the first run.

## Phase 3: training loop integration

- `grpo.py` `setup()`: build the manager after clusters exist, seed before
  worker groups; same for `sft.py`, `dpo.py`, `distillation.py`, `rm.py`.
- `grpo_train` and `async_grpo_train`: `manager.on_generation_ready()` after
  `prepare_for_generation`, `manager.on_step_end(step, counters)` each step,
  `manager.on_deadline()` where `should_save_by_timeout` is computed,
  `manager.on_teardown()` in `finally` before `cluster.shutdown()`.

## Phase 4: launcher and docs cleanup

- Remove cache blocks from `ultra_launch.sh`, `lightning35_launch.sh`,
  `super_launch.sh`, and the k8s example. Replace with the env var and flag.
- `docs/design-docs/compile-cache.md` (move the design there), update
  `docs/cluster.md` and `docs/design-docs/env-vars.md`, register in
  `docs/index.md`.
- Exemplar YAMLs: document `cluster.compile_cache` with defaults, and update
  `tests/unit/reference_configs` if the v1 to v2 migration check covers them.

## Phase 5 (optional): read-path optimizations

- Object-store broadcast behind `seed_transport: object_store`.
- Mega-cache blob for the policy role's Inductor portion.
- squashfs plus overlayfs prototype on a cluster that allows fuse.

## Rollout

1. Land phases 1 and 2 with `enabled: false`. No behavior change.
2. Enable on one recipe (ultra pipeclean) with a per-user tier. Compare against
   the phase 0 baseline. Verify warm bundles converge after two runs.
3. Create a team tier, promote the converged bundles, add it to `read_tiers`
   in the recipe.
4. Enable by default in the ultra and lightning recipes; remove the launcher
   blocks.

## Open questions

- Does any recipe run heterogeneous GPU archs inside one role today? If so the
  key split per arch group needs a test.
- Should `write_tier` default to `$HF_HOME/nemo_rl/compile_cache` when unset,
  mirroring `get_megatron_checkpoint_dir`, or stay opt-in? Opt-in is safer
  for the first release.
- For vLLM with `VLLM_DP_SIZE > 1`, confirm on a real run that the dp-rank
  directory aliasing produces hits rather than re-hashing.
- FlashInfer's autotune cache location differs between FlashInfer versions.
  Confirm the env var the pinned version honors before adding it to the
  generation layout.
- Whether `SETUP_COMMAND` pre-seeding for interactive clusters is worth
  keeping once the driver-side path exists, or whether "first driver seeds"
  is enough.
