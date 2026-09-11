# 01 — Investigation: compile caches in NeMo RL today

All paths and line numbers refer to `upstream/main` at `5d49fbf4e`.

## 1. What compiles at startup, and what each framework caches

NeMo RL jobs pay JIT compilation cost in three places. Each has its own on-disk
cache with its own directory variable and its own hashing scheme. NeMo RL does not
need to reimplement any of that hashing. It only needs to move the directories.

### 1.1 TorchInductor and Triton (training and vLLM alike)

- `TORCHINDUCTOR_CACHE_DIR` is the root for the FX graph cache, AOTAutograd cache,
  the Inductor Triton kernel cache, the PGO dynamic-shape cache, and the autotune
  cache. Default is `/tmp/torchinductor_<user>`.
- `TRITON_CACHE_DIR` is Triton's own cache of compiled cubins and PTX, used by
  Triton kernels launched outside Inductor (for example fused MoE kernels).
- Entries are content-addressed. PyTorch validates that artifacts match the same
  PyTorch and Triton versions and the same GPU, so a stale or foreign entry is a
  miss, never a wrong kernel. This is what makes a "superset" bundle safe.
- PyTorch 2.6+ also exposes a "mega-cache": `torch.compiler.save_cache_artifacts()`
  returns a single bytes blob covering every modular cache populated by the
  calling process, and `torch.compiler.load_cache_artifacts(bytes)` pre-populates
  them. It is documented as portable across machines with the same torch, Triton,
  and GPU. This is an alternative to tarring `TORCHINDUCTOR_CACHE_DIR` for the
  training role. It does not cover vLLM's own artifact store.
- NeMo RL already disables the Inductor local autotune cache for policy workers
  because of a PyTorch bug with reordered node bundles
  ([nemo_rl/models/policy/utils.py:229](../../nemo_rl/models/policy/utils.py#L229)).
  So the training bundle will not contain autotune results today.

Pinned versions: `torch==2.11.0`, `vllm==0.25.1`
([pyproject.toml:28](../../pyproject.toml#L28), [pyproject.toml:162](../../pyproject.toml#L162)).

### 1.2 vLLM 0.25.1 compile cache

Verified against `vllm/compilation/backends.py`, `vllm/compilation/caching.py`, and
`vllm/envs.py` at tag `v0.25.1`:

- Root is `VLLM_CACHE_ROOT` (default `~/.cache/vllm`).
- Layout is `VLLM_CACHE_ROOT/torch_compile_cache/<hash>/rank_<rank>_<dp_rank>/<prefix>/`.
  The hash is a 10-character SHA-256 prefix over four factors: environment compile
  factors (`envs.compile_factors()`), `vllm_config.compute_hash()`, a hash of the
  traced source files, and the compiler backend hash. Files written per directory
  are `vllm_compile_cache.py`, `computation_graph.py`, and `cache_key_factors.json`.
- With `torch>=2.10` the default is `VLLM_USE_AOT_COMPILE=1`: vLLM saves
  standalone-compiled artifacts (`torch._inductor.standalone_compile`,
  `AOTCompiledArtifact`) so a warm engine loads compiled code instead of
  re-tracing. `VLLM_COMPILE_CACHE_SAVE_FORMAT` defaults to `binary`, which the
  source describes as safe for multiprocess use. `VLLM_USE_MEGA_AOT_ARTIFACT` is
  gated on `torch>=2.12`, so it is off for us.
- Because the rank directory name includes `dp_rank`, and NeMo RL runs each vLLM
  engine as its own DP=1 instance unless `VLLM_DP_SIZE` is set, every engine on
  every node produces byte-identical `rank_<tp>_0` directories. One engine's cache
  is representative of all of them.
- vLLM also uses Inductor underneath, so a warm vLLM run needs both
  `VLLM_CACHE_ROOT` and the `TORCHINDUCTOR_CACHE_DIR` seen by the vLLM venv.
- Related JIT caches that live outside `VLLM_CACHE_ROOT`: DeepGEMM
  (`DG_JIT_CACHE_DIR`; `VLLM_DEEP_GEMM_WARMUP` defaults to `relax`, which compiles
  heuristically chosen shapes at engine init), FlashInfer cubins and autotune
  (`FLASHINFER_CUBIN_DIR`, `VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR`,
  `FLASHINFER_WORKSPACE_BASE`).

### 1.3 Where NeMo RL sets these variables today

- vLLM workers: [vllm_worker.py:275-280](../../nemo_rl/models/generation/vllm/vllm_worker.py#L275)
  appends `_{seed}` to `VLLM_CACHE_ROOT` per engine to work around
  vllm-project/vllm#18851 (concurrent engines racing on one cache dir). The seed is
  `node_idx * 1024 + bundle_id`, so it differs for every engine on every node.
- Megatron policy workers: `configure_worker` sets only
  `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES`
  ([megatron_policy_worker.py:522-549](../../nemo_rl/models/policy/workers/megatron_policy_worker.py#L522)).
  DTensor workers have no `configure_worker`. Neither sets any cache directory, so
  training inherits whatever the launcher exported.
- Precedence of env vars is documented in
  [docs/design-docs/env-vars.md](../../docs/design-docs/env-vars.md): `configure_worker`
  is the highest layer, YAML `env_vars` next, then the shell environment.

## 2. What the launchers do today

### 2.1 The ultra and lightning launchers

[ultra_launch.sh:603-700](../../examples/nemo_gym/nemotron-3-ultra/ultra_launch.sh#L603)
and the near-identical block in `lightning35_launch.sh`:

- `PERSISTENT_CACHE` on Lustre holds `cache_read/*.tar.zst` and `cache_write/*/`.
  The comment at line 636 explains the split: reads are tarballs because hundreds
  of nodes read them concurrently; writes are directories because "a sidecar
  rsyncs individual files (one sequential writer)".
- `SETUP_COMMAND` ([ultra_launch.sh:797-849](../../examples/nemo_gym/nemotron-3-ultra/ultra_launch.sh#L797))
  runs on every node inside the container before `ray start`. It `rm -rf`s the
  node-local dirs, then extracts the inductor and triton tarballs into
  `/tmp/nemo_rl_inductor_cache` and `/tmp/nemo_rl_triton_cache`, and the vLLM
  tarball into `/tmp/nemo_rl_vllm_cache_warm`.
- The training command exports `VLLM_CACHE_ROOT=/tmp/nemo_rl_vllm_cache`,
  `NRL_VLLM_CACHE_SEED_DIR=/tmp/nemo_rl_vllm_cache_warm`,
  `TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR`, `DG_JIT_CACHE_DIR`
  ([ultra_launch.sh:862-867](../../examples/nemo_gym/nemotron-3-ultra/ultra_launch.sh#L862)).

Findings:

1. **The vLLM warm seed is never used.** `git grep CACHE_SEED_DIR` over `nemo_rl`,
   `ray.sub`, and `docs` returns nothing. The only references are the launchers
   themselves and the k8s example. Each engine's `VLLM_CACHE_ROOT_{seed}` dir is
   empty on every job, so vLLM recompiles on every job. Only Inductor and Triton
   are warm.
2. **There is no write-back.** `CACHE_SYNC_FREQUENCY` is exported
   ([ultra_launch.sh:619](../../examples/nemo_gym/nemotron-3-ultra/ultra_launch.sh#L619))
   and read by nothing. The rsync sidecar the comments describe was part of a
   launch flow that is not in the repository (`git log -S CACHE_SYNC_FREQUENCY`
   finds only the commit that introduced the launcher, `64cb9f985`). So
   `cache_read/*.tar.zst` is populated by hand, and the bundle does not grow
   with new shapes.
3. **The cleanup and migration logic dominates the block.** Roughly sixty lines
   move legacy directories, purge `vllm_compile_cache_<seed>` copies written by
   the earlier sidecar, and clear stale `/tmp`. That is the residue of the two
   prior designs, and it must be re-created in every launcher.
4. **Launchers disagree.** The super launcher points `VLLM_CACHE_ROOT` and
   `DG_JIT_CACHE_DIR` directly at Lustre
   ([super_launch.sh:76](../../examples/nemo_gym/nemotron-3-super/super_launch.sh#L76),
   [super_launch.sh:148](../../examples/nemo_gym/nemotron-3-super/super_launch.sh#L148)),
   which is exactly the concurrent-writers-on-Lustre pattern the ultra launcher
   moved away from. The k8s example
   ([ultra_48n_pipeclean.gb300.infra.yaml:135-160](../../infra/nrl_k8s/examples/ultra_48n_pipeclean.gb300.infra.yaml#L135))
   copies the ultra env block by hand but has no equivalent of `SETUP_COMMAND`,
   so nothing seeds the pods.

### 2.2 What `ray.sub` provides

- `SETUP_COMMAND` is written to `$LOG_DIR/setup_command.sh` and executed inside
  the head and every worker container before each `ray start` attempt
  ([ray.sub:222-227](../../ray.sub#L222), [ray.sub:739-742](../../ray.sub#L739),
  [ray.sub:953-956](../../ray.sub#L953)). It re-runs on every retry.
- The head runs the driver only after all `worker_units` are registered
  ([ray.sub:786-826](../../ray.sub#L786)), and records `NRL_RAY_READY_EPOCH` so
  Python can measure post-ready init time ([ray.sub:832](../../ray.sub#L832)).
- Teardown is signal-driven: any failure touches `$LOG_DIR/ENDED`, a
  `monitor-sidecar` on every worker polls for it every 60 s and kills the
  container ([ray.sub:872-887](../../ray.sub#L872)). A driver exit ends the head
  srun, and the submit-side script exits with its status
  ([ray.sub:1094-1116](../../ray.sub#L1094)). There is no post-driver hook that
  runs on compute nodes with the Ray cluster still alive.
- Workers are a single batched `srun` with `--ntasks-per-node=1`
  ([ray.sub:1063-1073](../../ray.sub#L1063)); each identifies itself by
  `SLURM_PROCID` and `SLURMD_NODENAME`. `ray.sub` has no notion of which nodes
  host generation versus training. That mapping only exists in the driver after
  placement groups are created.
- `DEDICATED_RAY_HEAD=1` makes the head advertise zero GPUs
  ([ray.sub:713-716](../../ray.sub#L713)). Anything that must run on GPU nodes
  cannot assume the driver's node is one.
- Requeue: `LOG_DIR` is suffixed with `SLURM_RESTART_COUNT`
  ([ray.sub:207-211](../../ray.sub#L207)), so a requeued job gets fresh signal
  files but lands on possibly different nodes.
- Node-local `/tmp` inside the enroot container is host-backed on our clusters and
  survives across jobs. The launcher's "stale `/tmp` caches from previous jobs can
  cause hangs" comment ([ultra_launch.sh:804](../../examples/nemo_gym/nemotron-3-ultra/ultra_launch.sh#L804))
  is direct evidence. That is both a hazard (stale state) and an opportunity
  (a re-run on the same nodes could skip seeding).

### 2.3 Existing per-node orchestration in `nemo_rl`

Two precedents show the driver already does once-per-node work through Ray:

- `create_local_venv_on_each_node` builds a `STRICT_SPREAD` placement group with
  one CPU per alive node that has `CPU > 0`, launches one actor per bundle, and
  waits ([nemo_rl/utils/venvs.py:186-229](../../nemo_rl/utils/venvs.py#L186)).
- `RayWorkerGroup` pools one `IsolatedWorkerInitializer` actor per placement
  group index ([worker_groups.py:508-537](../../nemo_rl/distributed/worker_groups.py#L508)).
- The driver can map bundles to physical nodes with
  `placement_group_table(pg)["bundles_to_node_id"]`
  ([virtual_cluster.py:525-527](../../nemo_rl/distributed/virtual_cluster.py#L525)),
  and `VllmGeneration` already builds a node index from it
  ([vllm_generation.py:463-465](../../nemo_rl/models/generation/vllm/vllm_generation.py#L463)).

### 2.4 Hook points in the training loop

- Clusters are created in `setup()` in `grpo.py` before any worker group
  ([grpo.py:926](../../nemo_rl/algorithms/grpo.py#L926),
  [grpo.py:1122](../../nemo_rl/algorithms/grpo.py#L1122),
  [grpo.py:1145](../../nemo_rl/algorithms/grpo.py#L1145)). Seeding belongs
  between cluster creation and worker group creation.
- vLLM engines are built in `_load_model` → `_create_engine`
  ([vllm_worker.py:463](../../nemo_rl/models/generation/vllm/vllm_worker.py#L463),
  [vllm_worker.py:916](../../nemo_rl/models/generation/vllm/vllm_worker.py#L916)),
  which may be deferred to overlap with NeMo Gym init.
- The step loop consults `TimeoutChecker.check_save()` each step to decide
  whether the walltime deadline (`checkpointing.checkpoint_must_save_by`) has
  arrived ([grpo.py:5405](../../nemo_rl/algorithms/grpo.py#L5405),
  [timer.py:419-470](../../nemo_rl/utils/timer.py#L419)). That is the right
  place to trigger a final export while walltime remains.
- The `finally` block of `async_grpo_train` finalizes checkpoints and tears
  down actors ([grpo.py:5963](../../nemo_rl/algorithms/grpo.py#L5963)). It runs
  on normal exit and Python exceptions, but not on SIGKILL, node loss, or a
  SLURM time limit that expires before the grace period ends.

## 3. Why the previous approaches hurt

| Approach | Where it ran | Failure mode |
| --- | --- | --- |
| Every node writes its cache dir to Lustre, either directly (`VLLM_CACHE_ROOT` on Lustre) or via a per-node rsync sidecar | launcher, per node | Thousands of small files per node times hundreds of nodes. Every create, rename, and stat is an MDT operation. Concurrent writers to the same directory serialize on the directory lock. Lustre also does not honor `flock` unless mounted with `-o flock`, so the frameworks' own file locking is unreliable there. |
| One per-node sidecar polling and rsyncing on an interval | launcher, per node | The sidecar lifetime was tied to the srun task, which `monitor-sidecar` kills 60 s after `ENDED`. Partial rsyncs left half-written kernels in `cache_write/`. Per-seed vLLM dirs multiplied the file count. Nothing in the driver knew whether compilation had finished. |
| Tarballs in `cache_read/`, extracted by `SETUP_COMMAND` | launcher, per node | Works for reads. But it is launcher-specific, re-created per launcher, has no producer, and its vLLM half is silently broken. |

The read side of the tarball approach is worth keeping. The write side, the
placement decision, and the framework-specific plumbing belong in `nemo_rl`.

## 4. Constraints the design must satisfy

### 4.1 Lustre

- Metadata operations are the scarce resource, not bandwidth. Prefer few large
  files. Never write per-kernel files to Lustre from compute nodes.
- Rename within a directory is atomic and is the only primitive to rely on.
  Do not rely on `flock`, `O_EXCL` semantics across nodes, or mtime ordering.
- Large files read by many nodes should be striped across OSTs
  (`lfs setstripe -c -1` or a wide PFL layout on the bundle directory).
- Directory listings of large directories are expensive under contention.
  Readers should resolve a fixed-name pointer file, not `ls`.
- Quotas and ownership are per user. A file created by user A cannot be
  replaced by user B in a sticky-bit directory, and group-writable directories
  without sticky bit let any user clobber anyone's bundle.

### 4.2 SLURM and `ray.sub`

- Node membership is only known after allocation. Role membership (which nodes
  run generation versus training) is only known after the driver creates
  placement groups. Placement is therefore a driver concern.
- The head node may have no GPUs. Per-node work must be scheduled by node id,
  not by "wherever the driver is".
- Walltime ends with SIGTERM followed by SIGKILL after `KillWait`. A final
  write-back must start while walltime remains, which the training loop
  already tracks through `checkpoint_must_save_by`.
- Requeues and retries may land on the same or different nodes. Node-local
  state must be self-describing so it can be trusted or discarded.
- Interactive clusters (no `COMMAND`) have no driver at bring-up. Seeding must
  also be possible from the launcher or from the first driver a user runs.

### 4.3 Concurrency of users and jobs

- Multiple users run the same model on the same cluster. A team wants to share
  a warm bundle, but nobody should be able to corrupt another user's job by
  writing a partial or wrong bundle into a shared path.
- One user may run several jobs with the same model at once (sweeps). Two jobs
  may export concurrently.
- A job reading a bundle must never see it disappear or change underneath it.

### 4.4 Frameworks

- Caches are demand-driven. No framework emits a "compilation complete" event.
  vLLM finishes its torch.compile work inside engine construction, but DeepGEMM,
  FlashInfer, and MoE Triton autotune run on first use of a shape. Training
  compiles in the first forward and backward, and dynamo recompiles on the
  second distinct dynamic shape before generalizing.
- vLLM writes per-engine into `VLLM_CACHE_ROOT_{seed}`. Seeding must fan the
  bundle out into every engine's directory on the node.
- All three frameworks validate their artifacts, so a bundle that is a superset
  of what a job needs is always safe and a bundle from the wrong version is a
  clean miss.
