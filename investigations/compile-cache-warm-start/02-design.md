# 02 — Design: compile cache bundles managed by the driver

## 1. Goals and non-goals

Goals:

- Warm runs of a given model skip torch.compile, Inductor, Triton, vLLM AOT,
  DeepGEMM, and FlashInfer compilation on every node, for both the training and
  generation roles.
- Shared-filesystem traffic is bounded by a constant number of large sequential
  operations per node per job. No per-kernel files ever touch the shared FS from
  compute nodes.
- One mechanism for SLURM via `ray.sub`, for k8s via `nrl-k8s`, and for
  interactive clusters. Launchers pass a path and a flag, nothing else.
- Safe under concurrent users and concurrent jobs of one user. A job never
  reads a partial bundle and never writes into another user's space.
- Every failure in the cache path degrades to a cold start. It never fails a job.

Non-goals:

- Sharing caches across GPU architectures, torch or vLLM versions, or model
  configs. The frameworks reject those anyway; the design just avoids trying.
- Caching cudagraph capture. It is not persistable.
- Replacing the frameworks' own hashing. The bundle key is for locality; the
  frameworks decide correctness.

## 2. Overview

```mermaid
sequenceDiagram
    participant D as Driver (head)
    participant SFS as Shared FS (Lustre)
    participant A as CompileCacheAgent (1 per node, CPU only)
    participant W as Workers (vLLM / policy)

    D->>D: RayVirtualCluster creates placement groups
    D->>D: node ids per role from placement_group_table
    D->>A: spawn agent on every node (NodeAffinity)
    D->>SFS: read <key>/LATEST pointer per role (1 small read)
    D->>A: seed(bundle_path, role) on every node, in parallel
    A->>SFS: open + sequential read of one tarball
    A->>A: extract to node-local dir, write .seeded marker
    D->>W: create worker groups; configure_worker sets cache env vars
    W->>W: engines build / first step compiles (hits warm cache)
    D->>A: export(role) on ONE node per role, after anchors
    A->>A: manifest snapshot, tar file list to local tmp
    A->>SFS: write <key>/bundle-<ts>-<sha>.tar.zst.tmp, rename, rename LATEST
    D->>A: final export on walltime deadline / finally (no-op if unchanged)
```

Three pieces, all inside `nemo_rl`:

1. **Bundle store** (`nemo_rl/utils/compile_cache/bundle.py`): pure filesystem
   code with no Ray dependency. Keys, pointer files, atomic publish, manifest
   hashing, tar and untar, marker files, pruning. Unit-testable on tmp dirs.
2. **`CompileCacheAgent`** (`agent.py`): a Ray actor with `num_cpus=1`,
   `num_gpus=0`, one per physical node in the virtual cluster, placed with
   `NodeAffinitySchedulingStrategy`. Exposes `probe()`, `seed()`, `export()`.
3. **`CompileCacheManager`** (`manager.py`): driver-side. Computes keys, resolves
   tiers, spawns agents, runs seeding before worker groups exist, chooses
   exporter nodes, and drives exports at anchors, on quiescence, on the walltime
   deadline, and in `finally`.

Workers change only in `configure_worker` (env vars) and, for vLLM, a hardlink
fan-out before engine creation.

## 3. Bundle layout and keying

### 3.1 On the shared filesystem

```
<tier_root>/
  v1/                                   # store format version
    <model_slug>/<gpu_arch>/<role>/<cfg_hash>/
      LATEST                            # small JSON pointer, replaced atomically
      bundle-20260911T031500Z-3f9a1c2e.tar.zst
      bundle-20260910T221207Z-a07b55d1.tar.zst   # kept for keep_last generations
```

- `model_slug` is the model name or path basename, sanitized.
- `gpu_arch` is the compute capability of the nodes in the role's cluster,
  reported by the agent's `probe()` (the driver may have no GPU).
- `role` is `policy` or `generation`. Colocated jobs have both.
- `cfg_hash` is a short SHA-256 over: torch, vLLM, Triton, CUDA and driver
  versions, precision, tensor/pipeline/expert/context parallel sizes, the
  vLLM `compilation_config` and `enforce_eager`, the training backend
  (`dtensor` or `megatron`) and its compile-relevant flags, and the container
  image tag if available. Anything else is left to the frameworks' own hashes.
- `LATEST` contains `{"file", "sha256", "size_bytes", "num_entries",
  "created_at", "created_by", "job_id", "generation", "roles_seen"}`. Readers
  never list the directory. They read `LATEST` and open the named file.
- Bundle files are immutable. A new export always produces a new name. Old
  generations are pruned only after the new `LATEST` is in place, and only
  beyond `keep_last`, so a reader that resolved `LATEST` a moment ago still
  finds its file.

### 3.2 On the node

```
<local_root>/                          # default /tmp/nrl_compile_cache
  <key>/
    .seeded                            # JSON: bundle sha256, tier, time, generation
    .exported                          # JSON: manifest hash of last export from this node
    policy/{inductor,triton}
    generation/{vllm,inductor,triton,deep_gemm,flashinfer}
    generation/vllm_<seed>             # hardlink fan-out per engine (see 5.3)
```

`configure_worker` maps env vars onto this layout:

| Role | Env var | Local path |
| --- | --- | --- |
| policy | `TORCHINDUCTOR_CACHE_DIR` | `<key>/policy/inductor` |
| policy | `TRITON_CACHE_DIR` | `<key>/policy/triton` |
| generation | `VLLM_CACHE_ROOT` | `<key>/generation/vllm_<seed>` |
| generation | `TORCHINDUCTOR_CACHE_DIR` | `<key>/generation/inductor` |
| generation | `TRITON_CACHE_DIR` | `<key>/generation/triton` |
| generation | `DG_JIT_CACHE_DIR` | `<key>/generation/deep_gemm` |
| generation | `FLASHINFER_CUBIN_DIR`, `VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR` | `<key>/generation/flashinfer/...` |

Separating roles avoids two venvs writing one Inductor dir on colocated nodes,
and makes each bundle exactly one role's contents.

## 4. Tiers: who reads what, who writes where

```yaml
cluster:
  compile_cache:
    enabled: true
    write_tier: /lustre/scratch/${USER}/nrl_compile_cache      # this job's exports
    read_tiers:                                                 # searched in order
      - /lustre/scratch/${USER}/nrl_compile_cache
      - /lustre/projects/team/nrl_compile_cache                 # read-only for most users
```

Rules:

- A job writes only to `write_tier`, which defaults to a per-user location.
  Nothing in the job ever writes into a `read_tiers` entry that is not also
  the `write_tier`.
- On seed, the manager resolves `LATEST` in every read tier and picks the
  bundle with the highest `generation`, breaking ties by `created_at`. The
  chosen tier and sha are recorded in `.seeded` and logged.
- A shared team tier is populated by an explicit `promote` command run by
  someone with write access to it:
  `python -m nemo_rl.utils.compile_cache promote --from <user tier> --to <team tier> --key <key>`.
  Promote copies the file with a temp name and renames, then replaces `LATEST`.
  It never rewrites a bundle in place.
- Directory permissions enforce the policy. Team tiers are `0755` owned by a
  service or team lead account, or `2775` with a sticky bit if a group must
  write, and the manager refuses to treat a tier as writable unless it is the
  configured `write_tier`.

Why not let everyone write to the shared tier with atomic renames? Renames are
atomic but ownership is not. With a sticky bit, user B cannot replace user A's
`LATEST`. Without it, any user's misconfigured job can replace everyone's
bundle with a broken one. Quotas would also be charged to whoever exported
last. A per-user write tier plus explicit promotion keeps the blast radius of a
bad export to its owner and makes the shared tier auditable.

## 5. Read path

### 5.1 When

In `setup()` for each algorithm, after every `RayVirtualCluster` is created and
before the first worker group. This is also before `create_local_venv_on_each_node`
returns for the vLLM venv in cold-image cases, so seeding overlaps with venv
creation if the driver launches both and waits on both.

### 5.2 How

1. `manager.spawn_agents(cluster)`: read `placement_group_table(pg)` for each
   placement group, collect unique node ids, create one `CompileCacheAgent` per
   node with `NodeAffinitySchedulingStrategy(node_id, soft=False)`. Nodes that
   host both roles get one agent that handles both.
2. `agent.probe()` returns compute capability, torch and vLLM versions, and
   free space on the local root. The manager uses the first probe to finish
   the key, and refuses to seed a node whose probe disagrees (heterogeneous
   allocation), logging instead.
3. For each role, resolve `LATEST` across read tiers on the driver. One small
   read per tier per role, done once, not per node.
4. `ray.get([agent.seed.remote(role, bundle_path, sha, generation) for ...])`
   with a timeout. Each agent:
   - If `.seeded` matches the sha for that role, return immediately. This makes
     a requeue onto the same nodes free and replaces the unconditional
     `rm -rf` in the launcher.
   - Otherwise remove the role subtree, `tar --zstd -xf` the bundle to it,
     write `.seeded`, return timing and size.
   - Any failure removes the role subtree and returns a structured error. The
     manager logs and continues. The node starts cold.
5. Stale state that does not match any current key is pruned by the agent if
   the local root exceeds a configured size, oldest `.seeded` first.

Shared-FS cost per job: `tiers × roles` pointer reads from the driver, and one
tarball read per node per role. With the bundle directory striped wide
(`lfs setstripe -c -1` applied once by `init` or `promote`), hundreds of nodes
reading one file is the access pattern Lustre serves best.

### 5.3 vLLM fan-out

`configure_worker` keeps the `_{seed}` suffix on `VLLM_CACHE_ROOT` because the
race it works around is real. In `_load_model`, before `_create_engine`, the
worker checks whether `VLLM_CACHE_ROOT` exists. If it does not and
`<key>/generation/vllm` does, it populates it with
`shutil.copytree(src, dst, copy_function=os.link)`. Hardlinks cost nothing and
are safe because vLLM and Inductor write new files via temp plus rename rather
than modifying in place. If `os.link` fails (cross-device local root), fall
back to copy. This is the missing consumer of the seed directory.

When `VLLM_DP_SIZE > 1` is in use, the seeded `rank_<tp>_0` directories are
also linked to `rank_<tp>_<dp>` for each dp rank the engine will use. The
compiled graph is identical; only the directory name differs.

## 6. Write path

### 6.1 Who exports

The manager picks one exporter node per role:

- generation: the node hosting the DP-leader worker of DP replica 0
  (`worker_group.get_dp_leader_worker_idx(0)` → its placement group → node id).
- policy: the node hosting rank 0.

Every node of a role compiles the same kernels for the same model, so one is
representative. If the exporter node is lost, the manager re-selects on the
next export attempt.

### 6.2 When: anchors, quiescence, deadline, teardown

No framework signals completion, so exports are triggered by a combination:

| Trigger | Role | Why |
| --- | --- | --- |
| After `_load_model` returns on the exporter engine | generation | vLLM's torch.compile and AOT save happen inside engine construction. DeepGEMM `relax` warmup also runs there. |
| After the first generation batch completes | generation | FlashInfer autotune, MoE Triton configs, DeepGEMM shapes outside the warmup heuristic. |
| After optimizer step `export_after_step` (default 1) | policy | First forward and backward compile. |
| Quiescence: manifest unchanged across `quiescence_steps` consecutive step boundaries and different from `.exported` | both | Captures dynamo's second-shape recompiles and any late JIT without knowing about them. |
| `TimeoutChecker.would_save()` is true | both | Same deadline the checkpoint uses; runs while walltime remains. |
| `finally` block, before `cluster.shutdown()` | both | Backstop. Bounded wait so it cannot hold up exit. |

The policy worker can additionally report
`torch._dynamo.utils.counters["inductor"]` deltas each step. Zero new
`fxgraph_cache_miss` since the last export is a precise "nothing new compiled"
signal that avoids scanning the directory. vLLM's engine core is a subprocess,
so for generation the manifest scan is the only signal.

Exports are cheap: one node tars a few gigabytes from local disk. Exporting
"too early" is harmless because the next run seeds from that bundle, compiles
the remainder, and exports a superset. Bundles are monotone across runs.

### 6.3 How: atomic, single writer, superset-safe

`agent.export(role, dest_dir, generation, keep_last)`:

1. Snapshot the manifest: sorted `(relpath, size)` for the role subtree,
   excluding lock files and temp files. Hash it. If equal to `.exported`,
   return `unchanged`.
2. Tar exactly the files in the snapshot (`tar -T filelist`), not the
   directory. Every listed file is complete because the frameworks write via
   temp plus rename. Files that appear mid-export are picked up next time.
   Compress with `zstd -T0` to local tmp first, then compute sha256.
3. Copy to `<dest_dir>/.tmp/bundle-<ts>-<sha8>.tar.zst.<jobid>.<host>.<pid>`,
   then `os.replace` into `<dest_dir>/bundle-<ts>-<sha8>.tar.zst`. Temp and
   final live in the same directory so the rename is a single MDT operation.
4. Write `LATEST.tmp.<unique>` with `generation = seeded_generation + 1` and
   `os.replace` it over `LATEST`.
5. Prune bundles beyond `keep_last`, oldest first, never the one `LATEST`
   names.
6. Write `.exported` locally.

Shared-FS cost per export: one large sequential write, two renames, one small
write, and at most one unlink.

Concurrent exports of the same key by two jobs of the same user race only on
step 4. Both bundles are supersets of what they seeded from; last writer wins
on the pointer and the loser's file stays until pruned. A reader in between
holds a name, not the pointer, so it is unaffected.

## 7. Failure modes

| Failure | Behavior |
| --- | --- |
| No bundle for the key in any tier | Cold start. Export will create the first one. |
| Bundle sha mismatch after read, or extract error | Role subtree removed, node starts cold, structured error logged. |
| Agent cannot be placed on a node (no CPU left, node dead) | That node is skipped and starts cold. Job proceeds. |
| Seed exceeds `seed_timeout_s` | Remaining nodes are left to finish in the background; the job proceeds. Late seeds still help workers that have not compiled yet. |
| Exporter node dies mid-export | Temp file is orphaned under `.tmp/`; `init`/next export cleans temps older than a day. Pointer untouched. |
| Two users configure the same `write_tier` | Refused at config validation unless the directory is owned by the current user or `allow_shared_write_tier: true` is set explicitly. |
| Local disk full | Agent reports free space in `probe()`; the manager skips seeding below a threshold and logs. |
| Heterogeneous GPU archs in one role | Key differs per node; the manager seeds per arch group and only exports from the first group. |
| Stale `/tmp` from a prior job with a different key | Pruned lazily by size; ignored otherwise. Frameworks never see it because env vars point at the current key. |
| Frameworks upgraded, bundle now foreign | Every framework validates; cold compile; export replaces the bundle under a new `cfg_hash` anyway. |

## 8. Configuration

New user-facing config follows the v2 convention: a `pydantic.BaseModel` with
defaults on the fields, documented in the exemplar YAMLs. `ClusterConfig` is a
legacy `TypedDict`; it gains a `compile_cache: NotRequired[CompileCacheConfig]`
field and the manager validates the block with `CompileCacheConfig.model_validate`.

```python
class CompileCacheConfig(BaseModel, extra="allow"):
    enabled: bool = False
    write_tier: Optional[str] = None            # NRL_COMPILE_CACHE_DIR overrides when set
    read_tiers: list[str] = []                  # write_tier is always searched first
    local_root: str = "/tmp/nrl_compile_cache"
    roles: list[str] = ["policy", "generation"]
    export_after_step: int = 1
    quiescence_steps: int = 2
    export_at_deadline: bool = True
    export_at_teardown: bool = True
    seed_timeout_s: int = 600
    export_timeout_s: int = 600
    keep_last: int = 2
    local_max_gb: int = 200
    allow_shared_write_tier: bool = False
```

Launcher-facing overrides: `NRL_COMPILE_CACHE_DIR` sets `write_tier` and
prepends it to `read_tiers`; `NRL_COMPILE_CACHE_READ_TIERS` is a colon-separated
list appended to `read_tiers`. These exist so `ray.sub`-based launchers and the
k8s entrypoint do not need to edit YAML.

## 9. Integration with `ray.sub`, launchers, and k8s

`ray.sub` needs no coordination logic. The driver does the placement. Changes:

- Document `NRL_COMPILE_CACHE_DIR` next to `HF_HOME` in `docs/cluster.md` and
  pass it through like the other `NRL_*` variables.
- Keep `SETUP_COMMAND` as an optional pre-seed for interactive clusters. Ship a
  helper `tools/compile_cache_preseed.sh <tier> <key>` that does the same
  marker-aware extract the agent does, so a user attaching to an interactive
  cluster gets warm nodes without a driver. When the driver later runs, the
  agent sees a matching `.seeded` and skips.
- Remove the cache blocks from the ultra, lightning, and super launchers and the
  k8s example. Replace with `NRL_COMPILE_CACHE_DIR=$PERSISTENT_CACHE/compile_cache`
  and `cluster.compile_cache.enabled=true`. The migration and purge logic goes
  away with them.
- `nrl-k8s`: identical. Pods have node-local ephemeral storage for `local_root`
  and mount the PVC for the tiers. No init container needed.

## 10. Observability

Every seed and export logs one line with role, tier, sha8, generation, size,
and duration, prefixed `[COMPILE_CACHE]`, and the manager emits
`compile_cache/seed_hit_nodes`, `compile_cache/seed_time_s`,
`compile_cache/export_time_s`, `compile_cache/bundle_bytes` to the metrics
logger so warm versus cold startup is visible in W&B next to the existing
container-startup timing derived from `NRL_JOB_START_EPOCH`.

`python -m nemo_rl.utils.compile_cache status --key <key>` prints the resolved
tiers, the current `LATEST` in each, and generations, for debugging a miss.

## 11. Alternatives considered

| Alternative | Verdict |
| --- | --- |
| Keep the launcher scheme, fix the seed-dir consumer | Fixes vLLM but leaves no producer, no placement, and per-launcher copies. Rejected. |
| Per-node rsync sidecar to a shared directory | The Lustre metadata storm the team already hit. Rejected. |
| Point caches directly at Lustre | Concurrent writers, unreliable locking, MDT contention during compile. Rejected. |
| Ray object store broadcast of the bundle bytes | Cuts shared-FS reads to one per job per role, but costs gigabytes of plasma memory on every GPU node at startup. Keep as a flag for clusters with a weak shared FS. |
| squashfs bundle mounted read-only plus overlayfs upper on local disk | Zero extraction, lazy reads. Needs `/dev/fuse` or privileged containers. Prototype later. |
| PyTorch mega-cache (`save_cache_artifacts`) for the training role | One blob instead of a tar, and loadable without touching the filesystem. Good phase-2 replacement for the policy bundle's Inductor portion; does not cover vLLM. |
| Inductor remote cache (Redis) | Requires a Redis service reachable from compute nodes and only covers Inductor. Not pursued. |
| Have `ray.sub` pick exporter nodes | It cannot; role placement is decided in the driver after placement groups exist. |
