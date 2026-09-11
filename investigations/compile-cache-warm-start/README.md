# Compile cache warm start for NeMo RL

Investigation and design for persisting and restoring the JIT compile caches
(torch.compile / TorchInductor, Triton, vLLM AOT compile, DeepGEMM, FlashInfer)
across NeMo RL jobs so that warm runs skip minutes of compilation at startup.

Baseline: `upstream/main` at `5d49fbf4e` (feat(megatron): add nccl_reshard M-to-N refit).

| Document | What it covers |
| --- | --- |
| [01-investigation.md](01-investigation.md) | What exists on `upstream/main` today, with file and line evidence. What each framework caches, where, and how it is keyed. Why the previous approaches (rsync sidecar, shared directory, launcher tarballs) hurt on Lustre and at scale. SLURM, `ray.sub`, and multi-user constraints. |
| [02-design.md](02-design.md) | The proposed design: immutable bundles, a per-user write tier and read-only shared tiers, driver-orchestrated per-node seeding and single-node export, how population is detected, atomicity and concurrency rules, failure modes, config, `ray.sub` and k8s integration, testing, rollout. |
| [03-implementation-plan.md](03-implementation-plan.md) | File-by-file implementation plan, phased rollout, measurement plan, and open questions. |

## TL;DR

- The launcher-side scheme on `upstream/main` only warms Inductor and Triton. The
  vLLM warm cache it extracts is never consumed: `NRL_VLLM_CACHE_SEED_DIR` has no
  reader in `nemo_rl`, and each vLLM engine looks in a per-engine
  `VLLM_CACHE_ROOT_{seed}` directory that starts empty.
- There is no write-back at all. `CACHE_SYNC_FREQUENCY` is exported and never read.
  The rsync sidecar that comments still describe is not in the tree.
- Proposed: move the mechanism into `nemo_rl`. The driver already knows every node
  id from the placement group table, so it can seed every node in parallel before
  worker groups start and pick exactly one node per role to export. `ray.sub` needs
  no new coordination logic.
- Shared filesystem traffic becomes: one small pointer read plus one large
  sequential tarball read per node per role on the way in, and one tarball write
  plus one rename from one node per role on the way out. No per-file writes, no
  directory listings, no locks.
- Multi-user: every job writes only to its own user tier. Shared team tiers are
  read-only and populated by an explicit promote step. Bundles are immutable and
  content-named, so concurrent readers and writers never observe partial state.
