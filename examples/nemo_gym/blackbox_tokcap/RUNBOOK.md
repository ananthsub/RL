# Blackbox Harness Token-Capture Training Runbook

Train blackbox agent harnesses (Claude Code CLI, Codex CLI, Hermes agent) with
token-id capture and prefix supply, on async GRPO, on a GB200 Slurm cluster.
File-store capture path only — TransferQueue is explicitly out of scope.

These instructions are written so an agent with no prior context can execute the
experiment end to end. Every step has a verification command. Do the steps in
order; do not skip the smoke test.

- Written: 2026-08-30. Facts verified against code on that date.
- Companion page (same content, shareable): https://claude.ai/code/artifact/efc0093d-d6bb-4759-babc-32400b765287
- All commands run from the NeMo RL repository root unless stated otherwise.

---

## 0. What this experiment is

A blackbox harness drives its own multi-turn agent loop and returns a transcript
with **no token ids**. RL training needs the exact token ids and logprobs the
policy sampled. The mechanism under test:

1. The harness's model endpoint is pointed at the NeMo Gym model server with a
   rollout-scoped URL prefix: `<base_url>/ng-rollout/<rollout_id>/training-token-capture`.
   For Claude Code this rides in `ANTHROPIC_BASE_URL`; for Codex, in a generated
   `config.toml`; for Hermes, in the `AIAgent(base_url=...)` constructor.
2. The Gym model server strips the prefix, keys every model call by rollout id,
   and appends a `TokenEntry` (prompt token ids, generation token ids, generation
   logprobs, output items) to a per-rollout file store.
3. **Prefix supply** (the treatment arm): on each subsequent call in a rollout,
   the model server resolves the call's parent and hands vLLM the previous
   call's exact served tokens instead of letting the chat template re-render the
   conversation. Without it, re-rendering can diverge from what was sampled
   (worst case: thinking models whose templates strip prior-turn reasoning).
4. After the rollout, a builder chains the calls into one contiguous trainable
   Responses payload (longest-strict-prefix parent inference; generation tokens
   get loss mask 1, re-rendered prompt tokens mask 0). A failed rebuild masks
   that one sample (`mask_sample=true`) — it never corrupts the batch.
5. The NeMo RL `NemoGym` actor correlates rollouts, finalizes the rebuilt
   trajectory, trains on it, emits `token_capture/*` metrics, and retires the
   records.

The experiment: prove this full stack at 30B MoE scale with async GRPO, for all
three harnesses, and A/B prefix supply.

---

## 1. Prerequisites

Check every box before proceeding.

### 1.1 Access and credentials

- [ ] Slurm cluster with GB200 nodes (4 Blackwell GPUs per node, Grace CPUs =
      **aarch64**). If you are on 8-GPU x86 nodes instead, see §7.4 for the
      config deltas.
- [ ] A shared filesystem visible from all nodes (the repo, HF cache, and
      results live there).
- [ ] NeMo RL nightly container as a squashfs, **aarch64 build** for GB200.
      Set `CONTAINER_IMAGE_PATH` to it. An x86 image will not start on Grace.
- [ ] HuggingFace token (`HF_TOKEN`) — needed for the NVIDIA datasets.
- [ ] `WANDB_API_KEY` (or flip `logger.wandb_enabled: false` in the config).
- [ ] `SLURM_ACCOUNT` and `SLURM_PARTITION` values for your cluster.

### 1.2 The Gym side: upstream/main, pinned

The Gym capture stack (PRs 2180/2181 and successors) is **merged to
NVIDIA-NeMo/Gym `main`**. Verified against upstream/main at `1a12168c0`
(2026-08-30): the full `nemo_gym/token_id_capture/` module (including
`delivery.py`, `lineage.py`, `fingerprint.py`, `terminal.py`), every symbol
this RL branch imports at identical module paths and signatures
(`finalize_rollout_token_capture`, `retire_rollout_token_capture`,
`token_id_capture_enabled_for_agent`, `clear_token_captures_for_rollouts`,
`TokenCaptureStore`, `TokenSource`, `TokenIdCaptureConfig`,
`token_id_capture_dirs_from_config`), the unchanged
`TokenIdCaptureSettings` yaml schema, `vllm_model_supply_prefix.yaml`,
`vllm_model_for_training.yaml`, and the `reasoning_gym_*_model_server` agent
configs. `nemo_gym/cli/__init__.py` upstream even keeps a lazy re-export table
specifically so NeMo-RL's `RunHelper` / `GlobalConfigDictParserConfig` imports
stay stable.

Use upstream/main **pinned at `1a12168c0`** (§2.2). A newer main will probably
also work — the smoke test (§5) is the gate — but `1a12168c0` is the commit
this runbook's checks were run against. The historical `stack-v2-fixes`
worktree branch is superseded and no longer needed.

### 1.3 Node.js on aarch64 (needed for claude_code and codex arms only)

The CLI auto-installers hardcode x86 Node tarballs:

- `responses_api_agents/claude_code_agent/setup_claude_code.py:29` → `node-v22.15.0-linux-x64.tar.xz`
- `responses_api_agents/codex_agent/setup_codex.py:29` → same
- `responses_api_agents/claude_code_agent/scripts/claude_code_agent_deps.sh:19` → `node-v20.18.1-linux-x64.tar.xz`

Both setup functions return early if `node`/`npm` are already on `PATH`
(checked via `shutil.which`), so the fix is to make arm64 Node available before
the agent servers start. Any one of:

- Bake Node ≥ 20 (linux-arm64) into the container image, or
- Install `https://nodejs.org/dist/v22.15.0/node-v22.15.0-linux-arm64.tar.xz`
  into a directory mounted into the job and prepend its `bin/` to `PATH` in the
  launch environment, or
- Patch the three hardcoded URLs from `linux-x64` to `linux-arm64` in the Gym
  checkout (local-only change; do not commit to the Gym branch).

Verify inside the container: `node --version && npm --version`.

The **hermes arm needs none of this** (pure Python). If Node provisioning is
blocked, run the hermes arm first (§7.3) — it exercises the identical capture
and supply stack.

### 1.4 Compute-node internet egress

First startup npm-installs the CLI (claude_code/codex) and builds per-server
Python venvs (hermes pip-installs
`git+https://github.com/cmunley1/hermes-agent@26bb847a`). If compute nodes have
no egress:

- Pre-build venvs on a login node: `uv run python examples/nemo_gym/prefetch_venvs.py`
  (run from the RL root with the Gym submodule in place).
- Pre-install CLIs with the deps scripts
  (`responses_api_agents/*/scripts/*_deps.sh`) into a mounted `DEPS_DIR`.
- The config already sets `env.nemo_gym.skip_venv_if_present: true` so present
  venvs are reused.

---

## 2. Repository assembly

### 2.1 NeMo RL

```bash
git clone https://github.com/ananthsub/RL nemo-rl-blackbox && cd nemo-rl-blackbox
git checkout ananthsub/blackbox-tokcap-experiment   # this branch (contains this runbook)
```

This branch is `grpo-grouping-through-dynamic-sampling` (upstream main as of
2026-07 + 8 `nemo-gym` commits, base commit `e9963e650` "train on rollouts from
external agent harnesses", tip `9ec032b75`) plus the `blackbox_tokcap/`
experiment directory. The 8-commit stack provides: rollout correlation-id
stamping, rebuilt-response substitution, capture metrics, per-shard id
derivation, record retirement, sampling pinned on the model server, and the
grouping-id fix for the advantage estimator.

### 2.2 Gym submodule at pinned upstream/main

```bash
git submodule update --init 3rdparty/Gym-workspace/Gym
cd 3rdparty/Gym-workspace/Gym
git remote add upstream https://github.com/NVIDIA-NeMo/Gym.git 2>/dev/null || true
git fetch upstream main
git checkout 1a12168c0   # verified pin, 2026-08-30 (see §1.2)
cd ../../..
```

The uv workspace (`pyproject.toml` lists `3rdparty/Gym-workspace/Gym` as a
member) builds from the working tree, so no lockfile or gitlink edits are
needed.

### 2.3 Verify the assembly (do not proceed on failure)

```bash
# 1. The prefix-supply model config exists:
test -f 3rdparty/Gym-workspace/Gym/responses_api_models/vllm_model/configs/vllm_model_supply_prefix.yaml \
  && echo "supply config: OK"

# 2. The capture API generation matches what the RL branch imports
#    (first run builds the venv; slow once):
uv run --locked --extra nemo_gym python -c "
from nemo_gym.token_id_capture.delivery import finalize_rollout_token_capture, retire_rollout_token_capture
from nemo_gym.token_id_capture.config import token_id_capture_enabled_for_agent
from nemo_gym.token_id_capture import TokenCaptureStore, TokenIdCaptureConfig
print('gym capture API: OK')"

# 3. Gym submodule is at the right commit:
git -C 3rdparty/Gym-workspace/Gym log --oneline -1   # expect 1a12168c0
```

If check 2 raises `ImportError`, the submodule is on the wrong Gym generation
(e.g. a pre-merge branch or a stale checkout) — redo §2.2.

---

## 3. Model download

NeMo RL pulls the policy from HuggingFace into `HF_HOME` at job start (the
launch script sets `HF_HOME=$PWD/.cache`). Pre-download on a login node so the
job does not spend walltime on ~60 GB of weights. Qwen is not gated.

```bash
HF_HOME=$PWD/.cache huggingface-cli download Qwen/Qwen3-30B-A3B-Instruct-2507
# For the smoke test (3B) and the later thinking arm:
HF_HOME=$PWD/.cache huggingface-cli download Qwen/Qwen2.5-3B-Instruct
HF_HOME=$PWD/.cache huggingface-cli download Qwen/Qwen3-30B-A3B-Thinking-2507
```

Verify: `ls .cache/hub/ | grep -i qwen` shows all three.

---

## 4. Dataset preparation

Two datasets, one per environment. `gym dataset collate` downloads the HF
artifact, validates rows, and stamps each row's `agent_ref` so NeMo Gym routes
it to the correct agent server. Run from the Gym submodule with its own venv.

### 4.1 One required config edit first

The capture-correlated claude_code config declares only the `example` dataset.
Open
`3rdparty/Gym-workspace/Gym/resources_servers/reasoning_gym/configs/reasoning_gym_claude_code_agent_model_server.yaml`
and append this entry to the existing `datasets:` list (same indentation as the
`- name: example` entry already there):

```yaml
      - name: train
        type: train
        jsonl_fpath: resources_servers/reasoning_gym/data/Nemotron-RL-ReasoningGym-v1_train.jsonl
        huggingface_identifier:
          repo_id: nvidia/Nemotron-RL-ReasoningGym-v1
          artifact_fpath: data/train.jsonl
        license: Creative Commons Attribution 4.0 International
```

(This block is copied verbatim from the sibling
`reasoning_gym_claude_code_agent.yaml`, which declares it but is NOT usable for
training — see the warning below.) Make the same edit to
`reasoning_gym_codex_agent_model_server.yaml` if you will run the codex arm.

For the hermes arm the config is `environments/hermes_math/config.yaml`
(agent `hermes_math_agent`; model_server-bound, so capture-correlated). Its
declared `train` dataset comes from an internal GitLab artifact registry
(`gitlab_identifier: dapo17k`). If you have that access, no edit is needed —
`--download` resolves it. Otherwise swap the `train` entry for the public HF
math dataset:

```yaml
        - name: train
          type: train
          jsonl_fpath: environments/hermes_math/data/OpenMathReasoning_train.jsonl
          huggingface_identifier:
            repo_id: nvidia/Nemotron-RL-math-OpenMathReasoning
            artifact_fpath: train.jsonl
          license: Creative Commons Attribution 4.0 International
```

> **Warning — do not collate against the plain (non-`_model_server`) configs.**
> `reasoning_gym_claude_code_agent.yaml` and `reasoning_gym_codex_agent.yaml`
> bind the CLI directly to `anthropic_base_url` / OpenAI with **no Gym model
> server**, so rows prepared from them route to an agent whose calls are never
> captured. Only the `_model_server` variants set
> `model_server: {type: responses_api_models, name: policy_model}`, which is
> what makes the rollout-prefix URL (and therefore capture) happen.

### 4.2 Collate

```bash
cd 3rdparty/Gym-workspace/Gym
uv venv --python 3.12 --allow-existing .venv && source .venv/bin/activate
uv sync --active --extra dev
echo "hf_token: ${HF_TOKEN}" >> env.yaml   # env.yaml is gitignored

# claude_code arm (also covers codex; same dataset):
gym dataset collate \
    --config responses_api_models/vllm_model/configs/vllm_model_for_training.yaml \
    --resources-server reasoning_gym/reasoning_gym_claude_code_agent_model_server \
    --output-dir data/reasoning_gym_claude_code \
    --mode train_preparation --download +data_source=huggingface

# hermes arm (environments/hermes_math -- --config is repeatable and composes):
gym dataset collate \
    --config responses_api_models/vllm_model/configs/vllm_model_for_training.yaml \
    --config environments/hermes_math/config.yaml \
    --output-dir data/math_hermes \
    --mode train_preparation --download +data_source=huggingface

deactivate && cd ../../..
```

If the `--resources-server <dir>/<config-stem>` selector is rejected by your
CLI version, pass the resources-server yaml as an additional `--config` instead.

### 4.3 Verify the collated data

```bash
GYM=3rdparty/Gym-workspace/Gym
ls $GYM/data/reasoning_gym_claude_code/   # note the emitted filenames
ls $GYM/data/math_hermes/

# Every row must reference the _model_server agent. Inspect one row:
head -1 $GYM/data/reasoning_gym_claude_code/*.jsonl | python3 -m json.tool | grep -i -A3 agent_ref
```

The `agent_ref` must name `reasoning_gym_claude_code_agent_model_server`
(hermes rows: `hermes_math_agent`). If it names a plain/non-model-server agent,
you collated against the wrong config — redo §4.1/§4.2.

Then set the data paths. The shipped config points at:

- `${oc.env:GYM_ROOT}/data/reasoning_gym_claude_code/train.jsonl`
- `${oc.env:GYM_ROOT}/data/reasoning_gym_claude_code/validation.jsonl`

If collate emitted different filenames (or no validation split), either rename
the emitted files to match, or override at launch:
`++data.train.data_path=... ++data.validation.data_path=...`. If there is no
validation artifact, point validation at a held-out slice of train (e.g.
`head -n 512` into a separate file) — the config only validates every 1000
steps, but the path must exist and rows must carry `agent_ref`.

---

## 5. Smoke test (single node, ~15 GPU-minutes — mandatory)

Validates branch assembly, CLI install, capture, and supply before any 30B time
is spent. Uses the recovered 3B recipes in this directory (they were validated
end-to-end on 2 GPUs at RL commit `de2d24677`; they run Qwen2.5-3B, 2 training
steps, colocated, sync GRPO).

Get an interactive shell on one GB200 node inside the container:

```bash
srun --account=$SLURM_ACCOUNT --partition=$SLURM_PARTITION \
     --nodes=1 --gres=gpu:4 --time=1:00:00 \
     --container-image=$CONTAINER_IMAGE_PATH \
     --container-mounts=$PWD:$PWD --container-workdir=$PWD \
     --no-container-mount-home --pty bash
```

Inside, run arm A (capture on, supply off), then arm B (supply on):

```bash
export GYM_ROOT=$PWD/3rdparty/Gym-workspace/Gym
export RAY_TMPDIR=/tmp RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 CLAUDE_CODE_MAX_OUTPUT_TOKENS=4096
export HF_HOME=$PWD/.cache

DATA_OVERRIDES="++data.train.data_path=$GYM_ROOT/data/reasoning_gym_claude_code/train.jsonl \
  ++data.validation.data_path=$GYM_ROOT/data/reasoning_gym_claude_code/validation.jsonl"

# A: tools (capture, no supply)
uv run python examples/nemo_gym/run_grpo_nemo_gym.py \
  --config examples/nemo_gym/blackbox_tokcap/grpo_rg_claude_code_tokcap_tools.yaml \
  $DATA_OVERRIDES

# B: supply (same + vllm_model_supply_prefix.yaml)
uv run python examples/nemo_gym/run_grpo_nemo_gym.py \
  --config examples/nemo_gym/blackbox_tokcap/grpo_rg_claude_code_tokcap_supply.yaml \
  $DATA_OVERRIDES
```

Notes:

- The recipes' committed data paths reference `rg_ext_*.jsonl` files that no
  longer exist; the `$DATA_OVERRIDES` above replaces them. Everything else runs
  as committed.
- `grpo_rg_claude_code_tokcap_smoke.yaml` (0.5B, no tool parser, single-turn by
  design) exists as a fallback if the 3B runs fail early and you need a smaller
  repro.

**Smoke PASS criteria** (from the recipe headers; check stdout/metrics):

- `token_capture/rebuilt_fraction == 1.0` — every rollout produced a trajectory.
- `token_capture/calls_per_rollout_mean > 1` — tools actually ran.
- Supply run only: the Gym model server log reports a non-zero supplied count.
- Non-zero `grad_norm` on both steps.
- On Qwen2.5, A and B should look statistically identical (its template
  preserves assistant turns; supply has nothing to repair). That null result is
  itself the pass — it proves supply doesn't corrupt anything.

Do not continue to §6 until all of these hold.

---

## 6. The 30B async run

Config: `examples/nemo_gym/blackbox_tokcap/grpo_qwen3_30ba3b_claude_code_tokcap_async.yaml`.
It inherits the upstream 30B MoE recipe (`../grpo_qwen3_30ba3b_instruct.yaml`)
and overrides:

| Block | Setting | Why |
|---|---|---|
| `grpo` | `num_prompts_per_step: 64`, `num_generations_per_prompt: 8`, `max_rollout_turns: 1` | CLI rollouts are whole agent sessions; RL sees one rollout per prompt. Scale up only after step time is known. |
| `grpo.async_grpo` | `enabled: true`, `in_flight_weight_updates: true` | the async path under test |
| `loss_fn` | IS correction + TIS (ratio 5.0), token-level loss | async requires IS correction; mirrors the async 30B recipes |
| `policy` | `Qwen3-30B-A3B-Instruct-2507`, seq len 32768 | Claude Code's system prompt + tool defs is the fixed per-call cost |
| `policy.megatron_cfg` | TP4 / **EP4** / CP1 / PP1 | GB200 = 4 GPUs/node: TP4 is one node per replica (DP = train-node count); EP4 keeps the 128-expert all-to-all intra-node (32 experts/GPU fits 186 GB); upstream's EP8/CP2 assumed 8-GPU nodes |
| `policy.generation.vllm_cfg` | `async_engine: true`, `skip_tokenizer_init: false`, TP2, mem 0.8, prefix caching on | TP2 → ~31 GB weights/GPU → 2 engines per gen node, 8 engines total. The two REQUIRED flags are explained in §9. `tool_parser: hermes` + `enable_auto_tools` are inherited from the parent. |
| `policy.generation.colocated` | `enabled: false`, 4 nodes × 4 GPUs | `async_grpo_train` asserts non-colocated (`nemo_rl/algorithms/grpo.py:4212`) |
| `env` | `should_mask_flagged_samples: true`; `nemo_gym.config_paths` = model-for-training + claude_code-on-model-server + **supply-prefix**; `token_id_capture: {enabled, all_agents, dir: /tmp/ng_tokcap, rebuild_response: false}`; sampling pinned (temp 1.0 / top_p 1.0) | the experiment payload. Capture dir is node-local on purpose: the Gym model server and the finalizing NemoGym actor are subprocess and parent on the same node. |
| `cluster` | `gpus_per_node: 4`, `num_nodes: 8` | total; train = 8 − 4 gen = 4 nodes |

Launch from the login node, RL root, on shared FS:

```bash
export GYM_ROOT=$PWD/3rdparty/Gym-workspace/Gym
WANDB_API_KEY=... HF_TOKEN=... EXP_NAME=cc-tokcap-async-b \
NUM_ACTOR_NODES=8 REPO_LOCATION=$PWD \
CONTAINER_IMAGE_PATH=/path/to/nemo-rl-nightly-aarch64.sqsh \
SLURM_ACCOUNT=... SLURM_PARTITION=... \
bash examples/nemo_gym/launch_nemo_gym_multinode_training.sh \
  --config examples/nemo_gym/blackbox_tokcap/grpo_qwen3_30ba3b_claude_code_tokcap_async.yaml
```

The launcher submits `ray.sub` and runs
`uv run python examples/nemo_gym/run_grpo_nemo_gym.py` inside the container.
Make sure `RAY_TMPDIR=/tmp`, `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0`, and
`CLAUDE_CODE_MAX_OUTPUT_TOKENS=4096` reach the job environment (add them to the
launcher's exported command if your site scrubs env vars).

Expected startup sequence: Ray cluster forms → policy loads (megatron) → vLLM
engines start on the 4 gen nodes and expose HTTP → the NemoGym actor builds the
Gym venvs (slow on first run; see §1.4) → Gym servers start (ports 5000–5999)
→ agent servers npm-install/import the CLI → rollouts begin. First-step
wall-clock is dominated by venv/CLI setup; subsequent steps by rollout
generation.

### 6.1 Split tuning

Blackbox rollouts are generation-bound (each is a full CLI session of
sequential model calls). If the trainer idles waiting on trajectories, move
nodes from train to gen — `++cluster.num_nodes=8`
`++policy.generation.colocated.resources.num_nodes=5` gives 3 train + 5 gen.
Any train-node count works at TP4/EP4 (constraint: TP×DP divisible by EP →
4·DP % 4 == 0 always). Minimum sane footprint: 3 nodes (2 train + 1 gen); a
single train node leaves the distributed optimizer unsharded (DP1) and is tight
on memory.

---

## 7. Experiment arms

Run in this order. One variable changes at a time.

### 7.1 Arm A vs B: prefix supply off/on (claude_code, Instruct-2507)

- **B (shipped config)**: `vllm_model_supply_prefix.yaml` present in
  `env.nemo_gym.config_paths`.
- **A**: delete that one line (or copy the config and remove it).
- Compare `token_capture/delivered_fraction_mean` and reward curves. On
  Instruct-2507 the arms should match — the null result that proves plumbing.
  The divergence test is arm 7.5.

### 7.2 Codex arm

In `config_paths`, replace the claude_code line with:

```yaml
    - resources_servers/reasoning_gym/configs/reasoning_gym_codex_agent_model_server.yaml
```

Same dataset (after the §4.1 edit to the codex config). Codex speaks the
Responses API over buffered SSE against the Gym model server; `codex_version`
is a required field already set in that config. Needs Node (§1.3).

### 7.3 Hermes arm

In `config_paths`, replace the claude_code line with:

```yaml
    - environments/hermes_math/config.yaml
```

and repoint the data:

```
++data.train.data_path=$GYM_ROOT/data/math_hermes/train.jsonl
++data.validation.data_path=$GYM_ROOT/data/math_hermes/validation.jsonl
```

Hermes specifics:

- Pure Python, in-process (`AIAgent` from the pinned fork
  `cmunley1/hermes-agent@26bb847a`); no Node — the easiest arm on GB200.
- Judge is OFF in this config (`should_use_judge: false`); reward is
  math-verify matching. No judge traffic through the capture path.
- Tools (terminal, file, code_execution, skills, todo) run with
  `terminal_backend: local` in the agent-server process on the actor's node at
  concurrency 32 — watch CPU there.
- Hermes injects `chat_template_kwargs: {enable_thinking: true,
  truncate_history_thinking: false}` on every chat-completions call. Inert on
  Instruct-2507; on the thinking arm it composes with the reasoning parser.
- Uniform-environment alternative: `environments/hermes_reasoning_gym/config.yaml`
  is also model_server-bound, putting all three harnesses on reasoning_gym.
  Its declared train jsonl is a local knights-knaves file — add the §4.1 HF
  reasoning-gym block (adjust `jsonl_fpath` to that environment's `data/` dir)
  or generate data with `environments/hermes_reasoning_gym/prepare.py`, then
  collate as usual.

### 7.4 If you are on 8-GPU x86 nodes instead of GB200

Override: `++cluster.gpus_per_node=8`
`++policy.generation.colocated.resources.gpus_per_node=8`
`++policy.generation.colocated.resources.num_nodes=<gen nodes>`
`++policy.megatron_cfg.expert_model_parallel_size=8`
`++policy.megatron_cfg.context_parallel_size=2`
`++policy.generation.vllm_cfg.tensor_parallel_size=4`, and skip §1.3 (x86 Node
URLs are correct). H100-class nodes have less HBM headroom than B200 —
mirror the parent recipe's `gpu_memory_utilization: 0.7`.

### 7.5 Thinking arm — where supply becomes load-bearing

Switch model and add the reasoning parser:

```
++policy.model_name=Qwen/Qwen3-30B-A3B-Thinking-2507
++policy.generation.vllm_cfg.http_server_serving_chat_kwargs.reasoning_parser=deepseek_r1
```

The Thinking template strips prior-turn `<think>` content on re-render, so
without supply the trained tokens diverge from the sampled ones. (The upstream
`grpo_qwen3_30ba3b_thinking_swe1.yaml` works around exactly this with a
hand-patched chat template that re-inserts thinking; supply is the principled
replacement.) Run A and B again here: expect B to win on
`delivered_fraction` and logprob agreement. Consider raising
`max_total_sequence_length` — thinking output is long.

---

## 8. Metrics and pass criteria

All emitted by the NemoGym actor alongside timing metrics.

| Metric | Pass | Meaning on failure |
|---|---|---|
| `token_capture/rebuilt_fraction` | == 1.0 | rollouts without a rebuildable trajectory; inspect `token_capture/rollouts_unbuilt` and the capture dir |
| `token_capture/calls_per_rollout_mean` | > 1 | == 1 → the tool parser isn't firing (tool calls returned as text; harness never loops) |
| `token_capture/parent_link_failures` | == 0 | prompts re-rendered in a way the chainer can't parent — supply/dialect bug |
| `token_capture/quarantined_fraction_mean` | ≈ 0 | ambiguous parents (identical candidate prefixes) being quarantined |
| `token_capture/delivered_fraction_mean` | A/B comparison metric | fraction of prompt tokens supplied from the parent's served tokens |
| `token_capture/masked_rollouts`, `incomplete_rollouts` | ≈ 0 | samples degraded instead of trained on |
| `grad_norm` | > 0 every step | zero → nothing trained (check masking and data routing) |
| Gym model server log | non-zero supplied count (B arms) | supply configured but never engaged |
| reward mean | > 0 and moving | flat 0 with calls==0 → agent_ref misrouting (§9) |

---

## 9. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ImportError: nemo_gym.token_id_capture.delivery` in the actor | Gym submodule on the wrong branch | §2.2; verify with §2.3 check 2 |
| Every model call 500s, `assert tokenizer is not None` in vLLM logs | `skip_tokenizer_init` resolved true (the auto-default reads `expose_http_server` before `setup_nemo_gym_config` sets it) | keep `vllm_cfg.skip_tokenizer_init: false` pinned (already in the config) |
| `calls_per_rollout_mean == 1`, invalid-tool-call penalties firing | no `tool_parser` on the vLLM HTTP server | `http_server_serving_chat_kwargs: {tool_parser: hermes}` (inherited from the parent; verify it survived your overrides) |
| Rollouts run, reward flat 0, zero tool calls, agent logs show the plain agent | dataset rows routed to the non-capture agent | re-check `agent_ref` per §4.3; re-collate against the `_model_server` config |
| claude/codex subprocess fails to spawn; npm/node errors | x86 Node tarball on Grace, or no egress | §1.3 / §1.4 |
| `AssertionError: Colocated inference is not supported for async GRPO` | `colocated.enabled` flipped true | keep `enabled: false` with explicit gen resources |
| Ray workers crash at startup, AF_UNIX path errors | Lustre path exceeds the 107-byte socket limit | `RAY_TMPDIR=/tmp` in the job env |
| Every job start rebuilds Gym venvs for minutes | venv reuse off or venvs not on shared FS | `skip_venv_if_present: true` (set) + §1.4 prefetch |
| `token_id_capture requires a directory or sink` at startup | capture enabled with no `dir` | keep `env.nemo_gym.token_id_capture.dir` set |
| Two attempts of one rollout merged in the capture store | retries with deterministic rollout ids | handled: the actor calls `clear_token_captures_for_rollouts` pre-dispatch; if you see it anyway, check for a second run pointing at the same capture dir |
| Trainer idle, gen nodes saturated | generation-bound (expected for CLI harnesses) | shift the node split toward generation (§6.1) |

---

## 10. Known limits (do not spend time on these)

- **TransferQueue**: the `run_grpo_nemo_gym.py` async path uses an in-memory
  replay buffer and never touches TQ. That is the intended scope. The TQ lane
  (Gym PRs 2774–2776, RL 3837) replaces the sink/source seams later without
  changing this run's shape.
- **SWE-bench / Terminal-Bench / CVDP cannot be capture-trained yet**: the
  `anyswe`/`anyterminal`/`cvdp` wrapper harnesses monkey-patch the model URL
  inside the task container with no rollout prefix
  (`responses_api_agents/anyswe_agent/agent_runner.py:100-130`) — zero capture
  by construction. Fixing that seam is a separate work item; do not burn a run
  discovering it.
- **Blackbox rollouts cannot be partially resumed**: the loop state lives in
  the CLI process. A restart re-runs the rollout from scratch. This experiment
  does not involve the session-state checkpointing prototype.
- `opencode_sandboxed_agent` applies no rollout prefix at all — not usable here.
- **Not every `environments/` wiring is capture-ready.** Upstream's reorg
  (Gym #2151) added per-harness environments, but only some bind
  `model_server: policy_model` (required for the rollout prefix and capture):
  `hermes_math`, `hermes_reasoning_gym`, and `claude_code_math` are
  capture-wired; `claude_code_reasoning_gym` binds `anthropic_base_url`
  directly, and `codex_math`/`codex_reasoning_gym` bind `openai_base_url`
  directly — none of those three apply the prefix, so nothing is captured.
  For claude_code and codex on reasoning_gym, use the
  `resources_servers/reasoning_gym/configs/*_model_server.yaml` configs as this
  runbook does. (`claude_code_math` is a valid optional math arm for claude,
  same gitlab-vs-HF dataset choice as hermes_math.)

---

## 11. File map

| File | What |
|---|---|
| `RUNBOOK.md` | this document |
| `grpo_qwen3_30ba3b_claude_code_tokcap_async.yaml` | the 30B GB200 async experiment config (§6) |
| `grpo_rg_claude_code_tokcap_tools.yaml` | 3B smoke, capture on / supply off (§5 arm A) |
| `grpo_rg_claude_code_tokcap_supply.yaml` | 3B smoke, capture + supply (§5 arm B) |
| `grpo_rg_claude_code_tokcap_smoke.yaml` | 0.5B minimal fallback (no tool parser; single-turn by design) |
| `grpo_rg_claude_code_tokcap.yaml` | the original longer-run 3B variant, for reference |

The four `grpo_rg_*` recipes are restored from RL commit `de2d24677` (they were
removed from the branch tip in `c1f06710b` to keep the upstreamable stack
framework-neutral). Upstream scale references:
`examples/nemo_gym/grpo_qwen3_30ba3b_instruct.yaml` (the parent of the 30B
config) and `examples/nemo_gym/grpo_qwen3_30ba3b_thinking_swe1.yaml` (the
async + non-colocated shape this config's loss/async blocks mirror).
