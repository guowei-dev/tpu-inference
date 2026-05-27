# XLA autotune

One-factor-at-a-time (OFAT) sweep over XLA / libtpu flags, driven from
Buildkite.  Spins up `vllm serve` on a TPU pod, runs `benchmark_serving`
against it, and records the target metric (default `total_token_throughput`)
per candidate flag.

## Layout

```
.buildkite/xla_autotune/
├── autotuner.py             OFAT driver (CLI: --flag-list-file, --slice-{index,count}, ...)
├── vllm_test_framework.py   vllm serve + benchmark_serving driver, production defaults
├── flags.txt                Candidate flags, one per line
├── pipeline.yml             Buildkite pipeline definition (4-shard matrix)
└── README.md
```

The host-side wrapper that runs inside each Buildkite step lives at
`.buildkite/scripts/run_xla_autotune_shard.sh`.  It is responsible for
launching `autotuner.py` inside the docker container and incrementally
uploading per-trial JSON + log bundles to Buildkite artifacts as soon as
they land.  It also `git clone`s
[`kimbochen/bench_serving`](https://github.com/kimbochen/bench_serving)
into the docker workdir, matching the convention used by
`tests/e2e/benchmarking/bm_qwen3_coder.sh` so every serving benchmark in
the repo shares one harness.

## How it works

For a flag list of length `F` and a matrix of `N` shards:

* Shard `k` (1-based) gets `flags[(k-1)*ceil(F/N) : k*ceil(F/N)]` as its
  candidate flags.
* Each shard runs `AUTOTUNE_BASELINE_RUNS` baseline trials (no extra
  LIBTPU args) followed by one trial per candidate flag.
* Every trial runs `warmup_runs` warmup benchmark passes (results
  discarded) plus one measured pass per `(input_len, output_len)` shape.
  The warmup pass exists because the first benchmark after engine init
  reflects compile + cache-warm transients, not steady-state throughput.

Trial outputs:

* `<artifact-dir>/<trial_id>.json` — per-trial record (success, target
  metric, full benchmark metrics dict, duration, error).
* `<artifact-dir>/summary.jsonl` — same records, one per line, appended
  as trials finish.
* `<artifact-dir>/logs/<trial_id>_EXP_<ts>/` — full log bundle (vllm
  server log, env dump, benchmark stdout, the constructed commands).

All of the above ship to Buildkite as artifacts under
`shard_<k>_of_<N>/...`.

## Triggering a run

The pipeline file is `.buildkite/xla_autotune/pipeline.yml`.  Either
configure a Buildkite pipeline whose upload step is
`buildkite-agent pipeline upload .buildkite/xla_autotune/pipeline.yml`,
or trigger via the REST API:

```
curl -s -X POST -H "Authorization: Bearer $BUILDKITE_API_TOKEN" \
  -H "Content-Type: application/json" \
  "https://api.buildkite.com/v2/organizations/<org>/pipelines/<pipeline>/builds" \
  -d '{
    "commit": "<full-40-char-sha>",
    "branch": "<branch>",
    "message": "xla autotune",
    "ignore_pipeline_branch_filters": true
  }'
```

To trigger from a fork, set `branch` to the namespaced form
`<fork-owner>:<branch>`.

## Knobs

Pipeline `env:` block (with defaults):

| Var                       | Default                              | Notes                                                            |
|---------------------------|--------------------------------------|------------------------------------------------------------------|
| `AUTOTUNE_MODEL`          | `Qwen/Qwen3.5-397B-A17B-FP8`         | Must be a key in `DEFAULT_MODEL_CONFIGS`.                        |
| `AUTOTUNE_TARGET_METRIC`  | `total_token_throughput`             | Any key inside the benchmark_serving result JSON.                |
| `AUTOTUNE_BASELINE_RUNS`  | `2`                                  | ≥2 recommended so per-shard noise floor can be estimated.        |
| `AUTOTUNE_FLAGS`          | `.buildkite/xla_autotune/flags.txt`  | One `--flag=value` per line; `#` comments ignored.               |
| `AUTOTUNE_CONFIG`         | _(unset)_                            | Optional JSON of `VLLMTestParam` field overrides.                |

## Onboarding a new model

1. Add the model's `vllm serve` args and env vars to
   `DEFAULT_MODEL_CONFIGS` / `DEFAULT_MODEL_ENV_CONFIGS` /
   `DEFAULT_MODEL_BENCHMARK_CONFIGS` in `vllm_test_framework.py`.
2. Set `AUTOTUNE_MODEL` in `pipeline.yml`.
3. Confirm the model's persistent JAX cache namespace (in
   `run_in_docker.sh`) is appropriate.

## Cache notes

Engine startup is dominated by HLO compile.  Cold-cache init for the 397B
model is ~90 min; warm-cache init drops to ~10 min.  The TPU runner
pulls/pushes a persistent JAX cache from GCS in `run_in_docker.sh`, so
once a candidate flag's HLO modules have been compiled once on any VM
they are reused on subsequent runs.

JAX keys each compiled HLO module on the `(flag-set × shape)` tuple, so a
new candidate flag typically only invalidates the small subset of modules
it actually affects — observed +2–4 % init overhead vs baseline on the
397B model, not +90 min.

## Interpreting results

Always look at the baselines first.  If the baseline runs have a wide
spread (>10 %) the candidate signal is most likely noise.  For tighter
attribution: bump `AUTOTUNE_BASELINE_RUNS` and add candidate replicates
by appending the same flag multiple times in `flags.txt`.
