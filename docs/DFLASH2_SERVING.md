# Experimental DFlash2 API serving

This branch serves Qwen3.8-Flash-Next and Swift through the API with DFlash2, on one GPU or a layer split. A follow-up turn reuses its
conversation and several requests can run at once, the same as with MTP. It is an experimental port, not a released feature.

The split-stage tap transport and serial drafting integration are adapted from
[upstream draft PR #3869](https://github.com/Niko1221/Strata/pull/3869), commit
`5f436de2c3a9fbd36d2efa66629015be11308dcc`. That author's measurements are not validation of this branch.
Tesla P100 cards also need the VMAD fix of [PR #4274](https://github.com/Niko1221/Strata/pull/4274); the measurements below were made with it.

## Build for the mixed NVIDIA pool

P100, RTX 2080 Ti and RTX 3060 need a CUDA 12 build containing SM60, SM75 and SM86 code. Run this from this branch's
source checkout on the Docker host. These are shell commands for Linux:

```sh
docker build -t strata:dflash2-cuda12 \
  --build-arg STRATA_COMMIT=$(git rev-parse HEAD) \
  --build-arg CUDA_IMAGE=nvidia/cuda:12.9.1-devel-ubuntu24.04 \
  --build-arg 'CUDA_ARCHITECTURES=60;75;86' \
  --build-arg BUILD_VISION=1 .
```

Keep the GPU selection, model paths, volume mounts and API key from your existing container. Change its image and add:

```text
DRAFTING=dflash2
```

Recreate the container to apply its environment. To switch back, set `DRAFTING=mtp`. Conversations are kept in memory, so a switch reads
them again.

The default Docker entrypoint reads `DRAFTING`. A container that starts `strata serve --profile FILE` skips the entrypoint, and `DRAFTING`
still works there: it overrides the file's `"drafting"` choice at each start, and the file is not changed. `serve` does not download the
drafter: it must already be at `dflash2/qwen3.8-flash-next-dflash2.q8.pack` under the data folder of the file's `--pack` or `--mtp` path.
The file's `"exe"` must be this image's engine. With the default entrypoint, the Terminal downloads and packs the pinned drafter once.

The root Dockerfile builds both this engine and the Terminal and sets `STRATA_DFLASH2_SERVE=1`. That marker permits
the Terminal to download the drafter and pass its flags. It does not add capabilities to an older engine. Published
images and `docker/Dockerfile` use packaged engines, so their behavior stays unchanged.

## How it works

- **A request running alone** drafts with DFlash2. **Requests running at the same time** use batch slots without drafts, which is what MTP
  does unless `--batch-mtp` is set. When one is left, it drafts again.
- **Conversation reuse** is the normal cache: checkpoints, parked conversations and slots hold the main layers. The drafter's own memory is
  its ring, the last 2064 rows. A row depends only on the tokens before it, so a saved ring stays right for any later prompt that starts
  with those tokens. The engine keeps the live ring and up to twice `--conversation-cache-slots` saved copies, about 81 MiB each in RAM, and
  looks them up by the tokens. They are not counted in `--conversation-cache-mib`, but they stop at its RAM floor.
  `STRATA_DF2_RING_CACHE=0` turns the saved copies off. Measured, they add about 3 points of accepted drafts after a switch, and that varies by prompt.
- **Rows it cannot find are zeroed.** That can happen after a reply decoded in a slot, or after the saved copy was evicted. It can cost
  accepted drafts until new rows fill the ring. The main model still checks every draft, so it should not change the answer.

## Current limits

- `--batch-mtp` is refused: DFlash2 has no per-slot drafter. The engine says so and MTP runs.
- Serial verify windows only. A pipelined start is turned off for DFlash2 by itself; an explicit `--pipeline-windows` above zero falls back to
  MTP with a startup message.
- Session files (`SAVE`, `RESTORE`) are refused with DFlash2 on one GPU. With a layer split or parallel requests they are refused for every drafter.
- Head aliasing is not supported. A late DFlash2 binding or graph preparation failure stops startup rather than claiming
  to use an MTP layer that was not loaded.
- Model compatibility is the shared catalog's `qwen`, `gyro` and `swift` families. The drafter was trained on the original
  Qwen3.8-Flash-Next weights; Swift is a fine-tune with the same layers and tokenizer. Coder, DeepSeek and other families stay on MTP.
- HIP and SYCL builds and hardware tests must be recorded before this is ready for upstream review. No speedup is claimed.

## Measured

[The full report](../bench/results/2026-11-02-community-6gpu-dflash2-main-0.1.55/README.md) has the runs, data and limits: one machine with six mixed cards, engine 0.1.55, 129,282 output tokens in its main runs,
Swift 1.5 and original IQ3_S, two slots. MTP on this branch matched MTP on upstream main. Follow-up turns reused 99.7 to 99.8% of their tokens with either drafter. DFlash2 was 1.5 to 4% slower than MTP over all decode
prompts and about 1 s slower per fresh prompt read, and 1.0 to 3.4% faster on forced tool calls. Wider windows (`--spec 6`, `--spec 8`) helped it against MTP on code, but both were slower than at `--spec 4`.
Swift 1.5 and the original ran at about the same speed. No speedup is claimed.

## Checks before deployment

Use a separate test container and preserve the working image and configuration. Verify the startup log actually says
`DFlash2 drafter on CUDA...`; a `--draft dflash2 is off` message means MTP ran instead.

Check short and long prompts, repeated requests, alternating unrelated conversations, and cancellation followed by
another request. Exercise both streaming and non-streaming API calls and tool-call output. Check the logs for non-finite
values and GPU errors. Compare against MTP from the same source revision, model and hardware; record failures as well as successful runs.

## Possible follow-up

Drafting inside batch slots (the DFlash2 version of `--batch-mtp`) would share the drafter weights across slots and give each active request
its own ring, position state and graph buffers. Test interleaved requests against isolated runs before enabling it.
