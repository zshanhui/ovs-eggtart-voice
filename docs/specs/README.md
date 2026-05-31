# `docs/specs/` index

Engineering specs, design notes, and post-mortems for major workstreams in
`seeed-local-voice`. Skim the per-area index below for the right entry
point; the reproduction guides ("start here" rows) are the recommended
on-ramps.

## Kokoro RK (Radxa / RK3588) — TTS NPU optimization

Workstream goal: get the Kokoro v1.0 TTS to a viable RTF and TTFA on
RK3588 NPU. Final shipped state (2026-05-23): 34 % NPU residency
(decoder-front INT8 + vocoder front-half FP16), 4-stage pipeline with a
3-bucket dynamic router (seq_len 8 / 16 / 32) and misaki ZH G2P. HTTP RTF
0.59, short-EN TTFA 0.79 s, mid TTFA 1.6 s.

| Spec | Date | One-liner |
| --- | --- | --- |
| **[kokoro-rk-34pct-reproduction-guide.md](kokoro-rk-34pct-reproduction-guide.md)** ← start here | 2026-05-23 | End-to-end reproduction recipe (artifacts + env + verify) |
| [kokoro-rk-npu-42pct.md](kokoro-rk-npu-42pct.md) | 2026-05-22 | Original R&D plan; target revised 42 % → 34 % after BERT dead-code finding |
| [kokoro-rk-42pct-m1-boundary-report.md](kokoro-rk-42pct-m1-boundary-report.md) | 2026-05-22 | M1 — subgraph boundary tensor discovery |
| [kokoro-bert-ab-audio-report.md](kokoro-bert-ab-audio-report.md) | 2026-05-22 | A/B audio proves BERT is bit-exact dead code in Kokoro v1.0 ONNX |
| [kokoro-rk-42pct-m2-bert-fp16.md](kokoro-rk-42pct-m2-bert-fp16.md) | 2026-05-22 | M2 — BERT FP16 RKNN built but not wired (per A/B finding) |
| [kokoro-rk-42pct-m4-vocoder-fp16.md](kokoro-rk-42pct-m4-vocoder-fp16.md) | 2026-05-22 | M4 — vocoder front-half native FP16 RKNN (Sin polynomial rejected) |
| [kokoro-rk-34pct-m4m6-final.md](kokoro-rk-34pct-m4m6-final.md) | 2026-05-22 | 4-stage runtime + M6 manifest; opt-in (audio PASS, initial RTF FAIL) |
| [kokoro-rk-34pct-perf-diagnostic.md](kokoro-rk-34pct-perf-diagnostic.md) | 2026-05-22 | Per-stage timing investigation of the initial 4-stage RTF gap |
| [kokoro-rk-34pct-rtf-reconciliation.md](kokoro-rk-34pct-rtf-reconciliation.md) | 2026-05-23 | Root-caused the RTF gap to broken in-image `tts.py` |
| [kokoro-prod-image-stale-2026-05-23.md](kokoro-prod-image-stale-2026-05-23.md) | 2026-05-23 | Documented stale image + bind-mount workaround |
| [kokoro-rk-34pct-http-rtf-final.md](kokoro-rk-34pct-http-rtf-final.md) | 2026-05-23 | 4-stage promoted to default — HTTP RTF 0.59 (-10 % vs 3-stage) |
| [kokoro-rk-streaming.md](kokoro-rk-streaming.md) | 2026-05-23 | Per-sentence streaming context |
| [kokoro-rk-zh-mid-sentence-silent-fail-diag.md](kokoro-rk-zh-mid-sentence-silent-fail-diag.md) | 2026-05-23 | Root-caused the ZH silent fail (char-level lookup misses Han) |
| [kokoro-rk-zh-fix-misaki.md](kokoro-rk-zh-fix-misaki.md) | 2026-05-23 | misaki v1.1 ZH G2P wired at the tokenizer front-end |
| [kokoro-rk-bucket8-ttfa.md](kokoro-rk-bucket8-ttfa.md) | 2026-05-23 | Bucket-8 router shipped — short EN TTFA 0.79 s |
| [kokoro-rk-bucket16-mid-ttfa.md](kokoro-rk-bucket16-mid-ttfa.md) | 2026-05-23 | Bucket-16 router shipped — mid TTFA 1.5-1.6 s; misaki baked into Dockerfile.rk |

## ASR / TTS worker concurrency (Jetson)

| Spec | One-liner |
| --- | --- |
| [voice-pipeline-concurrency-plan.md](voice-pipeline-concurrency-plan.md) | Overall multi-client concurrency plan for the voice pipeline |
| [asr-worker-concurrency.md](asr-worker-concurrency.md) | ASR worker concurrency design |
| [asr-n2-phase-b-patches.md](asr-n2-phase-b-patches.md) | ASR N=2 Phase B patches |
| [tts-worker-concurrency.md](tts-worker-concurrency.md) | TTS worker concurrency design |
| [tts-worker-cancel-protocol.md](tts-worker-cancel-protocol.md) | Cooperative cancel protocol A-at-N=1 (landed 2026-05-22) |
| [tts-n2-throughput.md](tts-n2-throughput.md) | TTS N=2 throughput investigation |
| [tts-n2-shared-tensor-audit.md](tts-n2-shared-tensor-audit.md) | Shared-tensor audit for N=2 |
| [tts-n2-phase-b-patches.md](tts-n2-phase-b-patches.md) | TTS N=2 Phase B stability patches (landed 2026-05-22) |

## MOSS-TTS-Nano port (in progress)

| Spec | One-liner |
| --- | --- |
| [moss-tts-nano-kv-adapter.md](moss-tts-nano-kv-adapter.md) | KV adapter design (flat FP32 → paged FP16) |
| [moss-tts-nano-paged-kv-cpp.md](moss-tts-nano-paged-kv-cpp.md) | Paged KV C++ implementation notes |
