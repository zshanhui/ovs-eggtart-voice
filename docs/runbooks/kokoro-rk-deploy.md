# Kokoro RK Deploy Runbook

One-page deployment guide for the production Kokoro RK image on Radxa Rock
5B / 5B+ (RK3588). For reproduction from-scratch (artifact build, audio
parity, R&D decisions), see
`docs/specs/kokoro-rk-34pct-reproduction-guide.md`.

## TL;DR

The 2026-05-23-rebuilt image is **self-contained**: it bakes in the misaki
ZH G2P stack, the speaker_id fix (commit `6155ebe`), the 4-stage runtime +
3-bucket router (submodule `65b9a13` on
`feat/kokoro-rk-4stage-vocoder-front`), and all 14 active bucket-{8,16,32}
artifacts. No host bind-mount of model files or hot-patched `.py` files is
required.

```bash
docker pull <registry>/openvoicestream:rk-kokoro-2026-05-23-rebuilt   # or local build, see §4

docker run -d --name openvoicestream-kokoro --restart=unless-stopped \
  --privileged \
  --network host \
  -v /dev:/dev \
  -v /proc/device-tree/compatible:/proc/device-tree/compatible:ro \
  -v rk-asr-models:/opt/asr/models \
  -v $(pwd)/configs:/opt/speech/configs:ro \
  -v $(pwd)/deploy/artifacts:/opt/speech/deploy/artifacts:ro \
  --group-add video \
  \
  -e OVS_PROFILE=rk3588-kokoro-rknn-34pct \
  -e RK_ARTIFACT_SET=rk3588-kokoro-hybrid-34pct-2026-05-23 \
  -e RK_ARTIFACT_AUTO_DOWNLOAD=0 \
  \
  `# Bucket-32 (default seq_len=32; baked at /opt/kokoro-rknn)` \
  -e KOKORO_RKNN_VOCODER_FRONT_PATH=kokoro-vocoder-front-half.native.fp16.rknn \
  -e KOKORO_RKNN_TAIL_REST_PATH=kokoro-vocoder-tail-rest-cpu.onnx \
  -e KOKORO_RKNN_TAIL_REST_INT8_PATH=/opt/kokoro-rknn/kokoro-vocoder-tail-rest-cpu.int8.onnx \
  \
  `# Bucket-8 router (short sentences ≤ 8 phonemes)` \
  -e KOKORO_RKNN_BUCKET8_PREFIX_PATH=/opt/kokoro-bucket-8/kokoro-prefix-cpu-bucket8.onnx \
  -e KOKORO_RKNN_BUCKET8_DECODER_FRONT_PATH=/opt/kokoro-bucket-8/rk3588/kokoro-decoder-front-bucket8.fp16.rknn \
  -e KOKORO_RKNN_BUCKET8_VOCODER_FRONT_PATH=/opt/kokoro-bucket-8/rk3588/kokoro-vocoder-front-half-bucket8.native.fp16.rknn \
  -e KOKORO_RKNN_BUCKET8_TAIL_REST_PATH=/opt/kokoro-bucket-8/kokoro-vocoder-tail-rest-cpu-bucket8.onnx \
  -e KOKORO_RKNN_BUCKET8_TAIL_REST_INT8_PATH=/opt/kokoro-bucket-8/kokoro-vocoder-tail-rest-cpu-bucket8.int8.onnx \
  -e KOKORO_RKNN_BUCKET8_TAIL_REST_INT8STATIC_PATH=/opt/kokoro-bucket-8/kokoro-vocoder-tail-rest-cpu-bucket8.int8static.onnx \
  \
  `# Bucket-16 router (mid sentences 9–16 phonemes)` \
  -e KOKORO_RKNN_BUCKET16_PREFIX_PATH=/opt/kokoro-bucket-16/kokoro-prefix-cpu-bucket16.onnx \
  -e KOKORO_RKNN_BUCKET16_DECODER_FRONT_PATH=/opt/kokoro-bucket-16/kokoro-decoder-front-bucket16.fp16.rknn \
  -e KOKORO_RKNN_BUCKET16_VOCODER_FRONT_PATH=/opt/kokoro-bucket-16/kokoro-vocoder-front-half-bucket16.native.fp16.rknn \
  -e KOKORO_RKNN_BUCKET16_TAIL_REST_PATH=/opt/kokoro-bucket-16/kokoro-vocoder-tail-rest-cpu-bucket16.onnx \
  -e KOKORO_RKNN_BUCKET16_TAIL_REST_INT8_PATH=/opt/kokoro-bucket-16/kokoro-vocoder-tail-rest-cpu-bucket16.int8.onnx \
  \
  -e RK_ARTIFACT_MANIFEST=/opt/speech/deploy/artifacts/rk_manifest.json \
  \
  openvoicestream:rk-kokoro-2026-05-23-rebuilt \
  python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8621
```

## 1. Pre-flight

```bash
# Confirm host has the RK userspace devices.
ls /dev/dri /dev/dma_heap /dev/rga /dev/mpp_service

# Confirm Docker arm64.
docker version | grep -i arch

# Disk: image ≈ 1.1 GB, plus base ≈ 138 MB. Need ≥ 2 GB free in /var/lib/docker.
df -h /var/lib/docker

# The TL;DR `docker run` uses `$(pwd)/configs` and `$(pwd)/deploy/artifacts`
# bind-mounts — run it from a `seeed-local-voice/` checkout root (those paths
# must exist on the host). Also creates a named volume `rk-asr-models` for
# ASR model cache (auto-populated on first start). RK NPU access requires
# `--privileged` + `-v /dev:/dev` + bind of `/proc/device-tree/compatible`
# (RKNN runtime aborts with `_check_container` failure otherwise).
```

## 2. Image acquisition

Two paths:

- **Pulled from registry** (preferred when a private registry is wired
  up): `docker pull <registry>/openvoicestream:rk-kokoro-2026-05-23-rebuilt`.
- **Local build on a Rock 5B** (no cross-compile needed). See §4.

## 3. Container launch

Single `docker run` command above. After it returns, check:

```bash
docker logs openvoicestream-kokoro 2>&1 | grep -E 'Kokoro (4-stage|bucket-)|misaki' | head -10
# Expected:
#   Kokoro 4-stage path active: vocoder_front=... tail_rest=...
#   Kokoro bucket-8 router enabled:  ... (threshold n_tokens<=8)
#   Kokoro bucket-16 enabled: ... (threshold 8<n_tokens<=16)
#   Loaded Kokoro hybrid: prefix=... front=... vocoder=... tail=...
#   misaki ZH G2P (v1.1) loaded for Kokoro RKNN

curl -s http://localhost:8621/health
# {"tts":true,"tts_backend":"rk:kokoro_rknn","asr":true,...}
```

If any line is missing → `docker logs` for full error, then see §6.

### Smoke test (30 seconds)

```bash
for text in 'abc.' '你好。' 'Hello world.' '我感觉很棒。' 'hello world. how are you today?'; do
  curl -sS -o /tmp/t.wav -w "  text='$text' size=%{size_download} http=%{http_code} ttfa=%{time_starttransfer}\n" \
       -X POST http://localhost:8621/tts/stream \
       -H 'Content-Type: application/json' \
       -d "{\"text\":\"$text\"}"
done
```

Expected TTFA (single-shot, cold):

| Text                                  | Bucket | size (bytes)   | TTFA      |
| ------------------------------------- | :----: | -------------- | --------- |
| `abc.`                                | 8      | ~43 000        | ≤ 1.0 s   |
| `你好。`                              | 8      | ~30–50 000     | ≤ 1.0 s   |
| `Hello world.`                        | 16     | ~85–95 000     | ≤ 2.0 s   |
| `我感觉很棒。`                        | 16     | ~85–95 000     | ≤ 2.0 s   |
| `hello world. how are you today?`     | 32     | 251 908        | ≤ 3.5 s   |

`size < 200 B` on any case → bucket mis-routing or misaki failure; check
logs.

## 4. Build the image locally (on a Rock 5B host)

Build context lives under `seeed-local-voice/` checkout root. Bucket
artifacts are large (≈ 450 MB) and not committed to git — they must be
staged into `deploy/kokoro-artifacts/` first (see
`docs/specs/kokoro-rk-34pct-reproduction-guide.md` §3.B for HF download
sources).

```bash
# 1. Clone main repo + submodule at the right pins.
git clone https://github.com/suharvest/openvoicestream seeed-local-voice
cd seeed-local-voice
git submodule update --init --recursive
( cd third_party/rkvoice-stream && git checkout feat/kokoro-rk-4stage-vocoder-front )

# 2. Stage bucket artifacts (path layout matches Dockerfile.rk COPY).
#    See reproduction-guide §3.B; bucket-32 → deploy/kokoro-artifacts/bucket-32/,
#    bucket-8 → bucket-8/ (with rk3588/ subdir), bucket-16 → bucket-16/ (flat).
mkdir -p deploy/kokoro-artifacts/{bucket-8/rk3588,bucket-16,bucket-32}
# (download from HF harvestsu/seeed-local-voice-rk-artifacts/rk3588/kokoro-hybrid-v1/ …)

# 3. Build (arm64-native on a Rock 5B; ~25–30 min including misaki pip install).
docker build -f deploy/docker/Dockerfile.rk \
  -t openvoicestream:rk-kokoro-2026-05-23-rebuilt .
```

For build-host non-CN PyPI override:

```bash
docker build --build-arg PIP_INDEX=https://pypi.org/simple ...
```

## 5. Roll back

The previous image `openvoicestream:rk-kokoro-2026-05-23` (pre-rebuild) is
retained on the radxa as the rollback target. To revert:

```bash
docker stop openvoicestream-kokoro
docker rm openvoicestream-kokoro
# Re-run with the hot-patch overlay recipe from
# docs/specs/kokoro-rk-34pct-reproduction-guide.md §3.D
# (bind-mounts /tmp/fixed-tts.py, /tmp/fixed-kokoro_rknn.py, and
#  /home/radxa/models/tts/kokoro-bucket-* at the same destinations).
```

The misaki pip install in the writable layer of the old container is
*destroyed* by `docker rm` — re-run `docker exec ... pip3 install
'misaki[zh]'` after the rollback launch.

## 6. Troubleshooting

| Symptom                                        | Likely cause                                                                 | Fix                                                                                  |
| ---------------------------------------------- | ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| `bucket-N router enabled` line missing on boot | One of the four `KOKORO_RKNN_BUCKET{N}_*` env vars unset                     | Compare your `-e` block against §3 above.                                            |
| `misaki ZH G2P (v1.1) loaded` missing          | Running an old (pre-rebuild) image                                          | Image must be `2026-05-23-rebuilt` or newer. `docker inspect ... \| grep Image`.    |
| TTFA > 5 s on `abc.`                           | Falling through bucket-8 → bucket-32 (env or artifact missing)              | Check logs for `bucket=32 num_tokens=4` (router mis-fired).                          |
| Chinese smoke returns 4 B WAV                  | misaki not installed (legacy image)                                         | Rebuild image (§4) or fall back to hot-patch deploy + `pip3 install misaki[zh]`.    |
| `/opt/kokoro-rknn/...` not found at startup    | Image is the *baked-in* one but env var points at a non-existent file       | Either correct the env var or remove it (loader has graceful fallback).             |

## 7. Reference

- Reproduction guide (from-scratch, with R&D context): `docs/specs/kokoro-rk-34pct-reproduction-guide.md`
- R&D closure: `docs/specs/kokoro-rk-perf-r-and-d-closure.md`
- Bucket-8 perf: `docs/specs/kokoro-rk-bucket8-ttfa.md`
- Bucket-16 perf: `docs/specs/kokoro-rk-bucket16-mid-ttfa.md`
- HTTP RTF final: `docs/specs/kokoro-rk-34pct-http-rtf-final.md`
- HF artifact mirror: `harvestsu/seeed-local-voice-rk-artifacts/rk3588/kokoro-hybrid-v1/`
