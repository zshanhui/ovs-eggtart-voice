# Paraformer TRT Transcription CER Optimization Plan

Date: 2026-04-28

Scope: this plan is grounded in `app/backends/paraformer_trt.py` and the observed 8-sample CER error set. I also checked upstream sherpa-onnx `sherpa-onnx/csrc/offline-paraformer-model.cc` on GitHub. That file is model/session plumbing, not final token detokenization: it exposes model metadata such as `vocab_size`, `lfr_window_size`, `lfr_window_shift`, `neg_mean`, and `inv_stddev`, but does not show `@@` BPE string merging in the model wrapper. Therefore the BPE fix below is derived from the local token decoding path and the observed token text form. Hypothesis vs observed fact is labeled explicitly.

## Section 1: Error Pattern Quantification

### P1: Adjacent character omission

Frequency: 2/8 samples directly observed. S3 has `"实别"` where expected text is `"识别"`, and `"语合成"` where expected text is `"语音合成"`.

Severity: Medium. The missing character changes meaning, but this pattern is less catastrophic than the English `@@` fragmentation because it is sparse in the 8-sample set.

Root cause: mixed implementation/model hypothesis.

Observed implementation facts:

- CIF emits one acoustic embedding whenever accumulated alpha crosses `CIF_THRESHOLD = 1.0` (`app/backends/paraformer_trt.py:77`, `app/backends/paraformer_trt.py:235-240`). If alpha mass under-fires for a short adjacent syllable, no decoder token is ever requested for that character.
- The CIF loop carries residual mass across chunks, but only fires complete threshold crossings and returns any residual as `carry_weight` / `carry_embed` (`app/backends/paraformer_trt.py:223-243`).
- In offline `transcribe()`, the encoder path pads the last chunk with zero stacked frames before CIF (`app/backends/paraformer_trt.py:654-658`) and then feeds padded-frame encoder output through CIF (`app/backends/paraformer_trt.py:660-669`). This can change alpha distribution near chunk/tail boundaries.

Hypothesis:

- If the same omissions also appear in the sherpa-onnx ONNX path with the same model, treat P1 primarily as model capability / acoustic confusion. `"识别"` vs `"实别"` and missing `"音"` are plausible acoustic-language model errors rather than pure TRT defects.
- If sherpa-onnx produces the correct S3 text while TRT omits characters, inspect alpha sums and per-frame alphas around the omitted syllables. The local CIF implementation is simple and mathematically plausible, but it does not use `tail_threshold` inside `cif()` despite accepting the parameter (`app/backends/paraformer_trt.py:194-199`, `app/backends/paraformer_trt.py:242-243`), so all final-tail policy is outside the loop.

### P2: Adjacent character over-emit / repetition

Frequency: 3/8 samples directly observed. S1 has `"今天气"`, S2 has `"智能智能"`, and S5 has `"greatday嗯greatday"`.

Severity: Medium to High. Chinese duplicates are localized; English phrase repetition can dominate short utterances and heavily inflate CER/WER.

Root cause: implementation bug plus threshold sensitivity.

Observed implementation facts:

- `decode_ids()` suppresses only immediate adjacent *ID* repeats via `if tid == prev_tid: continue` (`app/backends/paraformer_trt.py:278-290`). It does not suppress non-adjacent phrase repeats or equivalent-token repeats that use different IDs.
- The offline `transcribe()` path decodes each decoder output separately and appends strings (`app/backends/paraformer_trt.py:649`, `app/backends/paraformer_trt.py:679-683`, `app/backends/paraformer_trt.py:697`). This means adjacent duplicate IDs across decoder-call boundaries are not visible to `decode_ids()`.
- The streaming path is better for this specific issue: it extends `_all_token_ids` and decodes the full accumulated ID sequence (`app/backends/paraformer_trt.py:419-423`). Offline should mirror that behavior.
- CIF fires while `accum_weight >= threshold` (`app/backends/paraformer_trt.py:235`). If alpha mass spikes around a strong phoneme or padded tail, multiple embeddings can be emitted in a tight region, which can present as over-emission. `CIF_THRESHOLD` is currently hard-coded to 1.0 (`app/backends/paraformer_trt.py:77`).

Hypothesis:

- The immediate safe implementation fix is to accumulate all token IDs in offline `transcribe()` and decode once. That improves cross-call dedup and also enables BPE merging across decoder-call boundaries after P3.
- Phrase-level repetition such as `"greatday嗯greatday"` may still require decoder-cache reset/tuning, a minimum alpha sanity check, or post-decode phrase repetition suppression. Add phrase suppression only after logging token IDs; otherwise it may hide legitimate repeats.

### P3: English BPE `@@` suffix not detokenized

Frequency: 3/8 samples directly observed. S4 has `"hel@@lonicetoetyou"`, and S6/S7 are described as full of `"@@"` fragments.

Severity: High. This is the highest-ROI defect because it is deterministic text post-processing. It can turn otherwise correct English subword output into many CER errors immediately.

Root cause: implementation bug.

Observed implementation facts:

- `load_tokens()` strips optional trailing numeric IDs and preserves token text (`app/backends/paraformer_trt.py:250-268`). A token such as `hel@@ 1234` would be loaded as `hel@@`, which is correct.
- `decode_ids()` appends token strings directly to `chars` and returns `"".join(chars)` (`app/backends/paraformer_trt.py:285-291`). There is no branch that detects or strips `@@`.
- The function docstring says it filters special tokens and suppresses immediate adjacent-ID repeats only (`app/backends/paraformer_trt.py:271-276`); it does not claim BPE support.

Sherpa-onnx reference:

- The checked upstream Paraformer C++ wrapper (`offline-paraformer-model.cc`) reads model metadata including global CMVN-style stats `neg_mean_` and `inv_stddev_`, and exposes them through accessors. It does not contain the final `@@` merge logic in that file. That is important: sherpa-onnx model forward code and text detokenization are separated. Local TRT currently has no equivalent detokenizer layer, so the local fix belongs in `decode_ids()`.

### P4: Tail truncation

Frequency: 1/8 samples directly observed. S3 ends with `"水"` instead of `"水平"`.

Severity: Medium. Tail truncation is localized but user-visible, and can affect command endings or short utterances.

Root cause: implementation threshold/tail policy hypothesis.

Observed implementation facts:

- `CIF_TAIL_THRESHOLD` is 0.5 (`app/backends/paraformer_trt.py:78`).
- Offline `transcribe()` flushes one final token only if `carry_w >= CIF_TAIL_THRESHOLD` (`app/backends/paraformer_trt.py:685-695`).
- Streaming finalization uses the same threshold in `_flush_cif_tail()` (`app/backends/paraformer_trt.py:461-475`).
- `decode_ids()` skips EOS but does not stop on EOS (`app/backends/paraformer_trt.py:281-282`), so this is unlikely to be caused by EOS prematurely ending the string. EOS skip can hide decoder artifacts, but it is not a stop condition.

Hypothesis:

- If `"平"` is represented by tail alpha mass below 0.5, the current flush policy will drop it. Lowering the tail threshold to 0.3 is a small, testable change. The risk is extra spurious final tokens on silence/noise, so this should be validated against at least the 8 wavs plus a few silence/short-command clips.

### P5: CMVN per-utterance vs global

Frequency: 8/8 potential exposure. Every sample flows through the local `compute_fbank()`.

Severity: Medium. If training/inference reference used model metadata global CMVN, local per-utterance CMVN can shift features and affect all token timings/logits. If the TRT-exported encoder was calibrated for per-utterance CMVN, then this is not a defect.

Root cause: configuration/feature parity risk.

Observed implementation facts:

- Local feature extraction applies utterance-level mean/std normalization for every call (`app/backends/paraformer_trt.py:163-167`).
- Upstream sherpa-onnx Paraformer model wrapper reads `neg_mean` and `inv_stddev` metadata and exposes `NegativeMean()` / `InverseStdDev()` accessors. That strongly suggests sherpa-onnx can use model-provided global CMVN stats for Paraformer models, although the exact application point is outside the checked file.
- Local `paraformer_trt.py` has no obvious configuration branch to choose global CMVN stats in `compute_fbank()`; the function signature accepts only `audio` and returns normalized features (`app/backends/paraformer_trt.py:128-169`).

Hypothesis:

- This is a parity issue worth measuring, but not a first patch unless we can locate the exact stats for this model. Blindly changing CMVN can regress all samples.

## Section 2: Patches (diff format, ordered by ROI)

### 1. P3: BPE `@@` merge in `decode_ids()`

ROI: highest. This should immediately fix visible English `@@` artifacts.

```diff
diff --git a/app/backends/paraformer_trt.py b/app/backends/paraformer_trt.py
--- a/app/backends/paraformer_trt.py
+++ b/app/backends/paraformer_trt.py
@@ -274,8 +274,9 @@ def decode_ids(token_ids: list[int], tokens: list[str]) -> str:
     Skips BLANK/SOS/EOS and suppresses immediate adjacent-id repeats
     (e.g. [6049, 6049] → one "好"). EOS is skipped, NOT used as a stop:
     Paraformer streaming may emit EOS mid-stream as cache-flush artifact.
+    BPE continuation suffix "@@" is stripped and merged with the next token.
     """
-    chars = []
+    pieces = []
     prev_tid: Optional[int] = None
     for tid in token_ids:
         if tid in (BLANK_ID, SOS_ID, EOS_ID):
@@ -286,9 +287,11 @@ def decode_ids(token_ids: list[int], tokens: list[str]) -> str:
             token = tokens[tid]
             if token.startswith("<") and token.endswith(">"):
                 continue
-            chars.append(token)
+            if token.endswith("@@"):
+                token = token[:-2]
+            pieces.append(token)
             prev_tid = tid
-    return "".join(chars)
+    return "".join(pieces)
```

Notes:

- This patch intentionally does not insert spaces. The current code joins all tokens without spaces, and the observed desired behavior from the task is to concatenate `["hel@@", "lo", "nice", "to", "meet", "you"]` without retaining the `@@` suffix.
- If the vocabulary uses leading-space tokens for English, this preserves those spaces because it strips only the trailing `@@`.

### 2. P2: Cross-call dedup/repetition suppression in offline `transcribe()`

ROI: medium. This makes offline `transcribe()` match the streaming path’s full-history decode behavior. It also makes P3 robust when a BPE continuation is split across decoder calls.

```diff
diff --git a/app/backends/paraformer_trt.py b/app/backends/paraformer_trt.py
--- a/app/backends/paraformer_trt.py
+++ b/app/backends/paraformer_trt.py
@@ -646,7 +646,7 @@ class ParaformerTRTBackend(ASRBackend):
         # only when audio exceeds engine max.
         ENGINE_MAX_FRAMES = 400
         chunk_frames = min(ENGINE_MAX_FRAMES, max(40, feats.shape[0]))
-        all_text_parts = []
+        all_token_ids = []
         carry_w = 0.0
         carry_e = np.zeros(512, dtype=np.float32)
         cache = [np.zeros((1, 512, 10), dtype=np.float32) for _ in range(16)]
@@ -678,10 +678,8 @@ class ParaformerTRTBackend(ASRBackend):
             )
             if sample_ids is not None:
                 new_ids = sample_ids.tolist()
-                text = decode_ids(new_ids, self._tokens)
-                if text:
-                    all_text_parts.append(text)
+                all_token_ids.extend(new_ids)
 
         # Flush tail
         if carry_w >= CIF_TAIL_THRESHOLD:
@@ -690,11 +688,10 @@ class ParaformerTRTBackend(ASRBackend):
                 dummy_enc, 1, acoustic_embeds, 1, cache,
             )
             if sample_ids is not None:
-                text = decode_ids(sample_ids.tolist(), self._tokens)
-                if text:
-                    all_text_parts.append(text)
+                all_token_ids.extend(sample_ids.tolist())
 
-        full_text = "".join(all_text_parts)
+        full_text = decode_ids(all_token_ids, self._tokens)
         return TranscriptionResult(text=full_text, language=language)
```

Notes:

- Existing adjacent-ID dedup remains in `decode_ids()` (`app/backends/paraformer_trt.py:278-290`).
- Do not add broad phrase-level suppression yet. `"智能智能"` could be an implementation duplicate, but repeated words can also be legitimate. First log `all_token_ids`, chunk boundaries, and `carry_w` on the 8 wavs.

### 3. P4: Lower tail flush threshold from 0.5 to 0.3

ROI: low to medium. This targets S3 final `"水平"` truncation.

```diff
diff --git a/app/backends/paraformer_trt.py b/app/backends/paraformer_trt.py
--- a/app/backends/paraformer_trt.py
+++ b/app/backends/paraformer_trt.py
@@ -75,7 +75,7 @@ HIGH_FREQ = 8000
 
 # CIF parameters
 CIF_THRESHOLD = 1.0
-CIF_TAIL_THRESHOLD = 0.5    # Minimum weight to fire tail token on finalize
+CIF_TAIL_THRESHOLD = 0.3    # Minimum weight to fire tail token on finalize
 
 # Tokens
 BLANK_ID = 0
```

Notes:

- This is a controlled experiment, not a guaranteed fix. Validate with silence and noisy tails because lower threshold can over-emit.
- A more conservative alternative is to make this environment-configurable:

```diff
diff --git a/app/backends/paraformer_trt.py b/app/backends/paraformer_trt.py
--- a/app/backends/paraformer_trt.py
+++ b/app/backends/paraformer_trt.py
@@ -75,7 +75,9 @@ HIGH_FREQ = 8000
 
 # CIF parameters
 CIF_THRESHOLD = 1.0
-CIF_TAIL_THRESHOLD = 0.5    # Minimum weight to fire tail token on finalize
+CIF_TAIL_THRESHOLD = float(os.environ.get(
+    "PARAFORMER_CIF_TAIL_THRESHOLD", "0.5"
+))  # Minimum weight to fire tail token on finalize
 
 # Tokens
 BLANK_ID = 0
```

Recommended experiment order: first land P3/P2, then run the 8 wavs with `0.5`, `0.4`, `0.3`. Choose the lowest threshold that does not add tail hallucinations.

### 4. P5: CMVN parity with sherpa-onnx/global stats

ROI: unknown until stats are available. Skip direct patch for now unless the model directory contains reliable `neg_mean` / `inv_stddev` or an ONNX metadata reader is added.

Current local code:

```diff
diff --git a/app/backends/paraformer_trt.py b/app/backends/paraformer_trt.py
--- a/app/backends/paraformer_trt.py
+++ b/app/backends/paraformer_trt.py
@@ -163,6 +163,7 @@ def compute_fbank(audio: np.ndarray) -> np.ndarray:
     # Utterance-level CMVN
     mean = mel_feats.mean(axis=0, keepdims=True)
     std = mel_feats.std(axis=0, keepdims=True)
     std = np.maximum(std, 1e-10)
     mel_feats = (mel_feats - mean) / std
```

Proposed direction, not a ready patch:

- Add a `cmvn_mode` argument or backend-level setting only after confirming the model’s expected normalization.
- If using sherpa-style metadata, apply `mel_feats = (mel_feats + neg_mean) * inv_stddev` or the exact equivalent from sherpa-onnx feature code. Do not guess the sign/order from metadata names alone.

## Section 3: Priority Order

1. P3 BPE `@@` detokenize: estimated 20-40 CER percentage point improvement on English-heavy samples S4/S6/S7, and near-zero risk for Chinese tokens. This is a text-layer deterministic bug.

2. P2 offline full-history decode / cross-call dedup: estimated 2-8 CER percentage point improvement depending on how many repetitions are chunk-boundary artifacts. It also strengthens P3 because BPE suffixes can span decoder calls.

3. P4 tail threshold experiment: estimated 1-5 CER percentage point improvement on utterances with final truncation. Risk is tail hallucination; keep behind config or validate threshold before hard-coding.

4. P5 CMVN parity investigation: possible 3-15 CER percentage point movement in either direction. This is potentially broad but unsafe without model stats. Treat as an experiment after P3/P2.

5. P1 CIF/acoustic omissions: estimated 0-5 CER percentage point implementation improvement unless alpha diagnostics prove under-fire. If the same Chinese omissions appear in sherpa-onnx ONNX inference, this is mostly model ceiling.

## Section 4: Non-fixable Issues

Likely implementation bugs:

- P3 is implementation-level. The local decoder loads `@@` token strings correctly but never detokenizes them (`app/backends/paraformer_trt.py:250-291`). Fix this first.
- Part of P2 is implementation-level in offline `transcribe()` because it decodes new IDs chunk-by-chunk and appends strings (`app/backends/paraformer_trt.py:679-683`) instead of mirroring streaming full-history decode (`app/backends/paraformer_trt.py:419-423`).
- Part of P4 may be implementation-level because the final CIF residual is dropped below 0.5 (`app/backends/paraformer_trt.py:685-695`). This needs threshold testing, not blind tuning.

Likely model capability ceiling or feature-parity issues:

- P1 Chinese adjacent-character omissions are not worth chasing deeply until compared against sherpa-onnx ONNX output using the same audio and model. If sherpa-onnx also says `"实别"` / `"语合成"`, the error is acoustic/language-model capability, not TRT post-processing.
- P5 CMVN affects all samples, but the correct fix depends on the training/export contract. Local per-utterance CMVN is an observed fact (`app/backends/paraformer_trt.py:163-167`); sherpa-onnx metadata global stats are an observed upstream fact; the required local transformation remains a hypothesis until verified against model metadata and reference outputs.
- Repetitions like `"greatday嗯greatday"` may be model/decoder behavior if they occur identically in sherpa-onnx. If they occur only in TRT, inspect decoder cache handling, CIF alpha sums, and offline chunk boundaries before adding text-level phrase suppression.

Recommended first fix: land P3 plus P2 together, because P3 removes the highest-severity English BPE artifact and P2 ensures the detokenizer sees the whole offline token stream. Then run the 8 wavs and record per-sample CER deltas before changing CIF thresholds or CMVN.
