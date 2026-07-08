# AGENTS.md

Notes for AI coding agents (and humans) running Spark-TTS inference, verified end-to-end on a
real **Tesla T4 (16GB, Turing, sm_75)**. Nothing here changes any default behavior — it only
documents things the README doesn't say, discovered by actually running the code.

## Correct invocation (README's own example is right, but easy to get wrong)

Run inference as a **module**, not as a script path:

```bash
python -m cli.inference \
    --text "text to synthesis." \
    --device 0 \
    --save_dir "path/to/save/audio" \
    --model_dir pretrained_models/Spark-TTS-0.5B \
    --prompt_text "transcript of the prompt audio" \
    --prompt_speech_path "path/to/prompt_audio"
```

Running `python cli/inference.py ...` instead (a natural first guess, since that's the file's
path) fails immediately with:

```
ModuleNotFoundError: No module named 'cli'
```

because `cli/inference.py` does `from cli.SparkTTS import SparkTTS`, a package-relative import
that only resolves when the process is launched with `-m` from the repo root.

## Verified working environment (Tesla T4)

- Python **3.10.12** worked fine end-to-end (README's badge says "Python 3.12+"; 3.12 was not
  available on the test box and was not required for anything we ran).
- `pip install -r requirements.txt` (verbatim) resolves `torch==2.5.1+cu124`,
  `torchaudio==2.5.1+cu124`, `transformers==4.46.2` with no conflicts, and
  `torch.cuda.is_available()` / `torch.cuda.get_device_name(0)` correctly report `Tesla T4`.
- Model download (`snapshot_download("SparkAudio/Spark-TTS-0.5B", ...)`) is ~3.7GB and took
  ~77s on this box.
- **Peak VRAM for one inference call: ~4.1-4.3GB** (measured with
  `nvidia-smi --query-gpu=memory.used --format=csv -l 1` sampled in the background, and cross-checked
  with `torch.cuda.max_memory_allocated()`). Comfortably fits a 16GB T4 with ~12GB to spare.

## dtype: do NOT naively force fp16 on Turing/T4 — it breaks generation

`cli/SparkTTS.py`'s `AutoModelForCausalLM.from_pretrained(f"{self.model_dir}/LLM")` call does
not pass `torch_dtype`, so the LLM loads in **fp32** by default — even though
`pretrained_models/Spark-TTS-0.5B/LLM/config.json` declares `"torch_dtype": "bfloat16"` (the
dtype the checkpoint was trained/saved in).

On a T4 it's tempting to "fix" this by forcing fp16 (Turing has no bf16 Tensor Core path, so
fp16 is normally the right general-purpose choice on this hardware generation).
**We tried it and it breaks:**

```python
AutoModelForCausalLM.from_pretrained(f"{model_dir}/LLM", torch_dtype=torch.float16)
# ... later, during model.generate():
RuntimeError: CUDA error: device-side assert triggered
# (root cause, before the async assert fires):
# probability tensor contains either `inf`, `nan` or element < 0
```

Reproduced twice (deterministic failure, not sampling-related). This happens because the
LLM (Qwen2.5-0.5B) was trained/saved in bf16, and casting straight to fp16 overflows the
logits' dynamic range during the sampling step (`temperature`/`top_k`/`top_p` in
`SparkTTS.inference()`), not because of any bug in Spark-TTS's own code.

Forcing `torch_dtype=torch.bfloat16` instead **does** run correctly (matches the checkpoint's
native dtype, numerically safe), but on this T4 it was empirically **slower than the current
fp32 default**, consistent with Turing having no bf16 Tensor Core acceleration:

| dtype | generation time (2 runs) | peak allocated |
|---|---|---|
| fp32 (current default, unspecified `torch_dtype`) | 7.59s / 8.35s | ~4.13-4.16 GB |
| bf16 (forced) | 9.21s / 9.33s | ~3.13 GB (smaller, but slower) |
| fp16 (forced) | **crashes** (NaN/Inf in sampling) | n/a |

**Takeaway for agents: on a T4, leave this repo's dtype exactly as-is (implicit fp32). It is
already the fastest of the three options we could get to actually run, and forcing fp16 will
break generation.** If VRAM headroom (not speed) is the binding constraint, bf16 is a safe
fallback, but expect it to be slower, not faster, on this hardware generation.

## Attention backend

The main LLM (`Qwen2ForCausalLM` via `transformers`) auto-selects `sdpa`
(confirmed via `model.config._attn_implementation == "sdpa"` after loading) — no
`attn_implementation` is passed explicitly anywhere in `cli/` or `sparktts/`, and no
`sdpa_kernel()`/`torch.backends.cuda.sdp_kernel()` forcing exists on this path, so PyTorch's
normal SDPA backend fallback applies (fine on T4 — flash-attention-2-style kernels aren't
expected to be selected on sm_75 anyway).

Separately, `sparktts/modules/speaker/perceiver_encoder.py` has an `Attend` class with explicit
A100-vs-other-GPU detection (`device_properties.major == 8 and minor == 0` → flash config,
else → math/mem-efficient config) that would be a nice T4-aware pattern — but it's dead code on
the default inference path: `PerceiverResampler(..., use_flash_attn=False)` is the only call
site (`sparktts/modules/speaker/speaker_encoder.py`), and `Attend.__init__` returns before the
GPU-detection logic ever runs when `use_flash=False`. Not a bottleneck (this submodule is a
small speaker-embedding component, not the autoregressive LLM), just noting it in case anyone
wonders why the "Non-A100 GPU detected" log line never appears.

## Quantization / CUDA Graphs — not applicable to the PyTorch inference path

- `bitsandbytes`/`int8` appear only in `runtime/triton_trtllm/scripts/convert_checkpoint.py`
  (TensorRT-LLM engine export for the optional Triton serving backend) — not in
  `requirements.txt` and not reachable from `python -m cli.inference`. No int8 quantization
  exists on the plain PyTorch CLI path this document covers.
- No `torch.cuda.graph` / `CUDAGraph` / `make_graphed_callables` usage anywhere in the repo
  (verified via `grep -rn` over the full tree — zero matches).

## Reproducibility note

`cli/inference.py` exposes no `--seed` argument and no `torch.manual_seed()` call, and
`SparkTTS.inference()` always calls `model.generate(..., do_sample=True, ...)`. Two runs of the
exact same documented command produce **different output lengths** (5.32s vs 5.86s in our
test) — this is expected sampling variance, not a bug, but agents scripting reproducible
benchmarks should be aware there is no built-in way to pin the seed from the CLI.
