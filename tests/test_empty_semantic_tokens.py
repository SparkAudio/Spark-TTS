# Copyright (c) 2025 SparkAudio
# Licensed under the Apache License, Version 2.0

"""
Regression tests for empty semantic token handling (fixes #82, #36).

When the LLM generates no bicodec_semantic_* tokens, the pipeline used to
crash with:
    RuntimeError: Calculated padded input size per channel: (0).
    Kernel size: (1). Kernel size can't be greater than actual input size

This was caused by passing a zero-length tensor through Conv1d in
FactorizedVectorQuantize.detokenize().

These tests verify:
1. FactorizedVectorQuantize.detokenize() raises ValueError on empty input
2. SparkTTS.inference() retries when no semantic tokens are generated
3. SparkTTS.inference() raises RuntimeError after exhausting retries
4. The regex extraction correctly handles various LLM outputs
5. Normal (non-empty) token flow is unaffected
"""

import os
import re
import sys
import types
import unittest
from unittest.mock import MagicMock

import torch

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Stub heavy dependencies that are NOT installed in the test environment.
# We only stub leaf dependencies; the actual modules under test are real.
# ---------------------------------------------------------------------------

def _ensure_stubs():
    """Install stubs for packages not available in the test environment."""

    # omegaconf
    if "omegaconf" not in sys.modules or not hasattr(sys.modules["omegaconf"], "OmegaConf"):
        omegaconf = types.ModuleType("omegaconf")
        omegaconf.DictConfig = dict

        class _OmegaConf:
            @staticmethod
            def load(path):
                import yaml
                with open(path) as f:
                    return yaml.safe_load(f)
            @staticmethod
            def to_container(cfg, resolve=True):
                return dict(cfg) if hasattr(cfg, "items") else cfg

        omegaconf.OmegaConf = _OmegaConf
        sys.modules["omegaconf"] = omegaconf

    # tqdm
    if "tqdm" not in sys.modules:
        tqdm_mod = types.ModuleType("tqdm")
        tqdm_mod.tqdm = lambda x, **kw: x
        sys.modules["tqdm"] = tqdm_mod

    # transformers
    if "transformers" not in sys.modules:
        transformers = types.ModuleType("transformers")
        transformers.AutoTokenizer = MagicMock()
        transformers.AutoModelForCausalLM = MagicMock()
        transformers.Wav2Vec2FeatureExtractor = MagicMock()
        transformers.Wav2Vec2Model = MagicMock()
        sys.modules["transformers"] = transformers

    # safetensors
    if "safetensors" not in sys.modules:
        safetensors = types.ModuleType("safetensors")
        safetensors_torch = types.ModuleType("safetensors.torch")
        safetensors_torch.load_file = MagicMock(return_value={})
        safetensors.torch = safetensors_torch
        sys.modules["safetensors"] = safetensors
        sys.modules["safetensors.torch"] = safetensors_torch

    # torchaudio
    if "torchaudio" not in sys.modules:
        torchaudio = types.ModuleType("torchaudio")
        torchaudio_transforms = types.ModuleType("torchaudio.transforms")
        torchaudio_transforms.MelSpectrogram = MagicMock()
        torchaudio.transforms = torchaudio_transforms
        sys.modules["torchaudio"] = torchaudio
        sys.modules["torchaudio.transforms"] = torchaudio_transforms

    # soundfile
    if "soundfile" not in sys.modules:
        sf = types.ModuleType("soundfile")
        sf.write = MagicMock()
        sys.modules["soundfile"] = sf

    # soxr
    if "soxr" not in sys.modules:
        soxr = types.ModuleType("soxr")
        soxr.resample = MagicMock()
        sys.modules["soxr"] = soxr

    # gradio
    if "gradio" not in sys.modules:
        gr = types.ModuleType("gradio")
        gr.Error = type("Error", (Exception,), {})
        gr.Blocks = MagicMock()
        gr.HTML = MagicMock()
        gr.Tabs = MagicMock()
        gr.TabItem = MagicMock()
        gr.Markdown = MagicMock()
        gr.Row = MagicMock()
        gr.Column = MagicMock()
        gr.Audio = MagicMock()
        gr.Textbox = MagicMock()
        gr.Button = MagicMock()
        gr.Radio = MagicMock()
        gr.Slider = MagicMock()
        sys.modules["gradio"] = gr


_ensure_stubs()


# ---------------------------------------------------------------------------
# Now import the real modules under test
# ---------------------------------------------------------------------------

from sparktts.modules.vq.factorized_vector_quantize import FactorizedVectorQuantize
from cli.SparkTTS import SparkTTS


class TestFactorizedVQDetokenizeGuard(unittest.TestCase):
    """Tests for the defensive guard in FactorizedVectorQuantize.detokenize()."""

    def _make_quantizer(self, input_dim=256, codebook_size=1024, codebook_dim=8):
        """Create a minimal quantizer for testing."""
        q = FactorizedVectorQuantize(
            input_dim=input_dim,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            commitment=1.0,
        )
        q.eval()
        return q

    def test_empty_indices_raises_valueerror(self):
        """Empty token sequence must raise ValueError, not Conv1d RuntimeError."""
        q = self._make_quantizer()
        empty_indices = torch.tensor([], dtype=torch.long).unsqueeze(0)  # shape (1, 0)
        with self.assertRaises(ValueError) as ctx:
            q.detokenize(empty_indices)
        self.assertIn("empty token sequence", str(ctx.exception).lower())

    def test_zero_numel_indices_raises_valueerror(self):
        """Zero-numel tensor with explicit shape must also raise."""
        q = self._make_quantizer()
        empty_indices = torch.zeros(1, 0, dtype=torch.long)
        with self.assertRaises(ValueError) as ctx:
            q.detokenize(empty_indices)
        self.assertIn("semantic tokens", str(ctx.exception).lower())

    def test_normal_indices_pass_through(self):
        """Non-empty indices should not trigger the guard."""
        q = self._make_quantizer(input_dim=8, codebook_size=16, codebook_dim=8)
        try:
            torch.nn.utils.remove_weight_norm(q.in_project)
        except ValueError:
            pass
        try:
            torch.nn.utils.remove_weight_norm(q.out_project)
        except ValueError:
            pass
        indices = torch.randint(0, 16, (1, 10))
        result = q.detokenize(indices)
        self.assertEqual(result.shape[0], 1)
        self.assertEqual(result.shape[1], 8)  # input_dim
        self.assertEqual(result.shape[2], 10)  # seq_len preserved

    def test_single_token_passes(self):
        """Single token (seq_len=1) should work fine."""
        q = self._make_quantizer(input_dim=8, codebook_size=16, codebook_dim=8)
        try:
            torch.nn.utils.remove_weight_norm(q.in_project)
        except ValueError:
            pass
        try:
            torch.nn.utils.remove_weight_norm(q.out_project)
        except ValueError:
            pass
        indices = torch.randint(0, 16, (1, 1))
        result = q.detokenize(indices)
        self.assertEqual(result.shape, (1, 8, 1))


class TestSemanticTokenExtraction(unittest.TestCase):
    """Tests for the regex extraction of semantic token IDs from LLM output."""

    PATTERN = r"bicodec_semantic_(\d+)"

    def test_normal_output_extracts_tokens(self):
        """Standard LLM output with semantic tokens."""
        text = (
            "<|bicodec_semantic_42|><|bicodec_semantic_100|>"
            "<|bicodec_semantic_7|>"
        )
        matches = re.findall(self.PATTERN, text)
        self.assertEqual(matches, ["42", "100", "7"])

    def test_empty_output_extracts_nothing(self):
        """LLM output with no semantic tokens returns empty list."""
        text = "<|bicodec_global_5|><|end_semantic_token|>"
        matches = re.findall(self.PATTERN, text)
        self.assertEqual(matches, [])

    def test_only_global_tokens(self):
        """Output containing only global tokens → no semantic matches."""
        text = "<|bicodec_global_0|><|bicodec_global_1|><|bicodec_global_2|>"
        matches = re.findall(self.PATTERN, text)
        self.assertEqual(matches, [])

    def test_mixed_tokens(self):
        """Both global and semantic tokens present → only semantic extracted."""
        text = (
            "<|bicodec_global_3|><|bicodec_semantic_10|>"
            "<|bicodec_global_4|><|bicodec_semantic_20|>"
        )
        matches = re.findall(self.PATTERN, text)
        self.assertEqual(matches, ["10", "20"])

    def test_completely_empty_string(self):
        """Empty string from LLM."""
        matches = re.findall(self.PATTERN, "")
        self.assertEqual(matches, [])

    def test_garbage_output(self):
        """Malformed LLM output."""
        text = "I cannot help with that request."
        matches = re.findall(self.PATTERN, text)
        self.assertEqual(matches, [])


class TestSparkTTSInferenceRetry(unittest.TestCase):
    """Tests for the retry logic in SparkTTS.inference()."""

    def _make_mock_sparktts(self):
        """Build a mock SparkTTS instance with controllable LLM output."""
        # Create a real instance by bypassing __init__
        obj = object.__new__(SparkTTS)
        obj.device = torch.device("cpu")
        obj.model_dir = "/fake"
        obj.configs = {"sample_rate": 16000}
        obj.sample_rate = 16000

        # Mock the tokenizer
        obj.tokenizer = MagicMock()
        mock_inputs = MagicMock()
        mock_inputs.input_ids = torch.zeros(1, 10, dtype=torch.long)
        mock_inputs.to.return_value = mock_inputs
        obj.tokenizer.return_value = mock_inputs

        # Mock the LLM
        obj.model = MagicMock()
        obj.model.generate.return_value = torch.zeros(1, 20, dtype=torch.long)

        # Mock the audio tokenizer
        obj.audio_tokenizer = MagicMock()
        obj.audio_tokenizer.detokenize.return_value = torch.randn(16000).numpy()
        obj.audio_tokenizer.tokenize.return_value = (
            torch.randint(0, 100, (1, 1, 32)),  # global_token_ids
            torch.randint(0, 500, (1, 50)),      # semantic_token_ids
        )

        return obj

    def test_retry_on_empty_semantic_tokens(self):
        """Inference retries when first attempt produces no semantic tokens."""
        obj = self._make_mock_sparktts()

        call_count = [0]
        def mock_batch_decode(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 2:
                return ["<|bicodec_global_1|>"]  # no semantic tokens
            return ["<|bicodec_semantic_42|><|bicodec_semantic_100|>"]

        obj.tokenizer.batch_decode = mock_batch_decode

        # Should succeed on 3rd attempt
        result = obj.inference(
            text="Hello world",
            prompt_speech_path="/fake/path.wav",
            max_retries=5,
        )

        self.assertEqual(call_count[0], 3)
        self.assertIsNotNone(result)

    def test_raises_after_max_retries(self):
        """Inference raises RuntimeError after max_retries with no semantic tokens."""
        obj = self._make_mock_sparktts()

        # All calls return no semantic tokens
        obj.tokenizer.batch_decode = MagicMock(
            return_value=["<|bicodec_global_1|>"]
        )

        with self.assertRaises(RuntimeError) as ctx:
            obj.inference(
                text="Hi",
                prompt_speech_path="/fake/path.wav",
                max_retries=2,
            )

        self.assertIn("semantic tokens", str(ctx.exception))
        self.assertIn("2 attempts", str(ctx.exception))
        self.assertEqual(obj.tokenizer.batch_decode.call_count, 2)

    def test_succeeds_on_first_try(self):
        """No retry needed when first attempt has semantic tokens."""
        obj = self._make_mock_sparktts()

        obj.tokenizer.batch_decode = MagicMock(
            return_value=["<|bicodec_semantic_1|><|bicodec_semantic_2|>"]
        )

        result = obj.inference(
            text="Hello world",
            prompt_speech_path="/fake/path.wav",
        )

        obj.tokenizer.batch_decode.assert_called_once()
        self.assertIsNotNone(result)

    def test_controlled_generation_no_retry_needed(self):
        """Controlled generation (gender mode) with semantic tokens succeeds."""
        obj = self._make_mock_sparktts()

        obj.tokenizer.batch_decode = MagicMock(
            return_value=[
                "<|bicodec_global_0|><|bicodec_global_1|>"
                "<|bicodec_semantic_10|><|bicodec_semantic_20|>"
            ]
        )

        result = obj.inference(
            text="Hello world",
            gender="male",
            pitch="moderate",
            speed="moderate",
        )

        obj.tokenizer.batch_decode.assert_called_once()
        self.assertIsNotNone(result)

    def test_max_retries_default_is_positive(self):
        """Default max_retries should be a positive integer."""
        import inspect
        sig = inspect.signature(SparkTTS.inference)
        default = sig.parameters["max_retries"].default
        self.assertIsInstance(default, int)
        self.assertGreater(default, 0)


class TestSourceCodeAudit(unittest.TestCase):
    """Source-level checks to prevent regressions."""

    def test_inference_has_max_retries_param(self):
        """SparkTTS.inference must accept max_retries parameter."""
        import inspect
        sig = inspect.signature(SparkTTS.inference)
        self.assertIn("max_retries", sig.parameters)

    def test_detokenize_has_guard(self):
        """FactorizedVectorQuantize.detokenize must check for empty input."""
        import inspect
        source = inspect.getsource(FactorizedVectorQuantize.detokenize)
        self.assertTrue(
            "numel" in source or "len(" in source or "size(" in source,
            "detokenize() must check for empty input before Conv1d"
        )

    def test_webui_catches_runtime_error(self):
        """webui.py must catch RuntimeError in voice_clone and voice_creation."""
        webui_path = os.path.join(PROJECT_ROOT, "webui.py")
        with open(webui_path, "r") as f:
            source = f.read()
        self.assertIn("except RuntimeError", source)
        self.assertIn("gr.Error", source)

    def test_guard_before_conv1d(self):
        """Verify detokenize validates before calling out_project."""
        import inspect
        source = inspect.getsource(FactorizedVectorQuantize.detokenize)
        lines = source.strip().split("\n")
        guard_line = None
        project_line = None
        for i, line in enumerate(lines):
            if "numel" in line or "ValueError" in line:
                guard_line = i
            if "out_project" in line:
                project_line = i
        self.assertIsNotNone(guard_line, "Guard not found in detokenize")
        self.assertIsNotNone(project_line, "out_project call not found")
        self.assertLess(guard_line, project_line,
                        "Guard must come before out_project call")


if __name__ == "__main__":
    unittest.main()
