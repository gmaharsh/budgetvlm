"""Inference backends: mock (local), vLLM (CUDA), transformers (optional)."""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .prompts import build_mcq_prompt, extract_letter
from .pruning import retained_frames
from .utils import get_logger
from .video_io import load_frames, probe_video

log = get_logger("backend")


@dataclass
class InferResult:
    prediction: str
    raw_text: str
    latency_sec: float
    ttft_sec: float
    n_frames: int
    visual_tokens: int | None
    pruning_rate: float
    video_duration_sec: float
    meta: dict[str, Any] = field(default_factory=dict)


class Backend(ABC):
    @abstractmethod
    def infer_mcq(
        self,
        video_path: str,
        question: str,
        options: list[str],
        pruning_rate: float,
        baseline_num_frames: int,
        max_pixels: int | None = None,
        ground_truth: str | None = None,
    ) -> InferResult:
        ...


class MockBackend(Backend):
    """Deterministic stand-in so Phases 3–12 can be tested without CUDA.

    Uses measured complexity to decide a max-safe pruning rate. If
    ``ground_truth`` is provided, returns it when pruning is safe, else a
    wrong letter — so accuracy vs pruning behaves like the hypothesis.
    """

    def __init__(self, seed: int = 42, t1: float = 0.15, t2: float = 0.40):
        self.rng = np.random.default_rng(seed)
        self.t1 = t1
        self.t2 = t2

    def infer_mcq(
        self,
        video_path: str,
        question: str,
        options: list[str],
        pruning_rate: float,
        baseline_num_frames: int,
        max_pixels: int | None = None,
        ground_truth: str | None = None,
    ) -> InferResult:
        from .complexity import frame_difference_complexity

        t0 = time.perf_counter()
        n_keep = retained_frames(baseline_num_frames, pruning_rate)
        frames, meta = load_frames(video_path, num_frames=max(8, n_keep), resize_max=128)
        probe = frames[:: max(1, len(frames) // 8)][:8]
        cx = frame_difference_complexity(probe)

        # High complexity → less pruning tolerance
        if cx < self.t1:
            safe_rate = 0.75
        elif cx < self.t2:
            safe_rate = 0.50
        else:
            safe_rate = 0.25

        true_letter = (ground_truth or "A").strip().upper()[:1]
        if true_letter not in "ABCD":
            true_letter = "A"

        keep_correct = pruning_rate <= safe_rate + 1e-9
        if keep_correct:
            pred = true_letter
        else:
            wrong = [L for L in "ABCD" if L != true_letter]
            pred = wrong[int(self.rng.integers(0, len(wrong)))]

        visual_tokens = n_keep * 64
        ttft = 0.05 + 0.002 * visual_tokens * (1.0 + 0.5 * cx)
        latency = (time.perf_counter() - t0) + ttft

        return InferResult(
            prediction=pred,
            raw_text=pred,
            latency_sec=latency,
            ttft_sec=ttft,
            n_frames=n_keep,
            visual_tokens=visual_tokens,
            pruning_rate=pruning_rate,
            video_duration_sec=meta.duration_sec,
            meta={"complexity_probe": cx, "safe_rate": safe_rate, "backend": "mock"},
        )


class VLLMBackend(Backend):
    """Qwen3-VL via vLLM offline inference."""

    def __init__(self, model_name: str, cfg: dict[str, Any] | None = None):
        cfg = cfg or {}
        self.model_name = model_name
        self.cfg = cfg
        self._llm = None
        self._processor = None

    def _lazy_init(self) -> None:
        if self._llm is not None:
            return
        from vllm import LLM, SamplingParams  # noqa: F401
        from transformers import AutoProcessor

        log.info("Loading vLLM model %s ...", self.model_name)
        self._llm = LLM(
            model=self.model_name,
            trust_remote_code=True,
            gpu_memory_utilization=float(self.cfg.get("gpu_memory_utilization", 0.9)),
            max_model_len=int(self.cfg.get("max_model_len", 32768)),
            limit_mm_per_prompt={"video": 1},
        )
        self._processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
        self._SamplingParams = SamplingParams

    def infer_mcq(
        self,
        video_path: str,
        question: str,
        options: list[str],
        pruning_rate: float,
        baseline_num_frames: int,
        max_pixels: int | None = None,
        ground_truth: str | None = None,
    ) -> InferResult:
        self._lazy_init()
        from qwen_vl_utils import process_vision_info

        n_keep = retained_frames(baseline_num_frames, pruning_rate)
        meta = probe_video(video_path)
        prompt_text = build_mcq_prompt(question, options)

        video_cfg: dict[str, Any] = {
            "type": "video",
            "video": video_path,
            "nframes": n_keep,
        }
        if max_pixels is not None:
            video_cfg["max_pixels"] = max_pixels

        messages = [
            {
                "role": "user",
                "content": [
                    video_cfg,
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            image_patch_size=self._processor.image_processor.patch_size,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        # Critical for Video-MME accuracy under vLLM (avoid double-resize)
        if self.cfg.get("do_resize") is False:
            video_kwargs = dict(video_kwargs or {})
            video_kwargs["do_resize"] = False

        mm_data: dict[str, Any] = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs

        visual_tokens = None
        try:
            # rough count from grid if present
            if hasattr(video_inputs, "shape"):
                visual_tokens = int(np.prod(video_inputs.shape) // 3)  # fallback
            thw = video_kwargs.get("video_grid_thw") if isinstance(video_kwargs, dict) else None
            if thw is not None:
                visual_tokens = int(np.array(thw).prod())
        except Exception:
            visual_tokens = n_keep * 64

        sp = self._SamplingParams(
            temperature=float(self.cfg.get("temperature", 0.0)),
            max_tokens=int(self.cfg.get("max_tokens", 16)),
        )

        t0 = time.perf_counter()
        ttft_box = {"t": None}

        def _on_first_tok():
            if ttft_box["t"] is None:
                ttft_box["t"] = time.perf_counter()

        outputs = self._llm.generate(
            [
                {
                    "prompt": text,
                    "multi_modal_data": mm_data,
                    "mm_processor_kwargs": video_kwargs or {},
                }
            ],
            sampling_params=sp,
        )
        latency = time.perf_counter() - t0
        raw = outputs[0].outputs[0].text if outputs else ""
        # vLLM offline doesn't expose TTFT easily; approximate with latency for short gens
        ttft = ttft_box["t"] - t0 if ttft_box["t"] else latency

        return InferResult(
            prediction=extract_letter(raw) or raw.strip()[:1].upper(),
            raw_text=raw,
            latency_sec=latency,
            ttft_sec=ttft,
            n_frames=n_keep,
            visual_tokens=visual_tokens,
            pruning_rate=pruning_rate,
            video_duration_sec=meta.duration_sec,
            meta={"backend": "vllm", "model": self.model_name},
        )


def get_backend(name: str, cfg: dict[str, Any]) -> Backend:
    name = (name or "mock").lower()
    if name == "mock":
        cx = cfg.get("complexity", {})
        return MockBackend(
            seed=int(cfg.get("seed", 42)),
            t1=float(cx.get("t1", 0.15)),
            t2=float(cx.get("t2", 0.40)),
        )
    if name == "vllm":
        return VLLMBackend(cfg["model"]["name"], {**cfg.get("serving", {}), **cfg.get("model", {})})
    raise ValueError(f"Unknown backend: {name}")
