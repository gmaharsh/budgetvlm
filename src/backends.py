"""Inference backends: mock (local), vLLM with engine-level video token pruning."""
from __future__ import annotations

import gc
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .prompts import build_mcq_prompt, extract_letter
from .pruning import retained_token_estimate
from .utils import get_logger, project_path
from .video_io import load_frames, probe_video

log = get_logger("backend")


@dataclass
class InferResult:
    prediction: str
    raw_text: str
    latency_sec: float  # end-to-end generation latency (primary)
    ttft_sec: float | None  # None until streaming instrumentation exists
    n_frames: int
    visual_tokens_pre: int | None
    visual_tokens_retained_est: int | None
    pruning_rate: float
    video_duration_sec: float
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def visual_tokens(self) -> int | None:
        """Backward-compatible alias → estimated retained tokens."""
        return self.visual_tokens_retained_est


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

    def close(self) -> None:
        return None


class MockBackend(Backend):
    """Offline stand-in: fixed frames, cost scales with retained-token estimate."""

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
        # Fixed frame budget (matches real vLLM experiment)
        frames, meta = load_frames(video_path, num_frames=baseline_num_frames, resize_max=128)
        probe = frames[:: max(1, len(frames) // 8)][:8]
        cx = frame_difference_complexity(probe)

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

        visual_pre = baseline_num_frames * 64
        visual_ret = retained_token_estimate(visual_pre, pruning_rate)
        # Simulate LLM-prefill cost scaling with retained tokens (not vision encode)
        latency = (time.perf_counter() - t0) + (0.05 + 0.002 * visual_ret * (1.0 + 0.5 * cx))

        return InferResult(
            prediction=pred,
            raw_text=pred,
            latency_sec=latency,
            ttft_sec=None,
            n_frames=baseline_num_frames,
            visual_tokens_pre=visual_pre,
            visual_tokens_retained_est=visual_ret,
            pruning_rate=pruning_rate,
            video_duration_sec=meta.duration_sec,
            meta={
                "complexity_probe": cx,
                "safe_rate": safe_rate,
                "backend": "mock",
                "pruning_kind": "simulated_token_prune",
            },
        )


class VLLMBackend(Backend):
    """Qwen3-VL via vLLM with engine-level ``video_pruning_rate`` (VidCom2).

    Important: ``video_pruning_rate`` is an *engine* setting. This backend is
    constructed for a single fixed rate. For a full matrix, create one backend
    per rate (see ``run_fixed_pruning``) to avoid OOM from multiple engines.
    """

    def __init__(
        self,
        model_name: str,
        cfg: dict[str, Any] | None = None,
        *,
        pruning_rate: float = 0.0,
    ):
        cfg = cfg or {}
        self.model_name = model_name
        self.cfg = cfg
        self.pruning_rate = float(pruning_rate)
        # Prefer VidCom2; fall back to EVS if this vLLM build lacks vidcom2.
        self.pruning_method = str(cfg.get("video_pruning_method", "vidcom2"))
        hint = project_path("results", "metrics", "pruning_method_hint.txt")
        if hint.exists():
            hinted = hint.read_text(encoding="utf-8").strip()
            if hinted in {"evs", "vidcom2"}:
                self.pruning_method = hinted
                log.info("Using pruning method from install hint: %s", hinted)
        self._llm = None
        self._processor = None
        self._SamplingParams = None

    def _lazy_init(self) -> None:
        if self._llm is not None:
            return
        from vllm import LLM, SamplingParams
        from transformers import AutoProcessor

        rate = self.pruning_rate
        # vLLM: pruning enabled when rate > 0
        vpr = None if rate <= 0.0 else rate
        log.info(
            "Loading vLLM %s | frames=FIXED | video_pruning_rate=%s | method=%s",
            self.model_name,
            vpr,
            self.pruning_method if vpr else "n/a",
        )
        kwargs: dict[str, Any] = dict(
            model=self.model_name,
            trust_remote_code=True,
            gpu_memory_utilization=float(self.cfg.get("gpu_memory_utilization", 0.9)),
            max_model_len=int(self.cfg.get("max_model_len", 32768)),
            limit_mm_per_prompt={"video": 1},
            video_pruning_rate=vpr,
        )
        if vpr is not None:
            kwargs["video_pruning_method"] = self.pruning_method

        self._llm = LLM(**kwargs)
        self._processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
        self._SamplingParams = SamplingParams

    def close(self) -> None:
        self._llm = None
        self._processor = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

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
        if abs(float(pruning_rate) - self.pruning_rate) > 1e-9:
            raise ValueError(
                f"VLLMBackend was built for pruning_rate={self.pruning_rate}, "
                f"got request rate={pruning_rate}. Use one engine per rate."
            )
        self._lazy_init()
        from qwen_vl_utils import process_vision_info

        meta = probe_video(video_path)
        prompt_text = build_mcq_prompt(question, options)

        # FIXED frame count — pruning is token-level inside vLLM
        video_cfg: dict[str, Any] = {
            "type": "video",
            "video": video_path,
            "nframes": baseline_num_frames,
        }
        if max_pixels is not None:
            video_cfg["max_pixels"] = max_pixels

        messages = [
            {
                "role": "user",
                "content": [video_cfg, {"type": "text", "text": prompt_text}],
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
        if self.cfg.get("do_resize") is False:
            video_kwargs = dict(video_kwargs or {})
            video_kwargs["do_resize"] = False

        mm_data: dict[str, Any] = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs

        visual_pre = _estimate_pre_prune_tokens(video_inputs, video_kwargs, baseline_num_frames)
        visual_ret = retained_token_estimate(visual_pre or 0, self.pruning_rate) if visual_pre else None

        sp = self._SamplingParams(
            temperature=float(self.cfg.get("temperature", 0.0)),
            max_tokens=int(self.cfg.get("max_tokens", 16)),
        )

        t0 = time.perf_counter()
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

        return InferResult(
            prediction=extract_letter(raw) or raw.strip()[:1].upper(),
            raw_text=raw,
            latency_sec=latency,
            ttft_sec=None,  # do not fake TTFT
            n_frames=baseline_num_frames,
            visual_tokens_pre=visual_pre,
            visual_tokens_retained_est=visual_ret,
            pruning_rate=self.pruning_rate,
            video_duration_sec=meta.duration_sec,
            meta={
                "backend": "vllm",
                "model": self.model_name,
                "pruning_kind": "vllm_video_token_prune",
                "video_pruning_method": self.pruning_method if self.pruning_rate > 0 else "none",
                "tokens_retained_are_estimate": True,
            },
        )


def _estimate_pre_prune_tokens(video_inputs: Any, video_kwargs: Any, n_frames: int) -> int | None:
    """Best-effort pre-pruning visual token count from processor metadata."""
    try:
        if isinstance(video_kwargs, dict):
            thw = video_kwargs.get("video_grid_thw")
            if thw is not None:
                arr = np.array(thw)
                # grid_thw often [T, H, W] patch grid; tokens ≈ prod / merge^2.
                # Without merge size here, report prod as upper-bound patch cells.
                return int(arr.reshape(-1)[-3:].prod()) if arr.size >= 3 else int(arr.prod())
        if hasattr(video_inputs, "shape"):
            # last-resort: not trustworthy — still record something labeled estimate
            return int(np.prod(video_inputs.shape) // max(1, 3))
    except Exception:
        pass
    return n_frames * 64


def get_backend(
    name: str,
    cfg: dict[str, Any],
    *,
    pruning_rate: float | None = None,
) -> Backend:
    name = (name or "mock").lower()
    if name == "mock":
        cx = cfg.get("complexity", {})
        return MockBackend(
            seed=int(cfg.get("seed", 42)),
            t1=float(cx.get("t1", 0.15)),
            t2=float(cx.get("t2", 0.40)),
        )
    if name == "vllm":
        rate = 0.0 if pruning_rate is None else float(pruning_rate)
        return VLLMBackend(
            cfg["model"]["name"],
            {**cfg.get("serving", {}), **cfg.get("model", {}), **cfg.get("pruning", {})},
            pruning_rate=rate,
        )
    raise ValueError(f"Unknown backend: {name}")
