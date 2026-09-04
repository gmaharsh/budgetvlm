"""Runtime patches for known vLLM 0.28 + Qwen3-VL video-pruning bugs."""
from __future__ import annotations

from .utils import get_logger

log = get_logger("vllm_patches")
_APPLIED = False


def apply_qwen3_vl_mrope_patch() -> None:
    """Clamp M-RoPE writes when local_start is past the sequence (vLLM bug).

    Upstream ``recompute_mrope_positions`` can compute local_start >= N for
    Qwen3-VL interleaved video + pruning, then assign a non-empty mm_pos into
    an empty slice and crash. Fall back to placing at the first media token.
    """
    global _APPLIED
    if _APPLIED:
        return
    try:
        import vllm.multimodal.video_prune.evs as evs
    except Exception as e:  # noqa: BLE001
        log.warning("Could not import vLLM EVS module for patch: %s", e)
        return

    import torch

    orig = evs.recompute_mrope_positions

    def _patched(
        input_ids,
        multimodal_positions,
        mrope_positions,
        num_computed_tokens,
        vision_start_token_id,
        image_token_id,
        video_token_id,
    ):
        try:
            return orig(
                input_ids,
                multimodal_positions,
                mrope_positions,
                num_computed_tokens,
                vision_start_token_id,
                image_token_id,
                video_token_id,
            )
        except RuntimeError as e:
            msg = str(e)
            if "expanded size of the tensor (0)" not in msg:
                raise
            log.warning(
                "vLLM M-RoPE recompute hit empty-slice bug; applying media-mask fallback. "
                "Original: %s",
                msg,
            )
            positions = mrope_positions.clone()
            n = int(input_ids.numel())
            image_mask = input_ids.eq(image_token_id)
            video_mask = input_ids.eq(video_token_id)
            media_mask = image_mask | video_mask
            text_mask = ~media_mask
            media_idx = media_mask.nonzero(as_tuple=True)[0]
            if len(media_idx) == 0 or not multimodal_positions:
                delta = int((positions.max().item() + 1) - n) if positions.numel() else -n
                return positions, delta

            cursor = int(media_idx[0].item())
            for mm_pos in multimodal_positions:
                if mm_pos is None or mm_pos.numel() == 0:
                    continue
                width = int(mm_pos.shape[1])
                local_start = cursor
                local_end = min(local_start + width, n)
                use = local_end - local_start
                if use <= 0:
                    continue
                base = (
                    int(positions[-1, max(local_start - 1, 0)].item()) + 1
                    if local_start > 0
                    else 0
                )
                positions[:, local_start:local_end] = mm_pos[0:3, :use] + base
                if mm_pos.shape[0] >= 5:
                    offset = int(mm_pos[0:3, :use].max().item()) + base + 1
                elif mm_pos.shape[0] >= 4:
                    offset = int(mm_pos[3, 0].item()) + base
                else:
                    offset = int(mm_pos[0:3, :use].max().item()) + base + 1
                if local_end < n:
                    text_pos_sum = torch.cumsum(text_mask[local_end:].long(), dim=0)
                    positions[:, local_end:n] = text_pos_sum + offset - 1
                cursor = local_end

            delta = int((positions.max().item() + 1) - n) if positions.numel() else -n
            return positions, delta

    evs.recompute_mrope_positions = _patched  # type: ignore[assignment]
    _APPLIED = True
    log.info("Applied Qwen3-VL M-RoPE empty-slice fallback patch")
