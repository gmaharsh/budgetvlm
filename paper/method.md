# Method

## Overview

BudgetVLM treats visual-token budget as a content-dependent control knob for video–language model (VLM) inference. Given a video and a multiple-choice question, a lightweight complexity score selects a pruning rate from a small discrete set. Inference then runs under that rate using engine-level video token pruning, keeping the sampled frame count and spatial resolution fixed so that cost reduction comes from fewer retained visual tokens—not from dropping frames.

This paper evaluates a **cached adaptive policy** (Stage A): we run a fixed matrix of pruning rates once, then simulate BudgetVLM and an oracle by indexing into cached outcomes. A true per-request multi-engine router is left as future systems work (Stage B).

## Video token pruning (fixed frames)

Let $V$ be an input video. We sample a fixed number of frames $N$ at a fixed maximum pixel budget $P$ for every pruning condition. The vision encoder therefore sees the same frame tensor across rates. After encoding, the serving stack applies video visual-token pruning at rate $r \in \{0, 0.25, 0.50, 0.75\}$:

$$
\text{frames}(N,P) \xrightarrow{\text{vision encoder}} \text{visual tokens}
\xrightarrow{\text{prune @ } r} \text{retained tokens}
\xrightarrow{\text{LLM prefill+decode}} \text{answer}.
$$

We use **VidCom2** as the primary pruning method via vLLM’s `video_pruning_rate` / `video_pruning_method` engine configuration. Optional **EVS** is reserved as an ablation when the installed vLLM build supports it. Importantly, $r$ is an **engine-level** setting in current vLLM: one engine instance is bound to one rate. Our Stage A protocol therefore launches (or sequentially reconstructs) one engine per rate, caches per-(video, question, rate) outcomes, and never claims online rate switching within a single engine.

We do **not** reduce $N$ when $r$ increases. Frame dropping would answer a different research question (frame budgeting). Our comparisons isolate token pruning under matched temporal and spatial inputs.

## Complexity estimator

Before VLM inference, BudgetVLM scores each video with a cheap, model-free estimator. We uniformly sample $N_c$ frames (default $N_c{=}8$), convert them to normalized grayscale, and compute the mean absolute consecutive-frame difference (motion score). An optional scene-change term counts the fraction of consecutive pairs whose difference exceeds a threshold; when enabled, complexity is a convex combination of motion and scene scores with weights $\alpha$ and $\beta$ (defaults $0.7$ / $0.3$). In the primary configuration we use the motion score alone so the policy depends on a single interpretable temporal-variation signal.

The estimator never requires a forward pass through the VLM and is intended to be negligible relative to generation cost.

## BudgetVLM policy

Let $c(V)$ denote the complexity score. With frozen thresholds $T_1 < T_2$, BudgetVLM maps complexity to a pruning rate:

$$
r(V) =
\begin{cases}
0.75 & \text{if } c(V) < T_1, \\
0.50 & \text{if } T_1 \le c(V) < T_2, \\
0.25 & \text{if } c(V) \ge T_2.
\end{cases}
$$

Low-complexity videos receive aggressive pruning; high-complexity videos retain more visual tokens. The policy always selects a positive pruning rate from $\{0.25, 0.50, 0.75\}$ (it does not choose the $0\%$ baseline). Thresholds are **not** tuned on the final test videos used for reporting.

## Threshold calibration

Given a video set with cached predictions at all rates and complexity scores, we split videos into a calibration subset (fraction $f$, default $0.3$) and a held-out test subset using a fixed seed. On the calibration subset only, we grid-search $(T_1, T_2)$ over a discrete grid subject to $T_1 < T_2$, selecting the pair that maximizes calibration accuracy and, as a tie-break, average pruning rate (favoring more savings) and lower average end-to-end latency. The chosen thresholds are then frozen and applied to the test subset for BudgetVLM evaluation.

## Oracle and baselines

**Fixed-rate baselines.** For each $r \in \{0, 0.25, 0.50, 0.75\}$, we evaluate all questions under that constant rate.

**Oracle (upper bound on adaptive pruning under our rate set).** For each video we record correctness at every rate (a video is correct at rate $r$ if all of its questions are correct at $r$). We report two oracle definitions:

- *Strict safe rate*: the largest $r$ such that the video remains correct at all rates $\le r$ (monotonic safety assumption).
- *Max observed correct rate*: the largest $r$ at which the video is correct, allowing non-monotonic failures.

Primary budgeting analysis uses the strict safe rate. Oracle adaptive evaluation indexes the cached prediction at that per-video rate.

**BudgetVLM.** Indexes the cached prediction at $r(V)$ from the calibrated policy.

## Metrics

- **Accuracy**: multiple-choice exact match on Video-MME questions (no subtitles in the primary setting).
- **End-to-end latency**: wall-clock time for one generate call (primary systems metric).
- **Visual tokens (pre)**: estimated from the processor video grid metadata before pruning.
- **Visual tokens (retained, estimated)**: $\lfloor t_{\mathrm{pre}} \cdot (1-r) \rfloor$ unless the engine exposes exact retained counts.

We do **not** report time-to-first-token (TTFT) until streaming serve instrumentation is in place; TTFT derived from non-streaming batch generate would be misleading.

## Limitations of Stage A (methodological)

Stage A measures what accuracy–cost tradeoffs are *available* under VidCom2 at discrete rates and how well a complexity heuristic can select among them. It does not yet measure the overhead of maintaining multiple engines or true online rate switching. Retained token counts are estimates unless instrumented in the engine. Complexity is a simple temporal proxy and may miss semantic difficulty that is not reflected in frame differences.
