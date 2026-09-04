# Experimental Setup

## Model and Serving

We evaluate **Qwen3-VL-4B-Instruct** using vLLM as the inference engine. Video visual-token pruning is controlled through vLLM's engine-level `video_pruning_rate` configuration, with **VidCom2** used as the primary pruning method through `video_pruning_method="vidcom2"`.

Because the pruning rate is configured at engine initialization, we instantiate one vLLM engine per fixed pruning rate. Engines are evaluated sequentially so that only one model instance occupies GPU memory at a time.

VidCom2 is required for the primary experiments. If an experimental environment does not support VidCom2, we do not substitute another pruning algorithm into the primary results. **EVS**, when supported, is evaluated only as a separately identified pruning-method ablation.

Generation uses deterministic decoding with

$$
\text{temperature}=0,
$$

and a maximum generation length of 16 tokens, which is sufficient for the multiple-choice answer format used in our evaluation.

Video preprocessing is held constant across pruning conditions. Where applicable, host-side resizing is disabled through the processor configuration so that the same preprocessing policy is used for all tested pruning rates.

We explicitly set

$$
\texttt{cap\_pixels\_per\_frame}=\texttt{True}
$$

for every pruning condition so that Qwen3-VL uses the reference per-frame pixel cap rather than a version-dependent default.

## Dataset

We evaluate on **Video-MME**, a video understanding benchmark containing multiple-choice questions associated with videos of varying duration and content.

The primary experiments are performed **without subtitles** so that model performance depends on visual-video information and the question prompt rather than auxiliary subtitle text.

We use a progressive evaluation schedule during development:

| Stage          |      Approx. videos | Purpose                                                            |
| -------------- | ------------------: | ------------------------------------------------------------------ |
| Sanity         |                 1–5 | Validate video loading, model inference, and pruning configuration |
| Pilot          |                 ~20 | Validate the complete fixed-rate and adaptive-policy pipeline      |
| Main           | **[TODO: final N]** | Produce reported paper results                                     |
| Full benchmark |                 900 | Optional evaluation if compute permits                             |

The sanity and pilot experiments are used only for pipeline validation and exploratory analysis. Final reported results use the frozen experimental configuration and the main evaluation set.

Video IDs and calibration/test split membership are saved as experiment artifacts to enable exact reproduction of the reported split.

## Matched Video Input Across Pruning Rates

To isolate the effect of visual-token pruning, all pruning conditions use identical video-sampling and spatial preprocessing settings.

| Parameter                  |            Primary setting |
| -------------------------- | -------------------------: |
| Sampled frames \(N\)       |                         32 |
| Maximum pixel budget \(P\) |                    360,360 |
| Pruning rates              | \(0,\;0.25,\;0.50,\;0.75\) |
| Pruning method             |                    VidCom2 |
| Pruning mechanism          |   vLLM video-token pruning |
| Frame dropping             |                       None |

For a given video, the same 32 sampled frames and spatial preprocessing budget are used at every pruning rate.

Thus, the comparison changes only the engine-level visual-token pruning configuration:

$$
r \in
\{0,\;0.25,\;0.50,\;0.75\}.
$$

This prevents reductions in frame count from being confounded with visual-token pruning.

## Complexity Estimator and Policy Configuration

BudgetVLM estimates video complexity independently of the VLM using a small uniformly sampled set of frames.

| Parameter                 | Setting                                               |
| ------------------------- | ----------------------------------------------------- |
| Complexity frames \(N_c\) | 8                                                     |
| Primary complexity signal | Mean consecutive-frame difference                     |
| Scene-change term         | Disabled in primary configuration                     |
| Motion/scene weights      | \(\alpha=0.7,\;\beta=0.3\) when scene term is enabled |
| Scene-change threshold    | 0.35 when enabled                                     |
| Calibration fraction      | 0.30                                                  |
| Random seed               | 42                                                    |
| Budget actions            | \(0.25,\;0.50,\;0.75\)                                |

The primary BudgetVLM configuration uses the motion-based complexity score alone.

The thresholds \(T_1\) and \(T_2\) are selected using only the calibration subset and are frozen before evaluation on the held-out test subset.

Candidate thresholds are drawn from a fixed discrete grid

$$
T_1 \in \{0.05, 0.10, 0.15, 0.20, 0.25\},
\qquad
T_2 \in \{0.30, 0.35, 0.40, 0.45, 0.50, 0.55\},
$$

restricted to pairs with \(T_1 < T_2\). Candidate pairs are evaluated using the calibration objective described in the Method section.

Any numerical threshold values stored in development configuration files before calibration are treated only as smoke-test defaults and are not used as final paper hyperparameters.

## Calibration and Test Split

Videos are randomly divided at the **video level**, using a fixed seed, into:

$$
30\% \text{ calibration}
$$

and

$$
70\% \text{ held-out test}.
$$

All questions belonging to the same video remain in the same split.

This prevents information from different questions about the same video from appearing in both calibration and evaluation data.

The calibration subset is used exclusively to select \(T_1\) and \(T_2\). Once selected, the thresholds are frozen.

**All primary BudgetVLM results are reported on the held-out test subset only.**

The fixed-rate baselines and oracle results used in headline comparisons are restricted to the same test videos so that every policy is evaluated on an identical set of examples.

## Hardware and Software

All final systems measurements are collected on the same GPU host.

| Component     | Final configuration            |
| ------------- | ------------------------------ |
| GPU           | **[TODO: record exact GPU]**   |
| GPU memory    | **[TODO]**                     |
| CUDA runtime  | **[TODO]**                     |
| NVIDIA driver | **[TODO]**                     |
| PyTorch       | **[TODO exact version]**       |
| vLLM          | **[TODO exact version/build]** |
| Transformers  | **[TODO exact version]**       |
| qwen-vl-utils | **[TODO exact version]**       |
| Python        | **[TODO exact version]**       |
| Host          | **[TODO: RunPod / other]**     |

The exact versions used to generate the final experiment artifacts are recorded after the first validated GPU run and are kept fixed for all reported experiments.

Development results produced using the local mock backend are not included in paper results.

## Video Decoding

Video-MME videos are decoded using OpenCV. We uniformly sample the required number of frames and convert the resulting RGB frames to the image representation expected by the multimodal processor.

This explicitly controlled decoding path is used for all pruning rates and avoids introducing rate-specific differences in video loading or temporal sampling.

## Latency Measurement

End-to-end generation latency is measured around the model generation call using wall-clock time.

Engine construction, weight loading, CUDA-graph capture, and vLLM multi-modal warm-up that occur during engine initialization are excluded from reported generation latency.

**Additionally, for every newly created pruning-rate engine we execute two discarded warm-up generates before collecting experiment measurements.** This absorbs residual Triton/JIT compilation (e.g. `_bilinear_pos_embed_kernel`) that otherwise contaminates the first timed request. Reported paper latency is therefore **steady-state warm inference latency**, not cold-start latency.

Logged prediction rows include `meta_latency_steady_state=true` once warm-up has completed for that engine.

All pruning conditions use the same measurement procedure.

We do not report streaming time-to-first-token because the Stage A implementation uses non-streaming offline inference and does not provide a reliable first-token timestamp.

## Evaluation Protocol

The complete evaluation proceeds as follows.

1. **Fixed-rate evaluation.**
   For each

   $$
   r\in\{0,\;0.25,\;0.50,\;0.75\},
   $$

   we initialize a vLLM engine using that pruning rate and evaluate every video-question pair. We record predictions, correctness, end-to-end latency, pre-pruning visual-token estimates, and nominal retained-token estimates.

2. **Per-video pruning tolerance.**
   We aggregate question-level outcomes by video and compute both the strict safe pruning rate and maximum observed correct pruning rate.

3. **Complexity estimation.**
   Each video receives a complexity score using the lightweight estimator described in the Method section.

4. **Calibration.**
   Videos are split into calibration and test subsets. Candidate \((T_1,T_2)\) pairs are evaluated using calibration videos only, after which the selected thresholds are frozen.

5. **Held-out BudgetVLM evaluation.**
   For every test video, BudgetVLM selects

   $$
   r(V)\in\{0.25,\;0.50,\;0.75\}
   $$

   using only the video's complexity score and the frozen thresholds.

6. **Policy comparison.**
   On the same held-out test videos, we compare:

   * no pruning;
   * fixed 25% pruning;
   * fixed 50% pruning;
   * fixed 75% pruning;
   * BudgetVLM; and
   * the oracle adaptive policy.

The primary comparison considers task accuracy together with average pruning rate, nominal retained visual-token budget, and end-to-end latency.

## Cached Policy Evaluation

The fixed-rate experiment produces one logged result for each

$$
(\text{video},\text{question},r)
$$

combination.

BudgetVLM does not re-run the VLM when evaluating a new pair of policy thresholds. Instead, once the rate for a video has been selected, the corresponding cached fixed-rate outcome is retrieved.

This design makes policy evaluation deterministic with respect to the fixed experiment matrix and allows alternative controllers to be analyzed without repeatedly executing expensive VLM inference.

All adaptive-policy results are therefore constrained to pruning conditions that were actually executed by the underlying model.

## Reproducibility

For each final experiment run, we retain:

* video IDs;
* question IDs;
* ground-truth answers;
* model predictions;
* pruning rate;
* pruning method;
* sampled frame count;
* complexity score;
* end-to-end latency;
* pre-pruning token estimate;
* retained-token estimate;
* calibration/test membership;
* frozen \(T_1,T_2\);
* model checkpoint;
* vLLM version;
* PyTorch/CUDA environment; and
* random seed.

Prediction-level results are stored in JSONL format, while aggregate tables and figures are regenerated from those artifacts.

## Scope of the Evaluation

The Stage A setup supports conclusions about the **quality-efficiency tradeoff available from content-dependent selection among fixed VidCom2 pruning rates**.

It does not by itself establish:

* improvements in streaming TTFT;
* exact post-pruning token counts unless engine instrumentation is added;
* the memory overhead of simultaneously maintaining multiple pruning-rate engines;
* scheduler behavior under concurrent multimodal workloads; or
* the performance of true per-request pruning-rate switching inside a single vLLM engine.

These systems questions are outside the scope of the Stage A evaluation and motivate the Stage B serving implementation.
