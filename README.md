# Autonomous driving 

## 1. DINO 

**Embeddings analysis**

1. DINO
2. PCA
3. UMAP

![Embeddings analysis image - UMAP](readme-data/umap-analysis.png)

**Retrieval demo**
![alt text](readme-data/demo-retrieval-readme.png)

## 2. Behavioral Cloning on comma2k19

The behavioral cloning stage uses the [comma2k19](https://github.com/commaai/comma2k19) dataset to predict steering angle from camera frames.
The current BC pipeline is intentionally split into three notebooks:

- `src/notebooks/3-behavioral_cloning-comma2k19.ipynb`: single-frame DINO embedding baseline.
- `src/notebooks/4_bc_dataset_analysis.ipynb`: steering target and video sanity-check analysis.
- `src/notebooks/5-temporal_behavioral_cloning-comma2k19.ipynb`: temporal BC experiments with an indexed single-frame baseline, temporal MLP, and GRU.

### 2.1 Single-frame DINO + MLP baseline

Implemented in `src/notebooks/3-behavioral_cloning-comma2k19.ipynb`.

The first BC baseline predicts the steering angle from one camera frame. Frames are encoded with a frozen DINO encoder, and a small MLP policy head is trained on top of the cached `[CLS]` embeddings.

```text
Image
↓
DINO Encoder (Frozen)
↓
MLP Policy Head
↓
Normalized steering angle
```

What was implemented:

- Built a comma2k19 frame index by traversing chunks, routes, and segments.
- Matched every selected video frame with the nearest CAN `steering_angle` timestamp.
- Added RGB conversion before passing OpenCV frames into the DINO image processor.
- Cached DINO embeddings so BC experiments can train only lightweight heads.
- Added `CommaDataset` over precomputed embeddings and steering targets.
- Added `CommaBCNet`: `LayerNorm(768) -> Linear -> GELU -> Dropout -> LayerNorm -> Linear -> GELU -> Dropout -> Linear(1)`.
- Trained with `GroupKFold` split by `route`, so validation routes are not mixed with training routes.
- Normalized steering targets per train fold: the model optimizes MSE in normalized target space, then predictions are denormalized back to degrees for reporting.
- Added AdamW support with weight decay, while keeping SGD as an option.
- Logged all train/validation metrics to ClearML.

Logged metrics are grouped in ClearML by plot title and series to keep the Scalars UI readable:

- `train/Loss_norm` and `val/Loss_norm`: MSE loss in normalized target space, with `fold_{n}` as the series.
- `train/MSE` and `val/MSE`: steering-angle MSE in degrees, with `fold_{n}` as the series.
- `val/MAE` and `val/RMSE`: validation metrics in degrees.
- `val/{MSE,MAE,RMSE}/baselines`: baseline error metrics for `baseline_zero` and `baseline_train_mean`.
- `val/*/thresholds_mean`: threshold metrics averaged across folds for steering magnitude bins `0-3`, `3-10`, `10-30`, and `30-inf` degrees.
- `val/Count/thresholds_total`: total number of validation samples in each threshold bucket across folds.
- `val/threshold_metrics_table`: ClearML table with per-epoch, per-fold threshold metrics for detailed inspection.
- `Direction_acc`: steering direction accuracy with a 1 degree dead zone around zero.
- `Angle_ratio_mean`: clipped mean of `predicted_angle / target_angle`, computed only for targets with absolute angle at least 1 degree.

Per-fold threshold scalar logging is disabled by default to avoid plots with too many lines. Set `cfg.log_fold_threshold_scalars = True` to enable detailed per-fold threshold scalar plots (`val/*/thresholds_by_fold`). Count-like helper values for direction and ratio are kept internally for correct averaging, but are not logged as separate scalar plots by default.

Key limitation found at this stage: a single image often does not contain enough temporal information to infer how the steering wheel is currently moving. This is especially visible on larger steering angles, where the model tends to understeer and regress toward small angles.

### 2.2 Dataset and target analysis

Implemented in `src/notebooks/4_bc_dataset_analysis.ipynb`.

This notebook checks whether the training target and frame indexing are sane before adding more model complexity.

What was implemented:

- Reused the comma2k19 index builder from notebook 3.
- Created a dataframe with route, segment, frame path/index, timestamp, and matched steering angle.
- Plotted the raw steering angle sequence to inspect distribution, continuity, and outliers.
- Added a steering-wheel overlay renderer with OpenCV.
- Generated a video clip from a selected dataframe slice (`df[20000:30000]`) with the current steering angle drawn on top of the driving frame.

The video is useful as a qualitative check that the matched CAN steering signal has the expected direction and timing relative to the road geometry.

### 2.3 Temporal Behavioral Cloning

Implemented in `src/notebooks/5-temporal_behavioral_cloning-comma2k19.ipynb`.

The temporal stage tests whether a short history of DINO embeddings improves steering prediction compared with a one-frame baseline trained on the same valid temporal subset.

```text
Image[t-4:t]
↓
Frozen DINO embeddings
↓
Temporal head
↓
Normalized steering angle[t]
```

What was implemented:

- Added `TemporalCommaDataset`, which returns a consecutive window of embeddings and predicts the steering angle at the last frame.
- Valid temporal samples are filtered so every sequence stays inside the same video and frame indices are consecutive.
- Kept an indexed `CommaDataset` mode for a fair single-frame baseline on exactly the same target frames as the temporal dataset.
- Added `TemporalFlattenCommaBCNet`: concatenates a sequence of embeddings and applies an MLP head.
- Added `TemporalGRUCommaBCNet`: feeds the embedding sequence into a GRU and predicts from the last hidden output.
- Reused the same `GroupKFold` by route, target normalization, AdamW optimizer, baseline metrics, threshold metrics, direction accuracy, understeering ratio, and ClearML table logging from the single-frame notebook.

Temporal experiment variants:

| Run | Input | Head | Purpose |
| --- | --- | --- | --- |
| `dino_mlp-baseline_indexed-2026-08-17_14-21` | `embedding[t]` | MLP | Single-frame baseline restricted to temporal-valid target frames. |
| `dino_mlp-temporal_bc-2026-08-17_13-43-35` | `embedding[t-4:t]` | Flatten + MLP | Tests whether simple concatenation of recent visual context helps. |
| `dino_mlp-temporal_bc_gru-2026-08-17_16-42-06` | `embedding[t-4:t]` | GRU + MLP | Tests whether a recurrent head models short-term dynamics better than flattening. |

### 2.4 Experiment comparison

The three downloaded ClearML JSON files contain the exported `val/threshold_metrics_table`. The comparison below aggregates fold rows with `count` weighting. The "best epoch" table uses the lowest overall weighted validation MSE across epochs; the threshold table shows the final epoch (`14`) because that is what the exported final table is usually inspected for.

Best validation epoch:

| Run | Best epoch | MSE | MAE | RMSE | Direction acc | Angle ratio mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Indexed single-frame MLP | 7 | 310.13 | 4.36 | 17.61 | 0.488 | 0.320 |
| Temporal flatten MLP | 11 | 297.07 | 4.12 | 17.24 | 0.454 | 0.285 |
| Temporal GRU | 12 | **283.74** | **4.06** | **16.84** | **0.551** | **0.365** |

Final epoch (`14`) by steering magnitude:

| Run | Threshold | Count | MSE | MAE | RMSE | Direction acc | Angle ratio mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Indexed single-frame MLP | 0-3 | 142740 | 30.61 | 1.79 | 5.53 | 0.306 | 0.219 |
| Indexed single-frame MLP | 3-10 | 70120 | 54.99 | 4.11 | 7.42 | 0.679 | 0.479 |
| Indexed single-frame MLP | 10-30 | 7871 | 389.37 | 15.28 | 19.73 | 0.568 | 0.147 |
| Indexed single-frame MLP | 30-inf | 3300 | 20329.67 | 97.11 | 142.58 | 0.498 | -0.013 |
| Temporal flatten MLP | 0-3 | 142740 | 36.34 | 1.67 | 6.03 | 0.244 | 0.186 |
| Temporal flatten MLP | 3-10 | 70120 | 45.37 | 4.07 | 6.74 | 0.656 | 0.431 |
| Temporal flatten MLP | 10-30 | 7871 | 407.74 | 15.43 | 20.19 | 0.575 | 0.133 |
| Temporal flatten MLP | 30-inf | 3300 | 21013.41 | 97.89 | 144.96 | 0.493 | -0.026 |
| Temporal GRU | 0-3 | 142740 | **12.12** | **1.56** | **3.48** | **0.357** | **0.294** |
| Temporal GRU | 3-10 | 70120 | **29.43** | **3.91** | **5.43** | 0.661 | 0.436 |
| Temporal GRU | 10-30 | 7871 | **311.74** | **15.06** | **17.66** | 0.537 | 0.131 |
| Temporal GRU | 30-inf | 3300 | **17780.58** | **95.78** | **133.34** | 0.490 | -0.006 |

Main takeaways:

- The indexed single-frame baseline is the correct reference for temporal experiments because it uses the same target frames as the sequence datasets.
- Temporal flattening slightly improves the best overall MSE/MAE over the indexed baseline, but its final epoch regresses, so it likely needs early stopping or stronger regularization.
- The GRU is the best of the three tested heads on overall MSE, MAE, RMSE, direction accuracy, and understeering ratio.
- All models still struggle on large turns (`30-inf`). This bucket is much smaller than the straight-driving buckets and has very high error, so the next useful experiments are weighted loss/sampling for turns, speed input, predicting steering delta instead of absolute angle, and post-training temporal metrics such as smoothness and jerk error.
