# Calcium Spike Inference Pipeline

## Project Goal

This project aims to build and evaluate a trace-level spike inference pipeline for calcium imaging data.

At this stage, we assume that image processing and ROI extraction have already been performed. Therefore, the pipeline starts from extracted calcium fluorescence traces and tries to infer neural activity over time.

The long-term goal is to compare several spike inference approaches under different sampling rates, especially rates relevant to our experimental setup.

## Data

We are currently using mock/benchmark data from the CRCNS calcium imaging dataset. The data includes:

- Calcium fluorescence traces
- Ground-truth electrophysiological spike trains
- Exact spike times
- Sampling rate information

Each recording/neuron contains fields such as:

```matlab
calcium
spikes
spike_times
fps
```

Some dataset files contain multiple recordings from the same biological cell. When available, the `cell_num` field is used to identify repeated segments from the same cell.

## Current Pipeline Scope

Current input:

```text
calcium fluorescence trace over time
```

Current ground truth:

```text
binned electrophysiological spike train
```

Current output target:

```text
predicted neural events / spike-related activity
```

At this stage, we are not yet processing TIFF movies or extracting fluorescence from ROIs.

## Current Pipeline Structure

The current pipeline follows this logic:

```text
load calcium/spike data
↓
inspect and validate data structure
↓
downsample calcium and spike trains
↓
run spike/event inference model
↓
evaluate predicted events
↓
save detailed and summary results to CSV
↓
visualize and diagnose model behavior
```

## Work Completed So Far

### 1. Data Inspection

We wrote an inspection script to load the `.mat` files and check:

- Which variables exist in each file
- How many recordings/neuron traces are included
- What fields each recording contains
- Trace length
- Sampling rate
- Recording duration
- Total spike count
- Mean firing rate

This confirmed that the calcium traces and spike trains are aligned and sampled at approximately 100 Hz before downsampling.

### 2. Visualization of One Neuron

We plotted calcium fluorescence traces together with ground-truth spike markers.

This helped verify visually that:

- The calcium signal can be inspected before model development
- Spike bins are present at expected times
- Some recordings are much noisier than others
- Diagnostic visualization is important for understanding model behavior

### 3. Downsampling Logic

Because the final lab data may have lower temporal resolution than the benchmark data, we tested downsampling from approximately 100 Hz to:

```text
100 Hz
50 Hz
20 Hz
10 Hz
```

The downsampling logic is:

- Calcium trace: averaged within each new time bin
- Spike train: summed within each new time bin

This preserves total spike counts while reducing temporal resolution.

### 4. Downsampling Sanity Checks

We wrote a downsampling function with sanity checks for:

- Matching calcium/spike lengths
- Valid sampling rates
- NaN or Inf values
- Negative spike counts
- Non-integer spike counts
- Spike-count preservation after re-binning
- Active spike-bin reduction after downsampling
- Multi-spike bins created by temporal merging

We also tested the function across all neurons in the current dataset files and all target sampling rates.

## Important Downsampling Observation

When downsampling, total spike count can be preserved, but separate spike bins may merge into the same lower-resolution bin.

Therefore, we track both:

```text
Total spike count
```

and:

```text
Number of active spike bins
```

This distinction is important because lower sampling rates reduce temporal precision even if they do not remove spikes.

## Implemented Models

### 1. Simple Baseline Model

We implemented a simple rule-based calcium event detector.

The model uses a combined evidence score based on:

1. Positive calcium derivative
2. Smoothed calcium amplitude

The score is:

```text
event_score =
    derivative_weight * positive_derivative
  + amplitude_weight  * smoothed_calcium
```

The baseline logic was moved into a reusable function:

```matlab
run_simple_baseline()
```

The function returns:

- Predicted binary event vector
- Event score
- Normalized calcium trace
- Smoothed calcium trace
- Model parameters
- Number of predicted events

Development-stage parameters included:

```matlab
smoothing_window_sec = 0.2;
derivative_weight = 0.7;
amplitude_weight = 0.3;
event_score_threshold = 0.9;
min_event_distance_sec = 0.10;
```

These are not final. The parameters are later optimized on validation data.

### 2. OASIS Model

We integrated the OASIS MATLAB toolbox as an external dependency.

OASIS performs calcium deconvolution and estimates a spike-like activity signal from fluorescence traces. The OASIS output is then converted into predicted event times by thresholding peaks in the inferred activity signal.

The OASIS logic was moved into a reusable function:

```matlab
run_oasis_model()
```

The function returns:

- Predicted binary event vector
- OASIS denoised calcium estimate
- OASIS inferred activity signal
- Z-scored OASIS activity signal
- OASIS parameters
- Number of predicted events

Development-stage OASIS parameters included:

```matlab
ar_model = 'ar1';
method = 'foopsi';
optimize_b = true;
optimize_pars = true;
oasis_threshold_z = 1.5;
min_event_distance_sec = 0.10;
```

These parameters are not final. Event-conversion parameters are optimized separately for each sampling rate.

## Evaluation Strategy

We implemented a shared evaluation function:

```matlab
evaluate_event_prediction()
```

This function compares predicted event times to the ground-truth binned spike train.

The evaluation currently supports two modes.

### 1. Spike-bin Evaluation

In this mode, every bin with at least one spike is treated as a true event.

```matlab
true_events = spikes > 0;
```

This is the stricter evaluation mode.

It asks:

```text
Did the model detect spike-containing time bins?
```

This mode is currently treated as the more relevant primary evaluation mode for spike inference over time.

### 2. Burst-onset Evaluation

In this mode, nearby spike bins are grouped into activity bursts, and only the first bin of each burst is treated as the true event.

Example:

```text
spike bins:    0 1 1 1 0 0 1 1 0
burst onsets:  0 1 0 0 0 0 1 0 0
```

This mode asks:

```text
Did the model detect the onset of an activity episode?
```

This is useful as a secondary diagnostic mode because calcium transients can represent broader activity episodes rather than every individual spike-containing bin.

## Evaluation Metrics

For each model, sampling rate, and evaluation mode, we calculate:

- True positives
- False positives
- False negatives
- Precision
- Recall
- F1 score
- Mean absolute timing error
- Median absolute timing error
- Mean signed timing error
- Median signed timing error
- Standard deviation of timing error
- Mean timing score
- Total timing score
- Timing-weighted recall

Timing error is calculated for matched events:

```text
timing_error = predicted_event_time - true_event_time
```

Interpretation:

```text
positive timing error = prediction occurred after the true event
negative timing error = prediction occurred before the true event
```

## Graded Timing Score

In addition to binary match/no-match evaluation, we added a graded timing score for matched events.

For each matched event:

```text
timing_score = max(0, 1 - abs(timing_error_sec) / tolerance_sec)
```

Interpretation:

```text
score = 1   → exact timing
score = 0.5 → halfway to the tolerance boundary
score = 0   → at the tolerance boundary
```

This score gives partial credit for predictions that are close in time but not perfectly aligned.

We also calculate:

```text
timing_weighted_recall = total_timing_score / number_of_true_events
```

This rewards models that both detect many true events and detect them close in time.

## Model Comparison Across All Data

We expanded the comparison workflow from one-neuron testing to all available mock dataset files.

The scalable comparison script runs:

```text
all dataset files
×
all recordings
×
4 sampling rates
×
all implemented models
×
2 evaluation modes
```

Current sampling rates:

```text
100 Hz
50 Hz
20 Hz
10 Hz
```

Current models:

```text
SimpleBaseline
OASIS
```

Current evaluation modes:

```text
spike_bins
burst_onsets
```

The script saves a detailed result table and a summary table.

Detailed output:

```text
results/tables/model_results_detailed.csv
```

Each row represents:

```text
dataset × recording × sampling rate × model × evaluation mode
```

Summary output:

```text
results/tables/model_results_summary.csv
```

Each row represents:

```text
model × sampling rate × evaluation mode
```

## Diagnostic Plots

We added plotting utilities that read saved CSV results and generate diagnostic figures.

Current plots include:

- Mean F1 vs sampling rate
- Timing-weighted recall vs sampling rate
- Precision vs recall scatter plots
- F1 distributions across recordings

These plots are used for development and diagnostics, not final conclusions.

## Train / Validation / Test Split

We created a split table to support fair hyperparameter optimization and later final evaluation.

Output file:

```text
results/tables/data_split.csv
```

The split is done by biological unit identity, not by result rows.

If a recording has `cell_num`, the unit identity is based on:

```text
dataset_id + cell_num
```

If a recording does not have `cell_num`, the unit identity is based on:

```text
dataset_id + recording_idx
```

This prevents segments from the same biological cell from leaking across train, validation, and test splits.

The split is performed within each dataset file to reduce the risk of one dataset being overrepresented in only one split.

Current split structure:

```text
train
validation
test
```

## Hyperparameter Optimization

We added validation-based hyperparameter optimization.

Optimization is performed separately for each sampling rate:

```text
100 Hz
50 Hz
20 Hz
10 Hz
```

This means each model can have a different best parameter set for each sampling rate.

Current optimization output files:

```text
results/tables/hyperparameter_search_results_spike_bins.csv
results/tables/best_model_params_by_fps_spike_bins.csv
```

The full search file contains one row per:

```text
model × sampling rate × parameter combination
```

The best-parameters file contains one row per:

```text
model × sampling rate
```

The current primary optimization objective is spike-bin F1 on the validation split.

Timing-weighted recall and timing error are saved as secondary metrics.

## Current Development Strategy

We are intentionally building the project in small, testable parts:

```text
inspect data
↓
visualize traces
↓
downsample correctly
↓
implement reusable model functions
↓
evaluate predictions
↓
compare models across all data
↓
split data by biological unit
↓
optimize hyperparameters on validation data
↓
diagnose model failure cases
↓
evaluate final performance on held-out test data
```

This keeps the pipeline understandable, testable, and easy to debug.

## Planned Next Steps

1. Use the optimized parameter table to run final evaluation on the held-out test split.
2. Save final detailed and summary test-result tables.
3. Continue using diagnostic plots to understand model failures.
4. Add additional models if feasible.
5. Compare models using:
   - spike-bin F1
   - burst-onset F1
   - timing-weighted recall
   - median absolute timing error
   - mean signed timing bias

## Future Scaling Plan

The final evaluation should compare models across:

```text
~50 biological units
×
4 sampling rates
×
multiple models
×
2 evaluation modes
```

The intended final result structure is:

```text
one performance summary per model per sampling rate
```

Main reported scores will likely include:

- Mean / median spike-bin F1 across units
- Mean / median burst-onset F1 across units
- Mean / median timing-weighted recall
- Median absolute timing error across units
- Mean signed timing bias across units

## External Dependencies and Credit

## External Toolboxes

This repository does not directly include external third-party toolboxes.

External toolboxes should be placed locally inside:

```text
external/
```

The `external/` folder is intentionally not tracked by Git and should be listed in `.gitignore`:

```gitignore
external/
```

This keeps the repository focused on our own code and avoids copying third-party toolbox code directly into the project.

Recommended local folder structure:

```text
project_root/
├── data/
├── external/
│   └── OASIS_matlab/
├── results/
├── scripts/
├── src/
└── README.md
```

External toolboxes are used as dependencies. They are not developed as part of this repository.

### OASIS

This project uses the OASIS MATLAB toolbox for calcium deconvolution.

OASIS is not included directly in this repository. To run OASIS-based scripts, download or clone OASIS locally into:

```text
external/OASIS_matlab/
```

The scripts add OASIS to the MATLAB path using:

```matlab
addpath(genpath(fullfile('..', 'external', 'OASIS_matlab')));
```

If the OASIS folder is missing, scripts that use `run_oasis_model()` will not run correctly.

If this work is used in a publication, report, or formal presentation, OASIS should be credited and cited.

Suggested citation:

```text
Friedrich J, Zhou P, Paninski L. (2017).
Fast online deconvolution of calcium imaging data.
PLoS Computational Biology, 13(3): e1005423.
doi:10.1371/journal.pcbi.1005423
```

OASIS is used here as an external deconvolution method. The current project wraps OASIS output into our own event-prediction and evaluation framework.

## Notes

The current implementation is still under active development.

Current results should be interpreted as development-stage outputs, not final model performance estimates. Reliable conclusions will require systematic testing across all neurons, sampling rates, optimized parameters, and held-out evaluation data.