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
save comparison results to CSV
```

## Work Completed So Far

### 1. Data inspection

We wrote an inspection script to load the `.mat` file and check:

- Which variables exist in the file
- How many recordings/neuron traces are included
- What fields each recording contains
- Trace length
- Sampling rate
- Recording duration
- Total spike count
- Mean firing rate

This confirmed that the calcium traces and spike trains are aligned and sampled at approximately 100 Hz.

### 2. Visualization of one neuron

We plotted a calcium fluorescence trace together with ground-truth spike markers.

This helped us verify visually that:

- The calcium signal looks reasonable
- Spike bins are present at expected times
- The data can be inspected before building models

### 3. Downsampling logic

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

### 4. Downsampling sanity checks

We wrote a downsampling function with sanity checks for:

- Matching calcium/spike lengths
- Valid sampling rates
- NaN or Inf values
- Negative spike counts
- Non-integer spike counts
- Spike-count preservation after re-binning

We also tested the function across all neurons in the current file and all target sampling rates.

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

The current score is:

```text
event_score =
    derivative_weight * positive_derivative
  + amplitude_weight  * smoothed_calcium
```

Current baseline parameters used during development:

```matlab
smoothing_window_sec = 0.2;
derivative_weight = 0.7;
amplitude_weight = 0.3;
event_score_threshold = 0.9;
min_event_distance_sec = 0.10;
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

Current OASIS parameters used during development:

```matlab
ar_model = 'ar1';
method = 'foopsi';
optimize_b = true;
optimize_pars = true;
oasis_threshold_z = 1.5;
min_event_distance_sec = 0.10;
```

These parameters are not final. They will be optimized later using validation data.

## Evaluation Strategy

We implemented a shared evaluation function:

```matlab
evaluate_event_prediction()
```

This function compares predicted event times to the ground-truth binned spike train.

The evaluation currently supports two modes:

### 1. Spike-bin evaluation

In this mode, every bin with at least one spike is treated as a true event.

```matlab
true_events = spikes > 0;
```

This is the stricter evaluation mode.

It asks:

```text
Did the model detect spike-containing time bins?
```

### 2. Burst-onset evaluation

In this mode, nearby spike bins are grouped into activity bursts, and only the first bin of each burst is treated as the true event.

This is usually fairer for calcium-event detectors because one calcium transient can correspond to multiple spike-containing bins.

It asks:

```text
Did the model detect the onset of an activity episode?
```

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

Timing error is calculated for matched events:

```text
timing_error = predicted_event_time - true_event_time
```

Interpretation:

```text
positive timing error = prediction occurred after the true event
negative timing error = prediction occurred before the true event
```

## Model Comparison

We created a comparison script for:

```text
Simple Baseline vs OASIS
```

The script runs both models on one neuron across four sampling rates:

```text
100 Hz
50 Hz
20 Hz
10 Hz
```

For each sampling rate, both models are evaluated using:

```text
spike-bin mode
burst-onset mode
```

The script prints a compact comparison table and saves a detailed CSV file.

Current output file:

```text
results/tables/baseline_vs_oasis_one_neuron_all_fps.csv
```

Each row in the CSV represents:

```text
model × sampling rate × evaluation mode
```

## Current Development Strategy

We are intentionally building the project in small, testable parts:

```text
inspect data
↓
visualize traces
↓
downsample correctly
↓
implement simple baseline
↓
evaluate predictions
↓
integrate OASIS
↓
compare models
↓
scale to all neurons and sampling rates
```

This keeps the pipeline understandable, testable, and easy to debug.

## Planned Next Steps

1. Run the current Simple Baseline vs OASIS comparison across all available neurons.
2. Add train / validation / test splitting across neurons.
3. Optimize model hyperparameters on validation neurons.
4. Evaluate final model performance on held-out test neurons.
5. Generate detailed and summary result tables:
   - detailed table: one row per model × neuron × sampling rate × evaluation mode
   - summary table: one row per model × sampling rate × evaluation mode
6. Add additional models if feasible, such as:
   - Suite2p deconvolution-related logic
   - STM or another supervised/statistical model

## Future Scaling Plan

The final evaluation should compare models across:

```text
~50 neurons
×
4 sampling rates
×
multiple models
×
2 evaluation modes
```

The intended final result structure is:

```text
one score per model per sampling rate
```

Main scores will likely include:

- Mean / median burst-onset F1 across neurons
- Mean / median spike-bin F1 across neurons
- Median absolute timing error across neurons
- Mean signed timing bias across neurons

## External Dependencies and Credit

### OASIS

This project currently uses the OASIS MATLAB toolbox for calcium deconvolution.

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

Current results should be interpreted as development-stage outputs, not final model performance estimates. Reliable conclusions will require systematic testing across all neurons, sampling rates, and held-out evaluation data.