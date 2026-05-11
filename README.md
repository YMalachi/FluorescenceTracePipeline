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

The uploaded example file contains one subset of the dataset. Each recording/neuron contains fields such as:

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

## Important Observation

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

## Planned Next Steps

1. Implement a simple baseline event detector.
2. Test it on one neuron and one sampling rate.
3. Compare predicted events to ground-truth spike bins.
4. Add evaluation metrics.
5. Run the baseline across all sampling rates and neurons.
6. Later compare additional methods such as OASIS, Suite2p deconvolution, and STM.

## Current Development Strategy

We are intentionally building the project in small parts:

```text
inspect data
↓
visualize traces
↓
downsample correctly
↓
build a simple baseline
↓
evaluate
↓
add advanced models
```

This keeps the pipeline understandable, testable, and easy to debug.
