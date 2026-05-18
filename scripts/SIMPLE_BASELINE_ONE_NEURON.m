%% SIMPLE_BASELINE_ONE_NEURON.m
% Goal:
%   Run the simple baseline calcium-event detector on one neuron.
%
% Current scope:
%   - One neuron
%   - One sampling rate
%   - One simple baseline model
%
% Model idea:
%   Calcium events are detected using a combined evidence score:
%
%       event_score =
%           derivative_weight * positive_derivative
%         + amplitude_weight  * smoothed_calcium
%
% Important:
%   This is a simple baseline, not a full spike inference model.
%   It detects calcium-related events, not exact spike counts.

clear; clc; close all;

%% Add src folder to MATLAB path

% This allows MATLAB to find:
%   downsample_calcium_and_spikes()
%   run_simple_baseline()
addpath(fullfile('..', 'src'));

%% Load data

dataset_file = fullfile('..', 'data', 'MockData', 'data.1.train.preprocessed.mat');

raw = load(dataset_file);
data = raw.data;

%% Choose neuron and sampling rate

neuron_idx = 1;
target_fps = 20;

rec = data{neuron_idx};

calcium = double(rec.calcium(:));
spikes = double(rec.spikes(:));
original_fps = double(rec.fps);

%% Downsample calcium and spikes

% The model uses only calcium_ds.
% spikes_ds is kept as ground truth for visualization and later evaluation.
ds = downsample_calcium_and_spikes(calcium, spikes, original_fps, target_fps);

calcium_ds = ds.calcium;
spikes_ds = ds.spikes;
fps = ds.actual_fps;

fprintf('Neuron index: %d\n', neuron_idx);
fprintf('Target fps: %.1f Hz\n', target_fps);
fprintf('Actual fps: %.4f Hz\n', fps);
fprintf('Downsample factor: %d\n', ds.downsample_factor);
fprintf('Total spike count after downsampling: %.0f\n', ds.downsampled_spike_count);
fprintf('Active spike bins after downsampling: %.0f\n\n', ds.downsampled_active_spike_bins);

%% Create ground-truth event vector

% For this first baseline, we evaluate events, not exact spike counts.
% A ground-truth event is any time bin that contains at least one spike.
ground_truth_events = spikes_ds > 0;

%% Simple baseline parameters

params = struct();

% Smooth the calcium trace before calculating derivative.
% Smaller value = more sensitive but noisier.
params.smoothing_window_sec = 0.2;

% Weight of positive derivative in the event score.
% Higher value means the model cares more about rising phases.
params.derivative_weight = 0.7;

% Weight of calcium amplitude in the event score.
% Higher value means the model cares more about elevated calcium.
params.amplitude_weight = 0.3;

% Threshold on the combined event score.
% Lower value = more sensitive but more false positives.
params.event_score_threshold = 0.9;

% Minimum time between predicted events.
% This prevents detecting too many adjacent bins inside the same transient.
params.min_event_distance_sec = 0.10;

%% Run simple baseline model

baseline_result = run_simple_baseline(calcium_ds, fps, params);

predicted_events = baseline_result.predicted_events;
calcium_z = baseline_result.calcium_z;
calcium_smooth = baseline_result.calcium_smooth;

fprintf('Simple baseline parameters:\n');
fprintf('Smoothing window: %.3f sec (%d samples)\n', ...
    params.smoothing_window_sec, baseline_result.smoothing_window_samples);
fprintf('Derivative weight: %.2f\n', params.derivative_weight);
fprintf('Amplitude weight: %.2f\n', params.amplitude_weight);
fprintf('Event score threshold: %.2f\n', params.event_score_threshold);
fprintf('Minimum event distance: %.3f sec (%d samples)\n\n', ...
    params.min_event_distance_sec, baseline_result.min_event_distance_samples);

fprintf('Number of predicted events: %d\n', baseline_result.n_predicted_events);
fprintf('Number of ground-truth active spike bins: %d\n', sum(ground_truth_events));


%% Evaluate simple baseline prediction

% Matching tolerance:
% A predicted event is counted as correct if it is within this time window
% of a true event.
tolerance_sec = 0.10;

% Burst gap:
% In burst mode, spike bins separated by less than this gap are treated
% as part of the same activity burst.
burst_gap_sec = 0.10;

% Mode 1:
% Strict evaluation against every spike-containing bin.
metrics_spike_bins = evaluate_event_prediction( ...
    predicted_events, ...
    spikes_ds, ...
    fps, ...
    tolerance_sec, ...
    'spike_bins', ...
    burst_gap_sec);

% Mode 2:
% Burst-level evaluation using only burst onsets as true events.
metrics_bursts = evaluate_event_prediction( ...
    predicted_events, ...
    spikes_ds, ...
    fps, ...
    tolerance_sec, ...
    'burst_onsets', ...
    burst_gap_sec);

%% Print evaluation results

fprintf('\nEvaluation metrics: spike-bin mode\n');
fprintf('Tolerance: %.3f sec (%d bins)\n', ...
    metrics_spike_bins.tolerance_sec, metrics_spike_bins.tolerance_bins);
fprintf('True events:      %d\n', metrics_spike_bins.n_true_events);
fprintf('Predicted events: %d\n', metrics_spike_bins.n_predicted_events);
fprintf('Matched events:   %d\n', metrics_spike_bins.n_matched_events);
fprintf('TP: %d | FP: %d | FN: %d\n', ...
    metrics_spike_bins.TP, metrics_spike_bins.FP, metrics_spike_bins.FN);
fprintf('Precision: %.3f\n', metrics_spike_bins.precision);
fprintf('Recall:    %.3f\n', metrics_spike_bins.recall);
fprintf('F1:        %.3f\n', metrics_spike_bins.F1);
fprintf('Median abs timing error: %.4f sec\n', ...
    metrics_spike_bins.median_abs_timing_error_sec);
fprintf('Mean signed timing error: %.4f sec\n', ...
    metrics_spike_bins.mean_signed_timing_error_sec);

fprintf('\nEvaluation metrics: burst-onset mode\n');
fprintf('Tolerance: %.3f sec (%d bins)\n', ...
    metrics_bursts.tolerance_sec, metrics_bursts.tolerance_bins);
fprintf('Burst gap: %.3f sec\n', metrics_bursts.burst_gap_sec);
fprintf('True burst events: %d\n', metrics_bursts.n_true_events);
fprintf('Predicted events:  %d\n', metrics_bursts.n_predicted_events);
fprintf('Matched events:    %d\n', metrics_bursts.n_matched_events);
fprintf('TP: %d | FP: %d | FN: %d\n', ...
    metrics_bursts.TP, metrics_bursts.FP, metrics_bursts.FN);
fprintf('Precision: %.3f\n', metrics_bursts.precision);
fprintf('Recall:    %.3f\n', metrics_bursts.recall);
fprintf('F1:        %.3f\n', metrics_bursts.F1);
fprintf('Median abs timing error: %.4f sec\n', ...
    metrics_bursts.median_abs_timing_error_sec);
fprintf('Mean signed timing error: %.4f sec\n', ...
    metrics_bursts.mean_signed_timing_error_sec);

%% Visualize result

% Plot only a short window so the figure is readable.
plot_duration_sec = 30;

n_plot_samples = min(round(plot_duration_sec * fps), length(calcium_ds));

t = (0:n_plot_samples-1) / fps;

calcium_plot = calcium_z(1:n_plot_samples);
calcium_smooth_plot = calcium_smooth(1:n_plot_samples);
ground_truth_plot = ground_truth_events(1:n_plot_samples);
predicted_plot = predicted_events(1:n_plot_samples);

%% Prepare event markers

% Ground-truth spike/event bins
gt_idx = ground_truth_plot > 0;
gt_times = t(gt_idx);
gt_y = -3 * ones(size(gt_times));

% Predicted calcium events
pred_idx = predicted_plot > 0;
pred_times = t(pred_idx);

%% Plot

figure('Color', 'w');

plot(t, calcium_plot, 'LineWidth', 1.0);
hold on;

plot(t, calcium_smooth_plot, 'LineWidth', 1.5);

% Ground-truth events
plot(gt_times, gt_y, ...
    'o', ...
    'MarkerSize', 4, ...
    'LineStyle', 'none');

% Predicted events as vertical dashed lines.
% Use a dummy line so the dashed-line style appears once in the legend.
plot(NaN, NaN, '--', ...
    'LineWidth', 1.0);

yl = ylim;

for i = 1:length(pred_times)
    xline(pred_times(i), '--', ...
        'LineWidth', 1.0, ...
        'HandleVisibility', 'off');
end

ylim(yl);

xlabel('Time (s)');
ylabel('Z-scored fluorescence / event markers');

title(sprintf('Simple baseline — neuron %d, %.1f Hz', neuron_idx, fps));

legend({ ...
    'Calcium trace (z-scored)', ...
    'Smoothed calcium', ...
    'Ground-truth spike bins', ...
    'Predicted calcium events'}, ...
    'Location', 'best');

grid on;