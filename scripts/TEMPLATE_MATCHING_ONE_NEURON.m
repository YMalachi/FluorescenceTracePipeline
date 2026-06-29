%% TEMPLATE_MATCHING_ONE_NEURON.m
% Goal:
%   Test the template-matching model on one calcium trace.
%
% Current scope:
%   - One neuron/recording
%   - One sampling rate
%   - Template-matching model only
%
% Model idea:
%   A calcium event has a characteristic transient shape:
%   fast rise + slower decay.
%
%   The model builds a calcium-event template, slides it across the trace,
%   and detects peaks in the template-matching score.

clear; clc; close all;

%% Add project paths

% Project source functions:
%   downsample_calcium_and_spikes()
%   run_template_matching_model()
%   evaluate_event_prediction()
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
% spikes_ds is kept as ground truth for visualization and evaluation.
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

% A ground-truth spike-bin event is any bin containing at least one spike.
ground_truth_events = spikes_ds > 0;

%% Template-matching model parameters

template_params = struct();

% Smooth calcium slightly before template matching.
template_params.smoothing_window_sec = 0.10;

% Template shape parameters.
% These define a fast-rise / slow-decay calcium transient.
template_params.template_rise_sec = 0.05;
template_params.template_decay_sec = 0.30;
template_params.template_duration_sec = 1.00;

% Event detection parameters.
% Lower threshold = more sensitive, but more false positives.
template_params.score_threshold_z = 1.0;
template_params.min_event_distance_sec = 0.10;

%% Run template-matching model

fprintf('Template-matching parameters:\n');
fprintf('Smoothing window: %.3f sec\n', template_params.smoothing_window_sec);
fprintf('Template rise: %.3f sec\n', template_params.template_rise_sec);
fprintf('Template decay: %.3f sec\n', template_params.template_decay_sec);
fprintf('Template duration: %.3f sec\n', template_params.template_duration_sec);
fprintf('Score threshold: %.2f z-score\n', template_params.score_threshold_z);
fprintf('Minimum event distance: %.3f sec\n\n', template_params.min_event_distance_sec);

template_result = run_template_matching_model( ...
    calcium_ds, ...
    fps, ...
    template_params);

predicted_events = template_result.predicted_events;

fprintf('Number of predicted events: %d\n', template_result.n_predicted_events);
fprintf('Number of ground-truth active spike bins: %d\n', sum(ground_truth_events));

%% Evaluate template-matching prediction

tolerance_sec = 0.10;
burst_gap_sec = 0.10;

metrics_spike_bins = evaluate_event_prediction( ...
    predicted_events, ...
    spikes_ds, ...
    fps, ...
    tolerance_sec, ...
    'spike_bins', ...
    burst_gap_sec);

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
fprintf('Timing-weighted recall: %.4f\n', ...
    metrics_spike_bins.timing_weighted_recall);

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
fprintf('Timing-weighted recall: %.4f\n', ...
    metrics_bursts.timing_weighted_recall);

%% Visualize result

plot_duration_sec = 30;
plot_start_sec = 0;

n_total_samples = length(calcium_ds);

start_sample = max(1, round(plot_start_sec * fps) + 1);
end_sample = min(n_total_samples, round((plot_start_sec + plot_duration_sec) * fps));

plot_idx = start_sample:end_sample;
t = (plot_idx - 1) / fps;

calcium_plot = template_result.calcium_z(plot_idx);
calcium_smooth_plot = template_result.calcium_smooth(plot_idx);
score_plot = template_result.event_score_z(plot_idx);

ground_truth_plot = ground_truth_events(plot_idx);
predicted_plot = predicted_events(plot_idx);

%% Prepare event markers

% Ground-truth spike-bin events
gt_idx = ground_truth_plot > 0;
gt_times = t(gt_idx);
gt_y = -3 * ones(size(gt_times));

% Predicted template-matching events
pred_idx = predicted_plot > 0;
pred_times = t(pred_idx);

%% Figure 1: calcium + predictions

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
plot(NaN, NaN, '--', 'LineWidth', 1.0);

yl = ylim;

for i = 1:length(pred_times)
    xline(pred_times(i), '--', ...
        'LineWidth', 1.0, ...
        'HandleVisibility', 'off');
end

ylim(yl);

xlabel('Time (s)');
ylabel('Z-scored fluorescence / event markers');

title(sprintf('Template matching — neuron %d, %.1f Hz', neuron_idx, fps));

legend({ ...
    'Calcium trace (z-scored)', ...
    'Smoothed calcium', ...
    'Ground-truth spike bins', ...
    'Template-matching predictions'}, ...
    'Location', 'best');

grid on;

%% Figure 2: template-matching score

figure('Color', 'w');

plot(t, score_plot, 'LineWidth', 1.0);
hold on;

yline(template_params.score_threshold_z, '--', ...
    'Threshold', ...
    'LineWidth', 1.0);

yl = ylim;

for i = 1:length(pred_times)
    xline(pred_times(i), '--', ...
        'LineWidth', 1.0, ...
        'HandleVisibility', 'off');
end

ylim(yl);

xlabel('Time (s)');
ylabel('Template score (z-score)');
title(sprintf('Template-matching score — neuron %d, %.1f Hz', neuron_idx, fps));

legend({ ...
    'Template-matching score', ...
    'Detection threshold', ...
    'Predicted events'}, ...
    'Location', 'best');

grid on;

%% Figure 3: template shape

figure('Color', 'w');

plot(template_result.template_time, template_result.template, 'LineWidth', 1.5);

xlabel('Time (s)');
ylabel('Normalized template amplitude');
title('Calcium-event template shape');

grid on;