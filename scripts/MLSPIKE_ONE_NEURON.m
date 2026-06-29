%% MLSPIKE_ONE_NEURON.m
% Goal:
%   Test the MLspike model on one calcium trace.
%
% Current scope:
%   - One neuron/recording
%   - One sampling rate
%   - MLspike only
%
% Important:
%   MLspike expects calcium with a nonzero baseline, not z-scored calcium.
%   The wrapper run_mlspike_model() handles this internally.

clear; clc; close all;

%% Add project paths

% Project source functions:
%   downsample_calcium_and_spikes()
%   run_mlspike_model()
%   evaluate_event_prediction()
addpath(fullfile('..', 'src'));

% Adding toolbox
addpath(genpath(fullfile('..', 'external')));
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

%% MLspike parameters

mlspike_params = struct();

% Start without autocalibration.
% This keeps the first test simpler and easier to debug.
mlspike_params.use_autocalibration = false;

% Fixed physiological/noise parameters.
% These are initial values only. We will tune later if the model works.
mlspike_params.a = 0.07;
mlspike_params.tau = 1.0;
mlspike_params.sigma = 0.02;
mlspike_params.saturation = 0.1;
mlspike_params.drift_parameter = 0.01;

% Bounds used only if use_autocalibration = true.
mlspike_params.amin = 0.02;
mlspike_params.amax = 0.20;
mlspike_params.taumin = 0.20;
mlspike_params.taumax = 2.00;

%% Run MLspike model

fprintf('MLspike parameters:\n');
fprintf('Use autocalibration: %d\n', mlspike_params.use_autocalibration);
fprintf('a: %.4f\n', mlspike_params.a);
fprintf('tau: %.4f sec\n', mlspike_params.tau);
fprintf('sigma: %.4f\n', mlspike_params.sigma);
fprintf('saturation: %.4f\n', mlspike_params.saturation);
fprintf('drift parameter: %.4f\n\n', mlspike_params.drift_parameter);

fprintf('Running MLspike...\n');

mlspike_result = run_mlspike_model( ...
    calcium_ds, ...
    fps, ...
    mlspike_params);

predicted_events = mlspike_result.predicted_events;

fprintf('MLspike finished.\n');
fprintf('Number of predicted events: %d\n', mlspike_result.n_predicted_events);
fprintf('Number of estimated spike times: %d\n', length(mlspike_result.spike_times_sec));
fprintf('Number of ground-truth active spike bins: %d\n', sum(ground_truth_events));

%% Evaluate MLspike prediction

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
print_metrics(metrics_spike_bins);

fprintf('\nEvaluation metrics: burst-onset mode\n');
print_metrics(metrics_bursts);

%% Visualize result

plot_duration_sec = 30;
plot_start_sec = 0;

n_total_samples = length(calcium_ds);

start_sample = max(1, round(plot_start_sec * fps) + 1);
end_sample = min(n_total_samples, round((plot_start_sec + plot_duration_sec) * fps));

plot_idx = start_sample:end_sample;
t = (plot_idx - 1) / fps;

% Z-score calcium only for display.
calcium_z = (calcium_ds - mean(calcium_ds)) / std(calcium_ds);
calcium_plot = calcium_z(plot_idx);

% MLspike-prepared calcium for display.
calcium_mlspike_plot = mlspike_result.calcium_mlspike(plot_idx);
calcium_mlspike_plot = (calcium_mlspike_plot - mean(calcium_mlspike_plot)) / std(calcium_mlspike_plot);

ground_truth_plot = ground_truth_events(plot_idx);
predicted_plot = predicted_events(plot_idx);

%% Prepare event markers

% Ground-truth spike-bin events
gt_idx = ground_truth_plot > 0;
gt_times = t(gt_idx);
gt_y = -3 * ones(size(gt_times));

% MLspike predicted events
pred_idx = predicted_plot > 0;
pred_times = t(pred_idx);

%% Figure 1: calcium + predictions

figure('Color', 'w');

plot(t, calcium_plot, 'LineWidth', 1.0);
hold on;

plot(t, calcium_mlspike_plot, 'LineWidth', 1.2);

% Ground-truth events
plot(gt_times, gt_y, ...
    'o', ...
    'MarkerSize', 4, ...
    'LineStyle', 'none');

% MLspike predicted events as vertical dashed lines.
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

title(sprintf('MLspike — neuron %d, %.1f Hz', neuron_idx, fps));

legend({ ...
    'Calcium trace (z-scored)', ...
    'MLspike input trace (display-normalized)', ...
    'Ground-truth spike bins', ...
    'MLspike predictions'}, ...
    'Location', 'best');

grid on;

%% Figure 2: MLspike fit and drift, if available

figure('Color', 'w');

fit_plot = double(mlspike_result.fit(:));
drift_plot = double(mlspike_result.drift(:));

fit_plot = fit_plot(plot_idx);
drift_plot = drift_plot(plot_idx);

% Normalize for display if possible.
if std(fit_plot) > 0
    fit_plot = (fit_plot - mean(fit_plot)) / std(fit_plot);
end

if std(drift_plot) > 0
    drift_plot = (drift_plot - mean(drift_plot)) / std(drift_plot);
end

plot(t, calcium_mlspike_plot, 'LineWidth', 1.0);
hold on;

plot(t, fit_plot, 'LineWidth', 1.2);
plot(t, drift_plot, 'LineWidth', 1.2);

xlabel('Time (s)');
ylabel('Display-normalized signal');

title(sprintf('MLspike fit/drift — neuron %d, %.1f Hz', neuron_idx, fps));

legend({ ...
    'MLspike input trace', ...
    'MLspike fit', ...
    'MLspike drift'}, ...
    'Location', 'best');

grid on;

%% Local helper function

function print_metrics(metrics)
% print_metrics
%
% Prints one evaluation metrics struct in a compact format.

    fprintf('Tolerance: %.3f sec (%d bins)\n', ...
        metrics.tolerance_sec, metrics.tolerance_bins);
    fprintf('True events:      %d\n', metrics.n_true_events);
    fprintf('Predicted events: %d\n', metrics.n_predicted_events);
    fprintf('Matched events:   %d\n', metrics.n_matched_events);
    fprintf('TP: %d | FP: %d | FN: %d\n', ...
        metrics.TP, metrics.FP, metrics.FN);
    fprintf('Precision: %.3f\n', metrics.precision);
    fprintf('Recall:    %.3f\n', metrics.recall);
    fprintf('F1:        %.3f\n', metrics.F1);
    fprintf('Median abs timing error: %.4f sec\n', ...
        metrics.median_abs_timing_error_sec);
    fprintf('Mean signed timing error: %.4f sec\n', ...
        metrics.mean_signed_timing_error_sec);

    if isfield(metrics, 'timing_weighted_recall')
        fprintf('Timing-weighted recall: %.4f\n', ...
            metrics.timing_weighted_recall);
    end
end