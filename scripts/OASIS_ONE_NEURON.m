%% OASIS_ONE_NEURON.m
% Goal:
%   Run OASIS deconvolution on one calcium trace.
%
% Current scope:
%   - One neuron
%   - One sampling rate
%   - OASIS only
%
% Model idea:
%   OASIS estimates a denoised calcium trace and a deconvolved
%   spike-like activity signal. We threshold the inferred activity signal
%   to create predicted event times.
%
% Important:
%   This script uses run_oasis_model(), which wraps the OASIS logic.
%   Hyperparameters are stored in oasis_params so we can optimize them later.

clear; clc; close all;

%% Add project paths

% Project source functions:
%   downsample_calcium_and_spikes()
%   evaluate_event_prediction()
%   run_oasis_model()
addpath(fullfile('..', 'src'));

% Add OASIS and all its subfolders.
addpath(genpath(fullfile('..', 'external', 'OASIS_matlab')));

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

%% Ground-truth event vector

% A ground-truth event is any bin that contains at least one spike.
ground_truth_events = spikes_ds > 0;

%% OASIS parameters

oasis_params = struct();

% OASIS internal deconvolution settings
oasis_params.ar_model = 'ar1';
oasis_params.method = 'foopsi';
oasis_params.optimize_b = true;
oasis_params.optimize_pars = true;

% Event-conversion settings
% These are not final. We will optimize them later.
oasis_params.oasis_threshold_z = 1.5;
oasis_params.min_event_distance_sec = 0.10;

%% Run OASIS model

fprintf('Running OASIS model...\n');

oasis_result = run_oasis_model(calcium_ds, fps, oasis_params);

predicted_events = oasis_result.predicted_events;
c_oasis = oasis_result.c_oasis;
s_oasis = oasis_result.s_oasis;
s_oasis_z = oasis_result.s_oasis_z;

fprintf('OASIS finished.\n');
fprintf('OASIS calcium estimate length: %d\n', length(c_oasis));
fprintf('OASIS activity estimate length: %d\n\n', length(s_oasis));

fprintf('OASIS parameters:\n');
fprintf('AR model: %s\n', oasis_params.ar_model);
fprintf('Method: %s\n', oasis_params.method);
fprintf('Optimize baseline: %d\n', oasis_params.optimize_b);
fprintf('Optimize parameters: %d\n', oasis_params.optimize_pars);
fprintf('OASIS threshold: %.2f z-score\n', oasis_params.oasis_threshold_z);
fprintf('Minimum event distance: %.3f sec (%d samples)\n\n', ...
    oasis_params.min_event_distance_sec, oasis_result.min_event_distance_samples);

fprintf('Number of OASIS predicted events: %d\n', oasis_result.n_predicted_events);
fprintf('Number of ground-truth active spike bins: %d\n', sum(ground_truth_events));

%% Evaluate OASIS predictions

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

plot_duration_sec = 30;

n_plot_samples = min(round(plot_duration_sec * fps), length(calcium_ds));

t = (0:n_plot_samples-1) / fps;

% Z-score calcium only for display.
calcium_z = (calcium_ds - mean(calcium_ds)) / std(calcium_ds);

calcium_plot = calcium_z(1:n_plot_samples);
c_oasis_plot = c_oasis(1:n_plot_samples);
ground_truth_plot = ground_truth_events(1:n_plot_samples);
predicted_plot = predicted_events(1:n_plot_samples);

% Normalize OASIS calcium estimate for display.
if std(c_oasis_plot) > 0
    c_oasis_plot = (c_oasis_plot - mean(c_oasis_plot)) / std(c_oasis_plot);
end

%% Prepare event markers

% Ground-truth spike/event bins
gt_idx = ground_truth_plot > 0;
gt_times = t(gt_idx);
gt_y = -3 * ones(size(gt_times));

% OASIS predicted events
pred_idx = predicted_plot > 0;
pred_times = t(pred_idx);

%% Plot

figure('Color', 'w');

plot(t, calcium_plot, 'LineWidth', 1.0);
hold on;

plot(t, c_oasis_plot, 'LineWidth', 1.5);

% Ground-truth events
plot(gt_times, gt_y, ...
    'o', ...
    'MarkerSize', 4, ...
    'LineStyle', 'none');

% Predicted OASIS events as vertical dashed lines.
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

title(sprintf('OASIS — neuron %d, %.1f Hz', neuron_idx, fps));

legend({ ...
    'Calcium trace (z-scored)', ...
    'OASIS calcium estimate', ...
    'Ground-truth spike bins', ...
    'OASIS predicted events'}, ...
    'Location', 'best');

grid on;