%% 
%% RUN_OASIS_ON_SIGNAL.m
% Goal:
%   Run OASIS spike inference on one calcium fluorescence trace.
%
% Input:
%   calcium - fluorescence trace vector
%   fps     - sampling rate in Hz
%
% Output:
%   Saves:
%       results/oasis_platform/oasis_predicted_events.csv
%       results/oasis_platform/oasis_output.mat
%       results/oasis_platform/oasis_plot.png
%
% Notes:
%   This script is designed for future use with calcium traces extracted
%   from videos.
%
%   For now, it can run in two modes:
%       1. Demo mode using mock data
%       2. User signal mode where you manually provide calcium and fps

clear; clc; close all;

%% Add paths

addpath(fullfile('..', 'src'));
addpath(genpath(fullfile('..', 'external')));

%% Output folder

output_dir = fullfile('..', 'results', 'oasis_platform');

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

%% Choose input mode

% Options:
%   "mock_data"     - load one trace from the existing MockData files
%   "manual_signal" - use calcium and fps variables defined manually below
input_mode = "mock_data";

%% Load / define calcium signal

switch input_mode

    case "mock_data"

        % Demo using the same data format we used until now.
        dataset_file = fullfile('..', 'data', 'MockData', 'data.1.train.preprocessed.mat');
        neuron_idx = 1;
        target_fps = 10;

        raw = load(dataset_file);
        data = raw.data;

        rec = data{neuron_idx};

        calcium_raw = double(rec.calcium(:));
        spikes_raw = double(rec.spikes(:));
        original_fps = double(rec.fps);

        % Downsample to desired FPS.
        ds = downsample_calcium_and_spikes( ...
            calcium_raw, ...
            spikes_raw, ...
            original_fps, ...
            target_fps);

        calcium = ds.calcium;
        fps = ds.actual_fps;

        % Ground truth exists only in mock data.
        has_ground_truth = true;
        ground_truth_spikes = ds.spikes;

        signal_name = sprintf('mock_dataset1_neuron%d_%.0fHz', neuron_idx, target_fps);

    case "manual_signal"

        % Use this mode later for traces extracted from videos.
        %
        % Expected:
        %   calcium = column vector or row vector
        %   fps = sampling rate in Hz
        %
        % Example:
        %   calcium = your_extracted_trace;
        %   fps = 20;

        calcium = [];  % Replace this with your extracted signal
        fps = [];      % Replace this with your sampling rate

        if isempty(calcium)
            error('Manual signal mode selected, but calcium is empty.');
        end

        if isempty(fps)
            error('Manual signal mode selected, but fps is empty.');
        end

        calcium = double(calcium(:));
        fps = double(fps);

        has_ground_truth = false;
        ground_truth_spikes = [];

        signal_name = 'manual_signal';

    otherwise

        error('Unknown input_mode: %s', input_mode);
end

%% Basic checks

if isempty(calcium)
    error('Calcium signal is empty.');
end

if any(isnan(calcium))
    error('Calcium signal contains NaN values.');
end

if any(isinf(calcium))
    error('Calcium signal contains Inf values.');
end

if isempty(fps) || ~isscalar(fps) || fps <= 0
    error('fps must be a positive scalar.');
end

fprintf('Signal loaded successfully.\n');
fprintf('Signal name: %s\n', signal_name);
fprintf('Number of samples: %d\n', length(calcium));
fprintf('FPS: %.4f Hz\n', fps);
fprintf('Duration: %.2f sec\n\n', length(calcium) / fps);

%% OASIS parameters

% These are default starting parameters.
% Replace oasis_threshold_z and min_event_distance_sec with optimized values
% if you have them from best_oasis_params_from_log.csv.

oasis_params = struct();

oasis_params.ar_model = 'ar1';
oasis_params.method = 'foopsi';
oasis_params.optimize_b = true;
oasis_params.optimize_pars = true;

% Event conversion parameters.
% These are the main parameters you will tune/use from optimization.
oasis_params.oasis_threshold_z = 0.75;
oasis_params.min_event_distance_sec = 0.03;

fprintf('OASIS parameters:\n');
fprintf('  ar_model: %s\n', oasis_params.ar_model);
fprintf('  method: %s\n', oasis_params.method);
fprintf('  optimize_b: %d\n', oasis_params.optimize_b);
fprintf('  optimize_pars: %d\n', oasis_params.optimize_pars);
fprintf('  oasis_threshold_z: %.3f\n', oasis_params.oasis_threshold_z);
fprintf('  min_event_distance_sec: %.3f\n\n', oasis_params.min_event_distance_sec);

%% Run OASIS

fprintf('Running OASIS...\n');

oasis_result = run_oasis_model( ...
    calcium, ...
    fps, ...
    oasis_params);

fprintf('OASIS finished.\n');
fprintf('Predicted events: %d\n\n', oasis_result.n_predicted_events);

%% Extract predicted events and event times

predicted_events = logical(oasis_result.predicted_events(:));
predicted_event_indices = find(predicted_events);
predicted_event_times_sec = (predicted_event_indices - 1) / fps;

time_sec = ((1:length(calcium))' - 1) / fps;

%% Optional evaluation if ground truth exists

if has_ground_truth

    tolerance_sec = 0.10;
    burst_gap_sec = 0.10;

    metrics_burst = evaluate_event_prediction( ...
        predicted_events, ...
        ground_truth_spikes, ...
        fps, ...
        tolerance_sec, ...
        'burst_onsets', ...
        burst_gap_sec);

    metrics_spike_bins = evaluate_event_prediction( ...
        predicted_events, ...
        ground_truth_spikes, ...
        fps, ...
        tolerance_sec, ...
        'spike_bins', ...
        burst_gap_sec);

    fprintf('Evaluation against mock-data ground truth:\n');
    fprintf('\nBurst-onset mode:\n');
    print_metrics(metrics_burst);

    fprintf('\nSpike-bin mode:\n');
    print_metrics(metrics_spike_bins);

else

    metrics_burst = [];
    metrics_spike_bins = [];
end

%% Create output table

output_table = table( ...
    time_sec, ...
    calcium(:), ...
    predicted_events, ...
    'VariableNames', { ...
        'time_sec', ...
        'calcium', ...
        'oasis_predicted_event'});

% Add OASIS continuous output if available.
if isfield(oasis_result, 's_oasis_z')
    output_table.oasis_activity_z = double(oasis_result.s_oasis_z(:));
end

if isfield(oasis_result, 's_oasis')
    output_table.oasis_activity_raw = double(oasis_result.s_oasis(:));
end

%% Save CSV and MAT outputs

csv_output_file = fullfile(output_dir, 'oasis_predicted_events.csv');
mat_output_file = fullfile(output_dir, 'oasis_output.mat');

writetable(output_table, csv_output_file);

save(mat_output_file, ...
    'calcium', ...
    'fps', ...
    'time_sec', ...
    'oasis_params', ...
    'oasis_result', ...
    'predicted_events', ...
    'predicted_event_indices', ...
    'predicted_event_times_sec', ...
    'metrics_burst', ...
    'metrics_spike_bins');

fprintf('\nSaved OASIS CSV output to:\n%s\n', csv_output_file);
fprintf('Saved OASIS MAT output to:\n%s\n', mat_output_file);

%% Plot result - improved version

calcium_z = zscore_safe(calcium);

% Make sure OASIS activity exists and is column vector.
if isfield(oasis_result, 's_oasis_z')
    oasis_activity_z = double(oasis_result.s_oasis_z(:));
else
    oasis_activity_z = [];
end

%% Figure 1: clean overview without vertical event lines

figure('Color', 'w');

plot(time_sec, calcium_z, 'LineWidth', 1.0);
hold on;

if ~isempty(oasis_activity_z)
    plot(time_sec, oasis_activity_z, 'LineWidth', 1.0);
end

xlabel('Time (s)');
ylabel('Z-scored signal');

title(sprintf('OASIS overview — %s', signal_name), ...
    'Interpreter', 'none');

if ~isempty(oasis_activity_z)
    legend({'Calcium trace', 'OASIS activity z-score'}, ...
        'Location', 'best');
else
    legend({'Calcium trace'}, ...
        'Location', 'best');
end

grid on;

overview_plot_file = fullfile(output_dir, 'oasis_overview_plot.png');
saveas(gcf, overview_plot_file);

fprintf('Saved OASIS overview plot to:\n%s\n', overview_plot_file);

%% Figure 2: zoomed event plot

% Choose a shorter window so predicted events are readable.
zoom_start_sec = 0;
zoom_duration_sec = 60;

zoom_end_sec = zoom_start_sec + zoom_duration_sec;

zoom_idx = time_sec >= zoom_start_sec & time_sec <= zoom_end_sec;
t_zoom = time_sec(zoom_idx);

calcium_zoom = calcium_z(zoom_idx);

if ~isempty(oasis_activity_z)
    oasis_zoom = oasis_activity_z(zoom_idx);
end

predicted_events_zoom = predicted_events(zoom_idx);
predicted_event_times_zoom = t_zoom(predicted_events_zoom);

figure('Color', 'w');

plot(t_zoom, calcium_zoom, 'LineWidth', 1.0);
hold on;

if ~isempty(oasis_activity_z)
    plot(t_zoom, oasis_zoom, 'LineWidth', 1.0);
end

% Predicted events: draw only in the zoom window.
yl = ylim;

for i = 1:length(predicted_event_times_zoom)
    xline(predicted_event_times_zoom(i), '--', ...
        'LineWidth', 0.8, ...
        'HandleVisibility', 'off');
end

ylim(yl);

% Ground truth, only if available.
% Display as small markers near the bottom instead of full-height lines.
if has_ground_truth

    ground_truth_events = ground_truth_spikes > 0;
    ground_truth_zoom = ground_truth_events(zoom_idx);
    gt_times_zoom = t_zoom(ground_truth_zoom);

    y_gt = yl(1) + 0.10 * (yl(2) - yl(1));

    plot(gt_times_zoom, y_gt * ones(size(gt_times_zoom)), ...
        'o', ...
        'MarkerSize', 4, ...
        'LineStyle', 'none');
end

xlabel('Time (s)');
ylabel('Z-scored signal');

title(sprintf('OASIS event zoom — %s | %.0f–%.0f sec', ...
    signal_name, zoom_start_sec, zoom_end_sec), ...
    'Interpreter', 'none');

if has_ground_truth && ~isempty(oasis_activity_z)
    legend({'Calcium trace', 'OASIS activity z-score', 'Ground truth spike bins'}, ...
        'Location', 'best');
elseif ~isempty(oasis_activity_z)
    legend({'Calcium trace', 'OASIS activity z-score'}, ...
        'Location', 'best');
else
    legend({'Calcium trace'}, ...
        'Location', 'best');
end

grid on;

zoom_plot_file = fullfile(output_dir, 'oasis_event_zoom_plot.png');
saveas(gcf, zoom_plot_file);

fprintf('Saved OASIS zoom plot to:\n%s\n', zoom_plot_file);

%% Figure 3: event raster only

figure('Color', 'w');

hold on;

% Predicted events as raster at y = 1
plot(predicted_event_times_sec, ...
    ones(size(predicted_event_times_sec)), ...
    '|', ...
    'MarkerSize', 8, ...
    'LineStyle', 'none');

if has_ground_truth

    gt_event_indices = find(ground_truth_spikes > 0);
    gt_event_times_sec = (gt_event_indices - 1) / fps;

    % Ground truth events as raster at y = 2
    plot(gt_event_times_sec, ...
        2 * ones(size(gt_event_times_sec)), ...
        '|', ...
        'MarkerSize', 8, ...
        'LineStyle', 'none');

    yticks([1 2]);
    yticklabels({'OASIS predicted', 'Ground truth'});
else
    yticks(1);
    yticklabels({'OASIS predicted'});
end

xlabel('Time (s)');
title(sprintf('OASIS event raster — %s', signal_name), ...
    'Interpreter', 'none');

ylim([0.5 2.5]);
grid on;

raster_plot_file = fullfile(output_dir, 'oasis_event_raster_plot.png');
saveas(gcf, raster_plot_file);

fprintf('Saved OASIS event raster plot to:\n%s\n', raster_plot_file);
%% Local helper functions

function x_z = zscore_safe(x)
% zscore_safe
%
% Computes z-score safely.

    x = double(x(:));

    if std(x) == 0
        x_z = zeros(size(x));
    else
        x_z = (x - mean(x)) / std(x);
    end
end

function print_metrics(metrics)
% print_metrics
%
% Prints evaluation metrics compactly.

    fprintf('  True events:      %d\n', metrics.n_true_events);
    fprintf('  Predicted events: %d\n', metrics.n_predicted_events);
    fprintf('  Matched events:   %d\n', metrics.n_matched_events);
    fprintf('  Precision: %.3f\n', metrics.precision);
    fprintf('  Recall:    %.3f\n', metrics.recall);
    fprintf('  F1:        %.3f\n', metrics.F1);

    if isfield(metrics, 'median_abs_timing_error_sec')
        fprintf('  Median abs timing error: %.4f sec\n', ...
            metrics.median_abs_timing_error_sec);
    end

    if isfield(metrics, 'timing_weighted_recall')
        fprintf('  Timing-weighted recall: %.4f\n', ...
            metrics.timing_weighted_recall);
    end
end