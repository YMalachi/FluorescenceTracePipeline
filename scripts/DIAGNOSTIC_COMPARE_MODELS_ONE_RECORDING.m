%% DIAGNOSTIC_COMPARE_MODELS_ONE_RECORDING.m
% Goal:
%   Automatically select and diagnose one interesting model-comparison case.
%
% This script:
%   1. Reads the detailed results CSV
%   2. Automatically selects a diagnostic case
%   3. Loads the matching dataset/recording
%   4. Loads the optimized model parameters
%   5. Runs SimpleBaseline and OASIS
%   6. Prints metrics
%   7. Creates diagnostic plots
%
% Current default:
%   Auto-select a case where OASIS performs much worse than SimpleBaseline
%   using spike-bin F1.

clear; clc; close all;

%% Add project paths

addpath(fullfile('..', 'src'));
addpath(genpath(fullfile('..', 'external', 'OASIS_matlab')));

%% Paths

data_dir = fullfile('..', 'data', 'MockData');
tables_dir = fullfile('..', 'results', 'tables');

% Best parameters chosen from spike-bin optimization.
best_params_file = fullfile(tables_dir, 'best_model_params_by_fps.csv');

% Detailed results file used for automatic case selection.
% IMPORTANT:
% This file should ideally be generated using the same optimized parameters
% that are loaded from best_model_params_by_fps_spike_bins.csv.
detailed_results_file = fullfile(tables_dir, 'model_results_detailed.csv');

if ~isfile(best_params_file)
    error('Best-parameters file not found: %s', best_params_file);
end

if ~isfile(detailed_results_file)
    error('Detailed results file not found: %s', detailed_results_file);
end

%% Auto diagnostic case settings

% Options:
%   "OASIS_worse_than_baseline"
%   "OASIS_better_than_baseline"
%   "worst_OASIS"
%   "worst_SimpleBaseline"
%   "best_OASIS"
%   "best_SimpleBaseline"
auto_case_type = "OASIS_worse_than_baseline";

% We currently care mainly about spike-bin evaluation.
auto_evaluation_mode = "spike_bins";

% Restrict case search to one FPS if wanted.
% Use NaN to search all FPS values.
auto_target_fps = NaN;

%% Plot settings

plot_start_sec = 0;
plot_duration_sec = 30;

%% Automatically select diagnostic case

diagnostic_case = select_diagnostic_case( ...
    detailed_results_file, ...
    auto_case_type, ...
    auto_evaluation_mode, ...
    auto_target_fps);

dataset_id = diagnostic_case.dataset_id;
recording_idx = diagnostic_case.recording_idx;
target_fps = diagnostic_case.target_fps;

fprintf('\nAuto-selected diagnostic case:\n');
fprintf('Case type: %s\n', auto_case_type);
fprintf('Evaluation mode: %s\n', auto_evaluation_mode);
fprintf('Dataset ID: %d\n', dataset_id);
fprintf('Recording index: %d\n', recording_idx);
fprintf('Target FPS: %.1f\n', target_fps);
fprintf('Baseline F1: %.3f\n', diagnostic_case.baseline_F1);
fprintf('OASIS F1: %.3f\n', diagnostic_case.oasis_F1);
fprintf('OASIS - Baseline F1 difference: %.3f\n\n', diagnostic_case.F1_difference);

%% Load best parameters table

best_params_table = readtable(best_params_file);

% Convert text columns to string for reliable comparisons.
text_columns = {'model', 'optimization_mode', 'ar_model', 'method'};

for c = 1:length(text_columns)
    col = text_columns{c};

    if ismember(col, best_params_table.Properties.VariableNames)
        best_params_table.(col) = string(best_params_table.(col));
    end
end

%% Load selected dataset

dataset_name = sprintf('data.%d.train.preprocessed.mat', dataset_id);
dataset_file = fullfile(data_dir, dataset_name);

if ~isfile(dataset_file)
    error('Dataset file not found: %s', dataset_file);
end

raw = load(dataset_file);

if ~isfield(raw, 'data')
    error('File %s does not contain variable named "data".', dataset_name);
end

data = raw.data;

if recording_idx < 1 || recording_idx > numel(data)
    error('recording_idx %d is out of range. This file has %d recordings.', ...
        recording_idx, numel(data));
end

rec = data{recording_idx};

calcium = double(rec.calcium(:));
spikes = double(rec.spikes(:));
original_fps = double(rec.fps);

if isfield(rec, 'cell_num')
    cell_num = double(rec.cell_num);
else
    cell_num = NaN;
end

fprintf('Diagnostic recording:\n');
fprintf('Dataset: %s\n', dataset_name);
fprintf('Dataset ID: %d\n', dataset_id);
fprintf('Recording index: %d\n', recording_idx);
fprintf('Cell number: %g\n', cell_num);
fprintf('Original FPS: %.4f\n', original_fps);
fprintf('Target FPS: %.1f\n\n', target_fps);

%% Downsample recording

ds = downsample_calcium_and_spikes( ...
    calcium, ...
    spikes, ...
    original_fps, ...
    target_fps);

calcium_ds = ds.calcium;
spikes_ds = ds.spikes;
fps = ds.actual_fps;

fprintf('Downsampling summary:\n');
fprintf('Actual FPS: %.4f\n', fps);
fprintf('Downsample factor: %d\n', ds.downsample_factor);
fprintf('Total spike count: %.0f\n', ds.downsampled_spike_count);
fprintf('Active spike bins: %.0f\n\n', ds.downsampled_active_spike_bins);

%% Load model parameters for this FPS

baseline_params = get_baseline_params(best_params_table, target_fps);
oasis_params = get_oasis_params(best_params_table, target_fps);

fprintf('Loaded SimpleBaseline parameters:\n');
disp(baseline_params);

fprintf('Loaded OASIS parameters:\n');
disp(oasis_params);

%% Run models

fprintf('\nRunning SimpleBaseline...\n');
baseline_result = run_simple_baseline(calcium_ds, fps, baseline_params);
fprintf('SimpleBaseline predicted events: %d\n', baseline_result.n_predicted_events);

fprintf('\nRunning OASIS...\n');
oasis_result = run_oasis_model(calcium_ds, fps, oasis_params);
fprintf('OASIS predicted events: %d\n', oasis_result.n_predicted_events);

%% Evaluate both models

tolerance_sec = 0.10;
burst_gap_sec = 0.10;

baseline_spike_metrics = evaluate_event_prediction( ...
    baseline_result.predicted_events, ...
    spikes_ds, ...
    fps, ...
    tolerance_sec, ...
    'spike_bins', ...
    burst_gap_sec);

baseline_burst_metrics = evaluate_event_prediction( ...
    baseline_result.predicted_events, ...
    spikes_ds, ...
    fps, ...
    tolerance_sec, ...
    'burst_onsets', ...
    burst_gap_sec);

oasis_spike_metrics = evaluate_event_prediction( ...
    oasis_result.predicted_events, ...
    spikes_ds, ...
    fps, ...
    tolerance_sec, ...
    'spike_bins', ...
    burst_gap_sec);

oasis_burst_metrics = evaluate_event_prediction( ...
    oasis_result.predicted_events, ...
    spikes_ds, ...
    fps, ...
    tolerance_sec, ...
    'burst_onsets', ...
    burst_gap_sec);

%% Print metrics

fprintf('\n============================================================\n');
fprintf('DIAGNOSTIC METRICS\n');
fprintf('============================================================\n');

print_metrics('SimpleBaseline', 'spike_bins', baseline_spike_metrics);
print_metrics('SimpleBaseline', 'burst_onsets', baseline_burst_metrics);
print_metrics('OASIS', 'spike_bins', oasis_spike_metrics);
print_metrics('OASIS', 'burst_onsets', oasis_burst_metrics);

%% Prepare plotting window

n_total_samples = length(calcium_ds);

start_sample = max(1, round(plot_start_sec * fps) + 1);
end_sample = min(n_total_samples, round((plot_start_sec + plot_duration_sec) * fps));

plot_idx = start_sample:end_sample;
t = (plot_idx - 1) / fps;

calcium_z = (calcium_ds - mean(calcium_ds)) / std(calcium_ds);
calcium_plot = calcium_z(plot_idx);

ground_truth_events = spikes_ds > 0;
ground_truth_plot = ground_truth_events(plot_idx);

% Burst-onset ground truth for display.
burst_metrics_for_plot = evaluate_event_prediction( ...
    false(size(spikes_ds)), ...
    spikes_ds, ...
    fps, ...
    tolerance_sec, ...
    'burst_onsets', ...
    burst_gap_sec);

burst_onsets = burst_metrics_for_plot.true_events;
burst_onsets_plot = burst_onsets(plot_idx);

%% Figure 1: SimpleBaseline diagnostic

figure('Color', 'w');

subplot(2, 1, 1);

plot(t, calcium_plot, 'LineWidth', 1.0);
hold on;

plot_event_markers(t, ground_truth_plot, -3, 'o');
plot_event_markers(t, burst_onsets_plot, -3.6, '^');

xlabel('Time (s)');
ylabel('Z-scored calcium');
title(sprintf('SimpleBaseline diagnostic — Dataset %d, Recording %d, %.1f Hz', ...
    dataset_id, recording_idx, fps));
legend({'Calcium trace', 'Spike bins', 'Burst onsets'}, 'Location', 'best');
grid on;

subplot(2, 1, 2);

baseline_score_plot = baseline_result.event_score(plot_idx);
baseline_pred_plot = baseline_result.predicted_events(plot_idx);

plot(t, baseline_score_plot, 'LineWidth', 1.0);
hold on;

yl = ylim;

baseline_pred_times = t(baseline_pred_plot > 0);

for i = 1:length(baseline_pred_times)
    xline(baseline_pred_times(i), '--', ...
        'LineWidth', 1.0, ...
        'HandleVisibility', 'off');
end

ylim(yl);

xlabel('Time (s)');
ylabel('Baseline event score');
title('SimpleBaseline event score and predictions');
legend({'Event score', 'Predicted events'}, 'Location', 'best');
grid on;

%% Figure 2: OASIS diagnostic

figure('Color', 'w');

subplot(2, 1, 1);

plot(t, calcium_plot, 'LineWidth', 1.0);
hold on;

% Normalize OASIS calcium estimate only for plotting.
c_oasis_plot = oasis_result.c_oasis(plot_idx);

if std(c_oasis_plot) > 0
    c_oasis_plot = (c_oasis_plot - mean(c_oasis_plot)) / std(c_oasis_plot);
end

plot(t, c_oasis_plot, 'LineWidth', 1.2);

plot_event_markers(t, ground_truth_plot, -3, 'o');
plot_event_markers(t, burst_onsets_plot, -3.6, '^');

xlabel('Time (s)');
ylabel('Z-scored calcium');
title(sprintf('OASIS diagnostic — Dataset %d, Recording %d, %.1f Hz', ...
    dataset_id, recording_idx, fps));
legend({'Calcium trace', 'OASIS calcium estimate', 'Spike bins', 'Burst onsets'}, ...
    'Location', 'best');
grid on;

subplot(2, 1, 2);

s_oasis_z_plot = oasis_result.s_oasis_z(plot_idx);
oasis_pred_plot = oasis_result.predicted_events(plot_idx);

plot(t, s_oasis_z_plot, 'LineWidth', 1.0);
hold on;

yl = ylim;

oasis_pred_times = t(oasis_pred_plot > 0);

for i = 1:length(oasis_pred_times)
    xline(oasis_pred_times(i), '--', ...
        'LineWidth', 1.0, ...
        'HandleVisibility', 'off');
end

ylim(yl);

xlabel('Time (s)');
ylabel('OASIS activity z-score');
title('OASIS inferred activity and predictions');
legend({'OASIS activity z-score', 'Predicted events'}, 'Location', 'best');
grid on;

%% Window summary

fprintf('\nWindow summary:\n');
fprintf('Window: %.2f to %.2f sec\n', t(1), t(end));
fprintf('Spike-bin events in window: %d\n', sum(ground_truth_plot));
fprintf('Burst onsets in window: %d\n', sum(burst_onsets_plot));
fprintf('SimpleBaseline predictions in window: %d\n', sum(baseline_pred_plot));
fprintf('OASIS predictions in window: %d\n', sum(oasis_pred_plot));

%% Local helper functions

function diagnostic_case = select_diagnostic_case(detailed_results_file, case_type, evaluation_mode, target_fps_filter)
% select_diagnostic_case
%
% Selects an interesting recording from the detailed results table.
%
% The function compares SimpleBaseline and OASIS on the same:
%   dataset_id, recording_idx, target_fps, evaluation_mode.

    results = readtable(detailed_results_file);

    % Convert text columns to string.
    text_columns = {'model', 'evaluation_mode', 'status', 'dataset_name', 'error_message'};

    for c = 1:length(text_columns)
        col = text_columns{c};

        if ismember(col, results.Properties.VariableNames)
            results.(col) = string(results.(col));
        end
    end

    % Keep successful rows and selected evaluation mode.
    if ismember('status', results.Properties.VariableNames)
        results = results(results.status == "ok", :);
    end

    results = results(results.evaluation_mode == evaluation_mode, :);

    if ~isnan(target_fps_filter)
        results = results(results.target_fps == target_fps_filter, :);
    end

    baseline_rows = results(results.model == "SimpleBaseline", :);
    oasis_rows = results(results.model == "OASIS", :);

    if isempty(baseline_rows) || isempty(oasis_rows)
        error('Could not find both SimpleBaseline and OASIS rows.');
    end

    comparison_table = table();

    for i = 1:height(baseline_rows)

        b = baseline_rows(i, :);

        matching_oasis = oasis_rows( ...
            oasis_rows.dataset_id == b.dataset_id & ...
            oasis_rows.recording_idx == b.recording_idx & ...
            oasis_rows.target_fps == b.target_fps, :);

        if isempty(matching_oasis)
            continue;
        end

        o = matching_oasis(1, :);

        row = table( ...
            b.dataset_id, ...
            b.recording_idx, ...
            b.target_fps, ...
            b.F1, ...
            o.F1, ...
            o.F1 - b.F1, ...
            b.timing_weighted_recall, ...
            o.timing_weighted_recall, ...
            'VariableNames', { ...
                'dataset_id', ...
                'recording_idx', ...
                'target_fps', ...
                'baseline_F1', ...
                'oasis_F1', ...
                'F1_difference', ...
                'baseline_timing_weighted_recall', ...
                'oasis_timing_weighted_recall'});

        comparison_table = [comparison_table; row];
    end

    if isempty(comparison_table)
        error('No matched Baseline/OASIS cases found.');
    end

    switch case_type

        case "OASIS_worse_than_baseline"

            comparison_table = sortrows(comparison_table, 'F1_difference', 'ascend');

        case "OASIS_better_than_baseline"

            comparison_table = sortrows(comparison_table, 'F1_difference', 'descend');

        case "worst_OASIS"

            comparison_table = sortrows(comparison_table, 'oasis_F1', 'ascend');

        case "worst_SimpleBaseline"

            comparison_table = sortrows(comparison_table, 'baseline_F1', 'ascend');

        case "best_OASIS"

            comparison_table = sortrows(comparison_table, 'oasis_F1', 'descend');

        case "best_SimpleBaseline"

            comparison_table = sortrows(comparison_table, 'baseline_F1', 'descend');

        otherwise

            error('Unknown auto_case_type: %s', case_type);
    end

    diagnostic_case = comparison_table(1, :);
end

function baseline_params = get_baseline_params(best_params_table, target_fps)
% get_baseline_params
%
% Extracts SimpleBaseline parameters for one target FPS.

    rows = best_params_table( ...
        best_params_table.model == "SimpleBaseline" & ...
        best_params_table.target_fps == target_fps, :);

    if isempty(rows)
        error('No SimpleBaseline parameters found for target FPS %.1f.', target_fps);
    end

    row = rows(1, :);

    baseline_params = struct();
    baseline_params.smoothing_window_sec = row.smoothing_window_sec;
    baseline_params.derivative_weight = row.derivative_weight;
    baseline_params.amplitude_weight = row.amplitude_weight;
    baseline_params.event_score_threshold = row.event_score_threshold;
    baseline_params.min_event_distance_sec = row.baseline_min_event_distance_sec;
end

function oasis_params = get_oasis_params(best_params_table, target_fps)
% get_oasis_params
%
% Extracts OASIS parameters for one target FPS.

    rows = best_params_table( ...
        best_params_table.model == "OASIS" & ...
        best_params_table.target_fps == target_fps, :);

    if isempty(rows)
        error('No OASIS parameters found for target FPS %.1f.', target_fps);
    end

    row = rows(1, :);

    oasis_params = struct();
    oasis_params.ar_model = char(row.ar_model);
    oasis_params.method = char(row.method);
    oasis_params.optimize_b = logical(row.optimize_b);
    oasis_params.optimize_pars = logical(row.optimize_pars);
    oasis_params.oasis_threshold_z = row.oasis_threshold_z;
    oasis_params.min_event_distance_sec = row.oasis_min_event_distance_sec;
end

function print_metrics(model_name, mode_name, metrics)
% print_metrics
%
% Prints one metric block in a compact format.

    fprintf('\n%s | %s\n', model_name, mode_name);
    fprintf('True events:      %d\n', metrics.n_true_events);
    fprintf('Predicted events: %d\n', metrics.n_predicted_events);
    fprintf('Matched events:   %d\n', metrics.n_matched_events);
    fprintf('TP: %d | FP: %d | FN: %d\n', metrics.TP, metrics.FP, metrics.FN);
    fprintf('Precision: %.3f\n', metrics.precision);
    fprintf('Recall:    %.3f\n', metrics.recall);
    fprintf('F1:        %.3f\n', metrics.F1);
    fprintf('Median abs timing error: %.4f sec\n', ...
        metrics.median_abs_timing_error_sec);
    fprintf('Mean signed timing error: %.4f sec\n', ...
        metrics.mean_signed_timing_error_sec);
    fprintf('Timing-weighted recall: %.4f\n', ...
        metrics.timing_weighted_recall);
end

function plot_event_markers(t, event_vector, y_value, marker_style)
% plot_event_markers
%
% Plots event markers at a fixed y-position.

    event_times = t(event_vector > 0);
    event_y = y_value * ones(size(event_times));

    plot(event_times, event_y, ...
        marker_style, ...
        'MarkerSize', 4, ...
        'LineStyle', 'none');
end