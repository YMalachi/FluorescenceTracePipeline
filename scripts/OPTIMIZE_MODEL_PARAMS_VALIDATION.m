%% OPTIMIZE_MODEL_PARAMS_VALIDATION.m
% Goal:
%   Optimize model hyperparameters on the validation split.
%
% Current models:
%   1. SimpleBaseline
%   2. OASIS
%
% Optimization is done separately for each sampling rate:
%   100, 50, 20, 10 Hz
%
% Main objective:
%   Maximize mean burst-onset F1 across validation recordings.
%
% Secondary metrics saved:
%   - timing-weighted recall
%   - precision
%   - recall
%   - median absolute timing error
%
% Outputs:
%   results/tables/hyperparameter_search_results.csv
%   results/tables/best_model_params_by_fps.csv

clear; clc; close all;

%% Add project paths

addpath(fullfile('..', 'src'));
addpath(genpath(fullfile('..', 'external', 'OASIS_matlab')));

%% Paths

data_dir = fullfile('..', 'data', 'MockData');
tables_dir = fullfile('..', 'results', 'tables');

split_file = fullfile(tables_dir, 'data_split.csv');

if ~isfile(split_file)
    error('Split file not found: %s', split_file);
end

%% Load split table

split_table = readtable(split_file);

% Convert text columns to string for reliable comparisons.
text_columns = {'dataset_name', 'unit_id', 'unit_type', 'split'};

for c = 1:length(text_columns)
    col = text_columns{c};

    if ismember(col, split_table.Properties.VariableNames)
        split_table.(col) = string(split_table.(col));
    end
end

validation_rows = split_table(split_table.split == "validation", :);

fprintf('Loaded split table:\n%s\n', split_file);
fprintf('Validation recordings: %d\n', height(validation_rows));
fprintf('Validation unique units: %d\n\n', length(unique(validation_rows.unit_id)));

%% Optimization settings

target_fps_list = [100, 50, 20, 10];

% We optimize using spike-bin mode.
% In this mode, every bin with at least one electrophysiological spike
% is treated as a true event.
optimization_mode = 'spike_bins';

% Evaluation settings.
tolerance_sec = 0.10;
burst_gap_sec = 0.10;   % only used when mode = 'burst_onsets'

%% Hyperparameter grids

% SimpleBaseline grid.
% This is reasonably broad but still manageable.
baseline_grid = create_baseline_grid();

% OASIS grid.
% For now, we optimize only the event-conversion parameters.
% Internal OASIS settings remain fixed.
oasis_grid = create_oasis_grid();

fprintf('Baseline parameter combinations: %d\n', height(baseline_grid));
fprintf('OASIS parameter combinations: %d\n\n', height(oasis_grid));

%% Prepare output table

search_results = table();

%% Main optimization loop

for fps_idx = 1:length(target_fps_list)

    target_fps = target_fps_list(fps_idx);

    fprintf('\n============================================================\n');
    fprintf('Optimizing models at %.1f Hz\n', target_fps);
    fprintf('============================================================\n');

    %% Load and downsample validation recordings once for this FPS

    fprintf('Preparing validation recordings...\n');

    validation_data = prepare_validation_data( ...
        validation_rows, ...
        data_dir, ...
        target_fps);

    fprintf('Prepared %d validation recordings.\n\n', length(validation_data));

    %% Optimize SimpleBaseline

    fprintf('Optimizing SimpleBaseline...\n');

    for p = 1:height(baseline_grid)

        params = struct();

        params.smoothing_window_sec = baseline_grid.smoothing_window_sec(p);
        params.derivative_weight = baseline_grid.derivative_weight(p);
        params.amplitude_weight = baseline_grid.amplitude_weight(p);
        params.event_score_threshold = baseline_grid.event_score_threshold(p);
        params.min_event_distance_sec = baseline_grid.min_event_distance_sec(p);

        metrics_list = [];

        for r = 1:length(validation_data)

            rec_data = validation_data(r);

            model_result = run_simple_baseline( ...
                rec_data.calcium_ds, ...
                rec_data.actual_fps, ...
                params);

            metrics = evaluate_event_prediction( ...
                model_result.predicted_events, ...
                rec_data.spikes_ds, ...
                rec_data.actual_fps, ...
                tolerance_sec, ...
                optimization_mode, ...
                burst_gap_sec);

            metrics_list = [metrics_list; metrics];
        end

        row = summarize_param_result( ...
            "SimpleBaseline", ...
            target_fps, ...
            optimization_mode, ...
            params, ...
            metrics_list, ...
            length(validation_data));

        search_results = [search_results; row];

        if mod(p, 100) == 0
            fprintf('  SimpleBaseline progress: %d/%d\n', p, height(baseline_grid));
        end
    end

    %% Optimize OASIS

    fprintf('\nOptimizing OASIS...\n');

    % OASIS internal parameters are fixed for now.
    oasis_internal_params = struct();
    oasis_internal_params.ar_model = 'ar1';
    oasis_internal_params.method = 'foopsi';
    oasis_internal_params.optimize_b = true;
    oasis_internal_params.optimize_pars = true;

    % To avoid running OASIS thousands of times, run deconvolution once per
    % validation recording, then threshold the inferred activity signal for
    % each parameter combination.
    fprintf('Running OASIS deconvolution once per validation recording...\n');

    oasis_cache = struct([]);

    for r = 1:length(validation_data)

        rec_data = validation_data(r);

        dummy_params = oasis_internal_params;
        dummy_params.oasis_threshold_z = 1.0;
        dummy_params.min_event_distance_sec = 0.10;

        oasis_result = run_oasis_model( ...
            rec_data.calcium_ds, ...
            rec_data.actual_fps, ...
            dummy_params);

        oasis_cache(r).s_oasis_z = oasis_result.s_oasis_z;
        oasis_cache(r).spikes_ds = rec_data.spikes_ds;
        oasis_cache(r).actual_fps = rec_data.actual_fps;
    end

    fprintf('OASIS deconvolution cache ready.\n');

    for p = 1:height(oasis_grid)

        oasis_threshold_z = oasis_grid.oasis_threshold_z(p);
        min_event_distance_sec = oasis_grid.min_event_distance_sec(p);

        params = oasis_internal_params;
        params.oasis_threshold_z = oasis_threshold_z;
        params.min_event_distance_sec = min_event_distance_sec;

        metrics_list = [];

        for r = 1:length(oasis_cache)

            s_oasis_z = oasis_cache(r).s_oasis_z;
            spikes_ds = oasis_cache(r).spikes_ds;
            actual_fps = oasis_cache(r).actual_fps;

            predicted_events = threshold_activity_signal( ...
                s_oasis_z, ...
                actual_fps, ...
                oasis_threshold_z, ...
                min_event_distance_sec);

            metrics = evaluate_event_prediction( ...
                predicted_events, ...
                spikes_ds, ...
                actual_fps, ...
                tolerance_sec, ...
                optimization_mode, ...
                burst_gap_sec);

            metrics_list = [metrics_list; metrics];
        end

        row = summarize_param_result( ...
            "OASIS", ...
            target_fps, ...
            optimization_mode, ...
            params, ...
            metrics_list, ...
            length(validation_data));

        search_results = [search_results; row];

        if mod(p, 25) == 0
            fprintf('  OASIS progress: %d/%d\n', p, height(oasis_grid));
        end
    end
end

%% Save full hyperparameter search results

search_output_file = fullfile(tables_dir, 'hyperparameter_search_results.csv');
writetable(search_results, search_output_file);

fprintf('\nSaved hyperparameter search results to:\n%s\n', search_output_file);

%% Select best parameters per model and FPS

best_params_table = select_best_params(search_results);

best_output_file = fullfile(tables_dir, 'best_model_params_by_fps.csv');
writetable(best_params_table, best_output_file);

fprintf('\nSaved best model parameters to:\n%s\n', best_output_file);

%% Print best parameters

fprintf('\n============================================================\n');
fprintf('BEST PARAMETERS BY MODEL AND FPS\n');
fprintf('============================================================\n');

disp(best_params_table);

fprintf('\nDone.\n');

%% Local helper functions

function grid = create_baseline_grid()
% create_baseline_grid
%
% Coarse development grid for SimpleBaseline.
% This is intentionally small so the optimization pipeline runs quickly.
% After seeing the best region, we can run a finer grid around it.

    smoothing_window_sec_list = [0.10, 0.20, 0.30];
    derivative_weight_list = [0.5, 0.7, 0.9];
    event_score_threshold_list = [0.6, 0.9, 1.2, 1.5];
    min_event_distance_sec_list = [0.05, 0.10, 0.20];

    grid = table();

    for s = 1:length(smoothing_window_sec_list)
        for w = 1:length(derivative_weight_list)
            for t = 1:length(event_score_threshold_list)
                for d = 1:length(min_event_distance_sec_list)

                    derivative_weight = derivative_weight_list(w);
                    amplitude_weight = 1 - derivative_weight;

                    row = table( ...
                        smoothing_window_sec_list(s), ...
                        derivative_weight, ...
                        amplitude_weight, ...
                        event_score_threshold_list(t), ...
                        min_event_distance_sec_list(d), ...
                        'VariableNames', { ...
                            'smoothing_window_sec', ...
                            'derivative_weight', ...
                            'amplitude_weight', ...
                            'event_score_threshold', ...
                            'min_event_distance_sec'});

                    grid = [grid; row];
                end
            end
        end
    end
end

function grid = create_oasis_grid()
% create_oasis_grid
%
% Coarse grid for OASIS event conversion.

    oasis_threshold_z_list = [0.75, 1.0, 1.25, 1.5, 2.0, 2.5];
    min_event_distance_sec_list = [0.05, 0.10, 0.15, 0.20];

    grid = table();

    for t = 1:length(oasis_threshold_z_list)
        for d = 1:length(min_event_distance_sec_list)

            row = table( ...
                oasis_threshold_z_list(t), ...
                min_event_distance_sec_list(d), ...
                'VariableNames', { ...
                    'oasis_threshold_z', ...
                    'min_event_distance_sec'});

            grid = [grid; row];
        end
    end
end

function validation_data = prepare_validation_data(validation_rows, data_dir, target_fps)
% prepare_validation_data
%
% Loads and downsamples all validation recordings for one target FPS.

    validation_data = struct([]);

    for i = 1:height(validation_rows)

        dataset_name = validation_rows.dataset_name(i);
        dataset_path = fullfile(data_dir, dataset_name);

        raw = load(dataset_path);

        if ~isfield(raw, 'data')
            error('File %s does not contain variable named "data".', dataset_name);
        end

        data = raw.data;

        recording_idx = validation_rows.recording_idx(i);
        rec = data{recording_idx};

        calcium = double(rec.calcium(:));
        spikes = double(rec.spikes(:));
        original_fps = double(rec.fps);

        ds = downsample_calcium_and_spikes( ...
            calcium, ...
            spikes, ...
            original_fps, ...
            target_fps);

        validation_data(i).dataset_name = dataset_name;
        validation_data(i).dataset_id = validation_rows.dataset_id(i);
        validation_data(i).recording_idx = recording_idx;
        validation_data(i).cell_num = validation_rows.cell_num(i);

        validation_data(i).calcium_ds = ds.calcium;
        validation_data(i).spikes_ds = ds.spikes;
        validation_data(i).actual_fps = ds.actual_fps;
    end
end

function predicted_events = threshold_activity_signal(activity_signal_z, fps, threshold_z, min_event_distance_sec)
% threshold_activity_signal
%
% Converts a continuous activity signal into binary predicted events.

    activity_signal_z = double(activity_signal_z(:));

    min_event_distance_samples = max(1, round(min_event_distance_sec * fps));

    [~, locs] = findpeaks(activity_signal_z, ...
        'MinPeakHeight', threshold_z, ...
        'MinPeakDistance', min_event_distance_samples);

    predicted_events = false(size(activity_signal_z));
    predicted_events(locs) = true;
end

function row = summarize_param_result(model_name, target_fps, optimization_mode, params, metrics_list, n_validation_recordings)
% summarize_param_result
%
% Aggregates validation metrics for one parameter set.

    F1_values = [metrics_list.F1]';
    precision_values = [metrics_list.precision]';
    recall_values = [metrics_list.recall]';
    median_abs_timing_values = [metrics_list.median_abs_timing_error_sec]';
    mean_signed_timing_values = [metrics_list.mean_signed_timing_error_sec]';
    timing_weighted_recall_values = [metrics_list.timing_weighted_recall]';

    % Default values for parameters that do not apply to every model.
    smoothing_window_sec = NaN;
    derivative_weight = NaN;
    amplitude_weight = NaN;
    event_score_threshold = NaN;
    baseline_min_event_distance_sec = NaN;

    ar_model = "";
    method = "";
    optimize_b = NaN;
    optimize_pars = NaN;
    oasis_threshold_z = NaN;
    oasis_min_event_distance_sec = NaN;

    if model_name == "SimpleBaseline"

        smoothing_window_sec = params.smoothing_window_sec;
        derivative_weight = params.derivative_weight;
        amplitude_weight = params.amplitude_weight;
        event_score_threshold = params.event_score_threshold;
        baseline_min_event_distance_sec = params.min_event_distance_sec;

    elseif model_name == "OASIS"

        ar_model = string(params.ar_model);
        method = string(params.method);
        optimize_b = double(params.optimize_b);
        optimize_pars = double(params.optimize_pars);
        oasis_threshold_z = params.oasis_threshold_z;
        oasis_min_event_distance_sec = params.min_event_distance_sec;
    end

    row = table( ...
        string(model_name), ...
        target_fps, ...
        string(optimization_mode), ...
        n_validation_recordings, ...
        mean(F1_values, 'omitnan'), ...
        median(F1_values, 'omitnan'), ...
        std(F1_values, 'omitnan'), ...
        mean(precision_values, 'omitnan'), ...
        mean(recall_values, 'omitnan'), ...
        mean(median_abs_timing_values, 'omitnan'), ...
        median(median_abs_timing_values, 'omitnan'), ...
        mean(mean_signed_timing_values, 'omitnan'), ...
        mean(timing_weighted_recall_values, 'omitnan'), ...
        median(timing_weighted_recall_values, 'omitnan'), ...
        smoothing_window_sec, ...
        derivative_weight, ...
        amplitude_weight, ...
        event_score_threshold, ...
        baseline_min_event_distance_sec, ...
        ar_model, ...
        method, ...
        optimize_b, ...
        optimize_pars, ...
        oasis_threshold_z, ...
        oasis_min_event_distance_sec, ...
        'VariableNames', { ...
            'model', ...
            'target_fps', ...
            'optimization_mode', ...
            'n_validation_recordings', ...
            'mean_F1', ...
            'median_F1', ...
            'std_F1', ...
            'mean_precision', ...
            'mean_recall', ...
            'mean_median_abs_timing_error_sec', ...
            'median_median_abs_timing_error_sec', ...
            'mean_signed_timing_error_sec', ...
            'mean_timing_weighted_recall', ...
            'median_timing_weighted_recall', ...
            'smoothing_window_sec', ...
            'derivative_weight', ...
            'amplitude_weight', ...
            'event_score_threshold', ...
            'baseline_min_event_distance_sec', ...
            'ar_model', ...
            'method', ...
            'optimize_b', ...
            'optimize_pars', ...
            'oasis_threshold_z', ...
            'oasis_min_event_distance_sec'});
end

function best_params_table = select_best_params(search_results)
% select_best_params
%
% Selects the best parameter set for each model and FPS.
%
% Primary criterion:
%   highest mean_F1
%
% Tie-breakers:
%   highest mean_timing_weighted_recall
%   lowest mean_median_abs_timing_error_sec

    models = unique(search_results.model, 'stable');
    fps_values = unique(search_results.target_fps, 'stable');

    best_params_table = table();

    for m = 1:length(models)

        model_name = models(m);

        for f = 1:length(fps_values)

            target_fps = fps_values(f);

            rows = search_results( ...
                search_results.model == model_name & ...
                search_results.target_fps == target_fps, :);

            if isempty(rows)
                continue;
            end

            sorted_rows = sortrows( ...
                rows, ...
                {'mean_F1', 'mean_timing_weighted_recall', 'mean_median_abs_timing_error_sec'}, ...
                {'descend', 'descend', 'ascend'});

            best_params_table = [best_params_table; sorted_rows(1, :)];
        end
    end
end