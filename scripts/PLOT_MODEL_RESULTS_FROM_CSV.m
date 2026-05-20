%% PLOT_MODEL_RESULTS_FROM_CSV.m
% Goal:
%   Visualize model comparison results from the detailed CSV table.
%
% Input:
%   results/tables/model_results_detailed.csv
%
% Output figures:
%   1. Mean F1 vs sampling rate
%   2. Mean timing-weighted recall vs sampling rate
%   3. Precision vs recall scatter
%   4. F1 boxplots across recordings
%
% Important:
%   These plots are diagnostic/development plots.
%   They are not final results yet because model hyperparameters
%   have not been optimized on validation data.

clear; clc; close all;

%% Paths

input_file = fullfile('..', 'results', 'tables', 'model_results_detailed.csv');
figures_dir = fullfile('..', 'results', 'figures');

if ~exist(figures_dir, 'dir')
    mkdir(figures_dir);
end

%% Load results table

if ~isfile(input_file)
    error('Results file not found: %s', input_file);
end

results = readtable(input_file);
%% Convert text columns to string

% readtable sometimes loads text columns as cell arrays of char.
% Converting them to string makes comparisons with == work reliably.

text_columns = {'model', 'evaluation_mode', 'status', 'dataset_name', 'error_message'};

for c = 1:length(text_columns)

    col = text_columns{c};

    if ismember(col, results.Properties.VariableNames)
        results.(col) = string(results.(col));
    end
end

fprintf('Loaded results table:\n%s\n', input_file);
fprintf('Number of rows: %d\n\n', height(results));

%% Keep only successful rows

if ismember('status', results.Properties.VariableNames)
    results = results(results.status == "ok", :);
end

fprintf('Number of successful result rows: %d\n\n', height(results));

%% Basic settings

% Display FPS in this order so the plots show degradation from high to low FPS.
target_fps_order = [100, 50, 20, 10];

models = unique(results.model, 'stable');
evaluation_modes = unique(results.evaluation_mode, 'stable');

fprintf('Models found:\n');
disp(models);

fprintf('Evaluation modes found:\n');
disp(evaluation_modes);

%% Create summary table for plotting

summary = summarize_for_plotting(results, target_fps_order);

%% Plot 1: Mean F1 vs sampling rate

plot_metric_vs_fps( ...
    summary, ...
    target_fps_order, ...
    models, ...
    evaluation_modes, ...
    'mean_F1', ...
    'sem_F1', ...
    'Mean F1', ...
    'Mean F1 vs Sampling Rate');

save_current_figure(figures_dir, 'mean_F1_vs_sampling_rate');

%% Plot 2: Mean timing-weighted recall vs sampling rate

plot_metric_vs_fps( ...
    summary, ...
    target_fps_order, ...
    models, ...
    evaluation_modes, ...
    'mean_timing_weighted_recall', ...
    'sem_timing_weighted_recall', ...
    'Mean timing-weighted recall', ...
    'Timing-weighted Recall vs Sampling Rate');

save_current_figure(figures_dir, 'timing_weighted_recall_vs_sampling_rate');

%% Plot 3: Precision vs recall scatter

figure('Color', 'w');
hold on;

for mode_idx = 1:length(evaluation_modes)

    mode = evaluation_modes(mode_idx);

    subplot(1, length(evaluation_modes), mode_idx);
    hold on;

    mode_rows = summary(summary.evaluation_mode == mode, :);

    for model_idx = 1:length(models)

        model_name = models(model_idx);
        model_rows = mode_rows(mode_rows.model == model_name, :);

        scatter(model_rows.mean_recall, model_rows.mean_precision, 60, 'filled');

        % Add FPS labels next to points.
        for i = 1:height(model_rows)
            text(model_rows.mean_recall(i), ...
                 model_rows.mean_precision(i), ...
                 sprintf(' %.0f Hz', model_rows.target_fps(i)), ...
                 'FontSize', 8);
        end
    end

    xlabel('Mean recall');
    ylabel('Mean precision');
    title(sprintf('Precision vs Recall — %s', string(mode)));
    legend(string(models), 'Location', 'best');
    grid on;
    xlim([0 1]);
    ylim([0 1]);
end

save_current_figure(figures_dir, 'precision_vs_recall_scatter');

%% Plot 4: F1 boxplots across recordings

% This plot shows variability across neurons/recordings.
% We create one figure per evaluation mode to avoid overcrowding.

for mode_idx = 1:length(evaluation_modes)

    mode = evaluation_modes(mode_idx);

    mode_results = results(results.evaluation_mode == mode, :);

    figure('Color', 'w');

    f1_values = [];
    group_labels = {};

    for model_idx = 1:length(models)

        model_name = models(model_idx);

        for fps_idx = 1:length(target_fps_order)

            target_fps = target_fps_order(fps_idx);

            rows = mode_results( ...
                mode_results.model == model_name & ...
                mode_results.target_fps == target_fps, :);

            values = rows.F1;

            f1_values = [f1_values; values];

            label = sprintf('%s\n%.0f Hz', string(model_name), target_fps);
            group_labels = [group_labels; repmat({label}, length(values), 1)];
        end
    end

    boxplot(f1_values, group_labels);
    ylabel('F1');
    title(sprintf('F1 distribution across recordings — %s', string(mode)));
    grid on;
    ylim([0 1]);

    save_current_figure(figures_dir, sprintf('F1_boxplot_%s', string(mode)));
end

%% Print compact summary to console

fprintf('\nCompact summary:\n');
fprintf('-------------------------------------------------------------------------------------------------------------\n');
fprintf('%18s %10s %15s %10s %10s %10s %14s %14s\n', ...
    'Model', 'FPS', 'Mode', 'MeanF1', 'MedF1', 'MeanRec', 'MedAbsErr', 'TimeWRecall');
fprintf('-------------------------------------------------------------------------------------------------------------\n');

for i = 1:height(summary)

    fprintf('%18s %10.1f %15s %10.3f %10.3f %10.3f %14.4f %14.4f\n', ...
        string(summary.model(i)), ...
        summary.target_fps(i), ...
        string(summary.evaluation_mode(i)), ...
        summary.mean_F1(i), ...
        summary.median_F1(i), ...
        summary.mean_recall(i), ...
        summary.mean_median_abs_timing_error_sec(i), ...
        summary.mean_timing_weighted_recall(i));
end

fprintf('-------------------------------------------------------------------------------------------------------------\n');

fprintf('\nFigures saved to:\n%s\n', figures_dir);

%% Local helper functions

function summary = summarize_for_plotting(results, target_fps_order)
% summarize_for_plotting
%
% Aggregates detailed results by:
%   model x target_fps x evaluation_mode
%
% The summary table is used for line plots and scatter plots.

    models = unique(results.model, 'stable');
    modes = unique(results.evaluation_mode, 'stable');

    summary = table();

    for model_idx = 1:length(models)

        model_name = models(model_idx);

        for fps_idx = 1:length(target_fps_order)

            target_fps = target_fps_order(fps_idx);

            for mode_idx = 1:length(modes)

                mode = modes(mode_idx);

                rows = results( ...
                    results.model == model_name & ...
                    results.target_fps == target_fps & ...
                    results.evaluation_mode == mode, :);

                if isempty(rows)
                    continue;
                end

                F1_values = rows.F1;
                precision_values = rows.precision;
                recall_values = rows.recall;
                median_abs_timing_values = rows.median_abs_timing_error_sec;
                mean_signed_timing_values = rows.mean_signed_timing_error_sec;

                if ismember('timing_weighted_recall', rows.Properties.VariableNames)
                    timing_weighted_recall_values = rows.timing_weighted_recall;
                else
                    timing_weighted_recall_values = NaN(height(rows), 1);
                end

                n = height(rows);

                row = table( ...
                    model_name, ...
                    target_fps, ...
                    mode, ...
                    n, ...
                    mean(F1_values, 'omitnan'), ...
                    median(F1_values, 'omitnan'), ...
                    std(F1_values, 'omitnan'), ...
                    sem(F1_values), ...
                    mean(precision_values, 'omitnan'), ...
                    median(precision_values, 'omitnan'), ...
                    mean(recall_values, 'omitnan'), ...
                    median(recall_values, 'omitnan'), ...
                    mean(median_abs_timing_values, 'omitnan'), ...
                    median(median_abs_timing_values, 'omitnan'), ...
                    mean(mean_signed_timing_values, 'omitnan'), ...
                    median(mean_signed_timing_values, 'omitnan'), ...
                    mean(timing_weighted_recall_values, 'omitnan'), ...
                    median(timing_weighted_recall_values, 'omitnan'), ...
                    sem(timing_weighted_recall_values), ...
                    'VariableNames', { ...
                        'model', ...
                        'target_fps', ...
                        'evaluation_mode', ...
                        'n_recordings', ...
                        'mean_F1', ...
                        'median_F1', ...
                        'std_F1', ...
                        'sem_F1', ...
                        'mean_precision', ...
                        'median_precision', ...
                        'mean_recall', ...
                        'median_recall', ...
                        'mean_median_abs_timing_error_sec', ...
                        'median_median_abs_timing_error_sec', ...
                        'mean_signed_timing_error_sec', ...
                        'median_signed_timing_error_sec', ...
                        'mean_timing_weighted_recall', ...
                        'median_timing_weighted_recall', ...
                        'sem_timing_weighted_recall'});

                summary = [summary; row];
            end
        end
    end
end

function plot_metric_vs_fps(summary, target_fps_order, models, evaluation_modes, metric_name, sem_name, y_label_text, plot_title)
% plot_metric_vs_fps
%
% Plots one metric across sampling rates.
%
% One subplot per evaluation mode.
% One line per model.

    figure('Color', 'w');

    x = 1:length(target_fps_order);
    x_labels = string(target_fps_order);

    for mode_idx = 1:length(evaluation_modes)

        mode = evaluation_modes(mode_idx);

        subplot(1, length(evaluation_modes), mode_idx);
        hold on;

        for model_idx = 1:length(models)

            model_name = models(model_idx);

            y = NaN(size(target_fps_order));
            e = NaN(size(target_fps_order));

            for fps_idx = 1:length(target_fps_order)

                target_fps = target_fps_order(fps_idx);

                row = summary( ...
                    summary.model == model_name & ...
                    summary.target_fps == target_fps & ...
                    summary.evaluation_mode == mode, :);

                if ~isempty(row)
                    y(fps_idx) = row.(metric_name);
                    e(fps_idx) = row.(sem_name);
                end
            end

            errorbar(x, y, e, '-o', ...
                'LineWidth', 1.5, ...
                'MarkerSize', 6);
        end

        xlabel('Sampling rate');
        ylabel(y_label_text);
        title(sprintf('%s — %s', plot_title, string(mode)));

        xticks(x);
        xticklabels(x_labels);

        ylim([0 1]);
        grid on;
        legend(string(models), 'Location', 'best');
    end
end

function save_current_figure(figures_dir, base_name)
% save_current_figure
%
% Saves the current figure as both PNG and FIG.

    png_file = fullfile(figures_dir, sprintf('%s.png', base_name));
    fig_file = fullfile(figures_dir, sprintf('%s.fig', base_name));

    saveas(gcf, png_file);
    saveas(gcf, fig_file);

    fprintf('Saved figure:\n%s\n', png_file);
end

function value = sem(values)
% sem
%
% Standard error of the mean, ignoring NaN values.

    values = values(~isnan(values));

    if isempty(values)
        value = NaN;
    else
        value = std(values) / sqrt(length(values));
    end
end