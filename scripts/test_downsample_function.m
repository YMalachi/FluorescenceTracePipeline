%% 03_TEST_DOWNSAMPLE_FUNCTION.m
% Goal:
%   Test the downsample_calcium_and_spikes() function on one neuron
%   and verify that it works for 100, 50, 20, and 10 Hz.
%
% This is not the main pipeline yet.
% It is only a function test.

clear; clc; close all;

%% Add src folder to MATLAB path

% This lets MATLAB find downsample_calcium_and_spikes.m
addpath(fullfile('..', 'src'));

%% Load mock data

dataset_file = fullfile('..', 'data', 'MockData', 'data.1.train.preprocessed.mat');

raw = load(dataset_file);
data = raw.data;

%% Choose one neuron for testing

neuron_idx = 1;
rec = data{neuron_idx};

calcium = double(rec.calcium(:));
spikes = double(rec.spikes(:));
original_fps = double(rec.fps);

%% Define target sampling rates

target_fps_list = [100, 50, 20, 10];

fprintf('Testing downsample_calcium_and_spikes() on neuron %d\n', neuron_idx);
fprintf('Original fps: %.4f Hz\n', original_fps);
fprintf('Original samples: %d\n', length(calcium));
fprintf('Original spike count: %.0f\n\n', sum(spikes));

%% Run function for all target rates

fprintf('Function test summary:\n');
fprintf('--------------------------------------------------------------------------------------\n');
fprintf('%10s %12s %12s %12s %12s %12s\n', ...
    'Target Hz', 'Actual Hz', 'Factor', 'Samples', 'Spikes', 'Removed');
fprintf('--------------------------------------------------------------------------------------\n');

for r = 1:length(target_fps_list)

    target_fps = target_fps_list(r);

    % Run the function we want to test.
    ds = downsample_calcium_and_spikes(calcium, spikes, original_fps, target_fps);

    % Print compact result summary.
    fprintf('%10.1f %12.4f %12d %12d %12.0f %12.0f\n', ...
        ds.target_fps, ...
        ds.actual_fps, ...
        ds.downsample_factor, ...
        ds.n_downsampled_samples, ...
        ds.downsampled_spike_count, ...
        ds.removed_spike_count);

    % Extra check inside the test script.
    % The function already checks this internally, but we print confirmation here.
    if ds.trimmed_spike_count ~= ds.downsampled_spike_count
        error('Test failed at %.1f Hz: spike count was not preserved.', target_fps);
    end
end

fprintf('--------------------------------------------------------------------------------------\n');
fprintf('\nAll downsampling function tests passed.\n');

%% Visualize downsampled traces from the function

% This section plots the output of downsample_calcium_and_spikes()
% for all target sampling rates.
%
% It is only for visual inspection:
%   - Does the calcium trace still look reasonable?
%   - Are spike bins displayed at the expected times?
%   - How much detail is lost at lower sampling rates?

plot_duration_sec = 20;

figure('Color', 'w');

for r = 1:length(target_fps_list)

    % Choose target sampling rate
    target_fps = target_fps_list(r);

    % Run downsampling function
    ds = downsample_calcium_and_spikes(calcium, spikes, original_fps, target_fps);

    % Extract downsampled calcium, spikes, and actual fps
    calcium_ds = ds.calcium;
    spikes_ds = ds.spikes;
    fps = ds.actual_fps;

    % Choose how many samples to plot
    n_plot_samples = min(round(plot_duration_sec * fps), length(calcium_ds));

    % Extract plotting window
    calcium_plot = calcium_ds(1:n_plot_samples);
    spikes_plot = spikes_ds(1:n_plot_samples);

    % Create time axis in seconds
    t = (0:n_plot_samples-1) / fps;

    % Z-score calcium only for visualization
    calcium_z = (calcium_plot - mean(calcium_plot)) / std(calcium_plot);

    % Plot only bins that contain at least one spike
    spike_idx = spikes_plot > 0;
    spike_times = t(spike_idx);

    % Place spike markers below the calcium trace
    spike_y_value = -3;
    spike_y = spike_y_value * ones(size(spike_times));

    % Create one subplot for each sampling rate
    subplot(length(target_fps_list), 1, r);

    plot(t, calcium_z, 'LineWidth', 1.2);
    hold on;

    plot(spike_times, spike_y, ...
        'o', ...
        'MarkerSize', 4, ...
        'LineStyle', 'none');

    xlabel('Time (s)');
    ylabel('Z-score');

    title(sprintf('Neuron %d — Target %.1f Hz, actual %.4f Hz', ...
        neuron_idx, ds.target_fps, ds.actual_fps));

    grid on;
end