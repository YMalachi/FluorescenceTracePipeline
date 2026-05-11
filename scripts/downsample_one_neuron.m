%% DOWNSAMPLE_ONE_NEURON_ALL_RATES.m
% Block 3
% Goal:
%   Downsample one neuron's calcium trace and binned spike train
%   to several sampling rates: 100 Hz, 50 Hz, 20 Hz, and 10 Hz.
%
% Important:
%   Calcium is averaged inside each new time bin.
%   Spikes are summed inside each new time bin, not averaged.
%
% Why?
%   Calcium is a continuous fluorescence signal.
%   Spikes are event counts, so we must preserve the total number of spikes.

clear; clc; close all;

%% Load data

dataset_file = fullfile('..', 'data', 'MockData', 'data.1.train.preprocessed.mat');

raw = load(dataset_file);
data = raw.data;

%% Choose neuron

neuron_idx = 1;

rec = data{neuron_idx};

calcium = double(rec.calcium(:));
spikes = double(rec.spikes(:));
original_fps = double(rec.fps);

fprintf('Neuron index: %d\n', neuron_idx);
fprintf('Original sampling rate: %.4f Hz\n', original_fps);
fprintf('Original calcium samples: %d\n', length(calcium));
fprintf('Original spike count: %.0f\n\n', sum(spikes));

%% Define target sampling rates

% 100 Hz is included as the original condition.
% Since the original fps is ~100 Hz, the downsample factor for 100 Hz
% should be 1.
target_fps_list = [100, 50, 20, 10];

%% Create storage for all downsampled versions

% Each entry in all_data will contain one sampling-rate version.
% This lets us plot and inspect all rates later in the script.
all_data = struct([]);

%% Downsample the neuron to each target sampling rate

fprintf('Downsampling summary:\n');
fprintf('--------------------------------------------------------------------------------------\n');
fprintf('%10s %12s %12s %12s %12s %12s\n', ...
    'Target Hz', 'Actual Hz', 'Factor', 'Samples', 'Spikes', 'Duration(s)');
fprintf('--------------------------------------------------------------------------------------\n');

for r = 1:length(target_fps_list)

    %% Select target sampling rate

    target_fps = target_fps_list(r);

    %% Calculate downsampling factor

    % For this first version, we use simple integer downsampling.
    % Examples:
    %   100 Hz -> 100 Hz: factor 1
    %   100 Hz -> 50 Hz:  factor 2
    %   100 Hz -> 20 Hz:  factor 5
    %   100 Hz -> 10 Hz:  factor 10
    downsample_factor = round(original_fps / target_fps);

    % The actual target fps may be slightly different because original_fps
    % is not exactly 100.0000 Hz.
    actual_target_fps = original_fps / downsample_factor;

    %% Trim signals to full downsampling blocks

    % We trim the end of the recording so the signal length is divisible
    % by the downsampling factor.
    n_original_samples = length(calcium);
    n_blocks = floor(n_original_samples / downsample_factor);
    n_trimmed_samples = n_blocks * downsample_factor;

    calcium_trimmed = calcium(1:n_trimmed_samples);
    spikes_trimmed = spikes(1:n_trimmed_samples);

    %% Reshape into blocks

    % Each column is one new lower-rate time bin.
    % Each column contains downsample_factor original samples.
    calcium_blocks = reshape(calcium_trimmed, downsample_factor, n_blocks);
    spike_blocks = reshape(spikes_trimmed, downsample_factor, n_blocks);

    %% Downsample

    % Calcium: average within each new time bin.
    calcium_ds = mean(calcium_blocks, 1)';

    % Spikes: sum within each new time bin.
    % This preserves spike counts after trimming.
    spikes_ds = sum(spike_blocks, 1)';

    %% Store results

    all_data(r).target_fps = target_fps;
    all_data(r).actual_fps = actual_target_fps;
    all_data(r).downsample_factor = downsample_factor;

    all_data(r).calcium = calcium_ds;
    all_data(r).spikes = spikes_ds;

    all_data(r).n_original_samples = n_original_samples;
    all_data(r).n_trimmed_samples = n_trimmed_samples;
    all_data(r).n_downsampled_samples = length(calcium_ds);

    all_data(r).original_spike_count = sum(spikes);
    all_data(r).trimmed_spike_count = sum(spikes_trimmed);
    all_data(r).downsampled_spike_count = sum(spikes_ds);

    all_data(r).original_duration_sec = length(calcium) / original_fps;
    all_data(r).downsampled_duration_sec = length(calcium_ds) / actual_target_fps;

    %% Print compact summary

    fprintf('%10.1f %12.4f %12d %12d %12.0f %12.2f\n', ...
        target_fps, ...
        actual_target_fps, ...
        downsample_factor, ...
        length(calcium_ds), ...
        sum(spikes_ds), ...
        length(calcium_ds) / actual_target_fps);
end

fprintf('--------------------------------------------------------------------------------------\n');

%% Spike preservation checks

% For each sampling rate, the downsampled spike count should match
% the trimmed spike count exactly.
%
% The original spike count may differ only if spikes were located in
% the tiny trimmed ending of the recording.

fprintf('\nSpike preservation checks:\n');

for r = 1:length(all_data)

    fprintf('Target %.1f Hz: trimmed spikes = %.0f, downsampled spikes = %.0f\n', ...
        all_data(r).target_fps, ...
        all_data(r).trimmed_spike_count, ...
        all_data(r).downsampled_spike_count);
end

%% Visual check: calcium and spikes at all sampling rates

% This plot lets us visually compare how the signal changes when moving
% from 100 Hz to 50, 20, and 10 Hz.
%
% This is still only for inspection.
% It is not a final result figure yet.

plot_duration_sec = 20;

figure('Color', 'w');

for r = 1:length(all_data)

    %% Extract one sampling-rate version

    calcium_ds = all_data(r).calcium;
    spikes_ds = all_data(r).spikes;
    fps = all_data(r).actual_fps;

    %% Select short plotting window

    n_plot_samples = min(round(plot_duration_sec * fps), length(calcium_ds));

    calcium_plot = calcium_ds(1:n_plot_samples);
    spikes_plot = spikes_ds(1:n_plot_samples);

    % Time axis in seconds.
    t = (0:n_plot_samples-1) / fps;

    %% Normalize calcium for visualization

    % Z-scoring is used only for plotting.
    % It does not change the actual downsampled data.
    calcium_z = (calcium_plot - mean(calcium_plot)) / std(calcium_plot);

    %% Prepare spike markers

    % Plot only bins that contain at least one spike.
    % Plotting zero-spike bins would make the graph unreadable.
    spike_idx = spikes_plot > 0;
    spike_times = t(spike_idx);

    % Place spike markers below the calcium trace.
    spike_y_value = -3;
    spike_y = spike_y_value * ones(size(spike_times));

    %% Plot

    subplot(length(all_data), 1, r);

    plot(t, calcium_z, 'LineWidth', 1.2);
    hold on;

    plot(spike_times, spike_y, ...
        'o', ...
        'MarkerSize', 4, ...
        'LineStyle', 'none');

    xlabel('Time (s)');
    ylabel('Z-score');

    title(sprintf('Neuron %d — Target %.1f Hz, actual %.4f Hz', ...
        neuron_idx, all_data(r).target_fps, fps));

    grid on;
end