clear; clc;

% Path to the mock data file.
dataset_file = fullfile('..', 'data','MockData', 'data.1.train.preprocessed.mat');
raw = load(dataset_file);

% Check which variables exist inside the loaded .mat file.
% For this dataset, we expect to see a variable named 'data'.
disp('Variables inside the file:');
disp(fieldnames(raw));

% Extract the main data variable.
% In this dataset, data is a cell array where each cell is one recording/neuron.
data = raw.data;

fprintf('Number of recordings in this file: %d\n', numel(data));

% Take the first recording only to inspect its internal structure.
% Each recording should contain calcium trace, spike train, exact spike times, and fps.
first_recording = data{1};

disp('Fields inside first recording:');
disp(fieldnames(first_recording));

%% Inspect one neuron numerically

% Choose which neuron/recording to inspect.
neuron_idx = 1;
rec = data{neuron_idx};

% Convert signals to column vectors.
% This makes later calculations more consistent.
calcium = rec.calcium(:);

% Convert spikes to double so numerical operations like sum() are safe and clear.
% rec.spikes is a binned spike train: each sample/bin may contain 0, 1, or more spikes.
spikes = double(rec.spikes(:));

% spike_times contains the exact spike times in milliseconds.
spike_times = rec.spike_times(:);

% fps is the sampling rate of the calcium and binned spike signals.
fps = double(rec.fps);

% Basic recording statistics.
n_samples = length(calcium);
duration_sec = n_samples / fps;

% Total number of spikes in the binned spike train.
n_spikes = sum(spikes);

% Mean firing rate across the full recording.
firing_rate_hz = n_spikes / duration_sec;

% Print a compact numerical summary for the selected neuron.
fprintf('\nNeuron %d numerical summary:\n', neuron_idx);
fprintf('---------------------------------\n');
fprintf('Number of calcium samples: %d\n', n_samples);
fprintf('Sampling rate: %.4f Hz\n', fps);
fprintf('Duration: %.2f seconds\n', duration_sec);
fprintf('Number of binned spikes: %.0f\n', n_spikes);
fprintf('Number of exact spike times: %d\n', length(spike_times));
fprintf('Mean firing rate: %.3f Hz\n', firing_rate_hz);
fprintf('---------------------------------\n');

%% Inspect all neurons numerically

% This section loops over all recordings in the current file.
% The goal is to quickly compare recording length, spike count,
% and firing rate across neurons.

fprintf('\nSummary of all neurons in this file:\n');
fprintf('-------------------------------------------------------------\n');
fprintf('%6s %12s %12s %12s %12s\n', ...
    'Index', 'Samples', 'Duration(s)', 'Spikes', 'Rate(Hz)');
fprintf('-------------------------------------------------------------\n');

for i = 1:numel(data)

    % Extract one recording
    rec = data{i};

    % Convert calcium and spikes to column vectors for consistency
    calcium = rec.calcium(:);
    spikes = double(rec.spikes(:));

    % Sampling rate for this recording
    fps = double(rec.fps);

    % Basic recording statistics
    n_samples = length(calcium);
    duration_sec = n_samples / fps;
    n_spikes = sum(spikes);
    firing_rate_hz = n_spikes / duration_sec;

    % Print one row in the summary table
    fprintf('%6d %12d %12.2f %12.0f %12.3f\n', ...
        i, n_samples, duration_sec, n_spikes, firing_rate_hz);
end

fprintf('-------------------------------------------------------------\n');

%% Basic sanity checks for all neurons

% This section checks that every recording has the fields we expect,
% and that the calcium trace and binned spike train have matching lengths.
% These checks help catch data-format problems early.

fprintf('\nRunning basic sanity checks...\n');

required_fields = {'calcium', 'spikes', 'spike_times', 'fps'};

all_checks_passed = true;

for i = 1:numel(data)

    rec = data{i};

    %% Check required fields

    for f = 1:numel(required_fields)

        field_name = required_fields{f};

        if ~isfield(rec, field_name)
            fprintf('Recording %d is missing field: %s\n', i, field_name);
            all_checks_passed = false;
        end
    end

    %% Check signal lengths

    % Skip length checks if one of the required fields is missing.
    if ~isfield(rec, 'calcium') || ~isfield(rec, 'spikes')
        continue;
    end

    calcium = rec.calcium(:);
    spikes = rec.spikes(:);

    if length(calcium) ~= length(spikes)
        fprintf('Recording %d has mismatched calcium/spike lengths.\n', i);
        fprintf('  Calcium length: %d\n', length(calcium));
        fprintf('  Spikes length:  %d\n', length(spikes));
        all_checks_passed = false;
    end

    %% Check sampling rate

    if isfield(rec, 'fps')

        fps = double(rec.fps);

        % The dataset description says the traces were resampled to ~100 Hz.
        % We allow a small range because the stored fps may not be exactly 100.
        if fps < 95 || fps > 105
            fprintf('Recording %d has unusual fps: %.4f Hz\n', i, fps);
            all_checks_passed = false;
        end
    end

    %% Check for invalid numerical values

    % NaN or Inf values would break many downstream operations.
    if any(isnan(double(calcium))) || any(isinf(double(calcium)))
        fprintf('Recording %d calcium contains NaN or Inf values.\n', i);
        all_checks_passed = false;
    end

    if any(isnan(double(spikes))) || any(isinf(double(spikes)))
        fprintf('Recording %d spikes contain NaN or Inf values.\n', i);
        all_checks_passed = false;
    end
end

%% Final sanity-check message

if all_checks_passed
    fprintf('All basic sanity checks passed.\n');
else
    fprintf('Some sanity checks failed. Review the messages above.\n');
end

%% Plot one neuron: calcium trace and ground-truth spikes

% This section visualizes one selected neuron.
% We plot only a short time window so the spike markers are readable.

neuron_idx = 1;
plot_duration_sec = 20;

rec = data{neuron_idx};

calcium = double(rec.calcium(:));
spikes = double(rec.spikes(:));
fps = double(rec.fps);

% Number of samples to plot
n_plot_samples = min(round(plot_duration_sec * fps), length(calcium));

% Time axis in seconds
t = (0:n_plot_samples-1) / fps;

% Extract the selected time window
calcium_plot = calcium(1:n_plot_samples);
spikes_plot = spikes(1:n_plot_samples);

% Z-score calcium for visualization.
% This makes the scale easier to interpret across different neurons.
calcium_z = (calcium_plot - mean(calcium_plot)) / std(calcium_plot);

% Find only bins that contain spikes.
% This avoids drawing thousands of zero-valued stems.
spike_idx = spikes_plot > 0;
spike_times_plot = t(spike_idx);

% Use a fixed y-value for spike markers so they appear below the calcium trace.
spike_y_value = -3;
spike_y_plot = spike_y_value * ones(size(spike_times_plot));

figure('Color', 'w');

plot(t, calcium_z, 'LineWidth', 1.2);
hold on;

plot(spike_times_plot, spike_y_plot, ...
    'o', ...
    'MarkerSize', 4, ...
    'MarkerFaceColor', [0.8500 0.3250 0.0980], ...
    'MarkerEdgeColor', [0.8500 0.3250 0.0980], ...
    'LineStyle', 'none');

xlabel('Time (s)');
ylabel('Z-scored fluorescence / spike markers');
title(sprintf('Neuron %d: calcium trace and ground-truth spikes', neuron_idx));

legend({'Calcium trace', 'Ground-truth spikes'}, 'Location', 'best');
grid on;