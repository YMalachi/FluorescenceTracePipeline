%% TEST_SIMPLE_BASELINE_FUNCTION.m
% Goal:
%   Test run_simple_baseline() on one neuron across several sampling rates.
%
% This checks that the function runs correctly and returns outputs with
% the expected size.

clear; clc; close all;

%% Add src folder

addpath(fullfile('..', 'src'));

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

%% Baseline parameters

params = struct();

params.smoothing_window_sec = 0.2;
params.derivative_weight = 0.7;
params.amplitude_weight = 0.3;
params.event_score_threshold = 0.9;
params.min_event_distance_sec = 0.10;

%% Test sampling rates

target_fps_list = [100, 50, 20, 10];

fprintf('Testing run_simple_baseline() on neuron %d\n\n', neuron_idx);

fprintf('---------------------------------------------------------------\n');
fprintf('%10s %12s %12s %18s\n', ...
    'Target Hz', 'Actual Hz', 'Samples', 'Predicted Events');
fprintf('---------------------------------------------------------------\n');

for r = 1:length(target_fps_list)

    target_fps = target_fps_list(r);

    % Downsample first
    ds = downsample_calcium_and_spikes(calcium, spikes, original_fps, target_fps);

    calcium_ds = ds.calcium;
    fps = ds.actual_fps;

    % Run baseline model
    result = run_simple_baseline(calcium_ds, fps, params);

    % Basic output checks
    if length(result.predicted_events) ~= length(calcium_ds)
        error('Predicted events length does not match calcium length at %.1f Hz.', target_fps);
    end

    if length(result.event_score) ~= length(calcium_ds)
        error('Event score length does not match calcium length at %.1f Hz.', target_fps);
    end

    if any(isnan(result.event_score)) || any(isinf(result.event_score))
        error('Event score contains NaN or Inf values at %.1f Hz.', target_fps);
    end

    fprintf('%10.1f %12.4f %12d %18d\n', ...
        target_fps, ...
        fps, ...
        length(calcium_ds), ...
        result.n_predicted_events);
end

fprintf('---------------------------------------------------------------\n');
fprintf('\nAll simple baseline function tests passed.\n');