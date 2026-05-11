%% TEST_DOWNSAMPLE_ALL_NEURONS.m
% Block 4
% Goal:
%   Test downsample_calcium_and_spikes() on all neurons in one dataset file.
%
% This script checks whether the downsampling function works reliably for:
%   all neurons x all target sampling rates
%
% No models yet.
% No final analysis yet.

clear; clc; close all;

%% Add src folder to MATLAB path

addpath(fullfile('..', 'src'));

%% Load mock data

dataset_file = fullfile('..', 'data', 'MockData', 'data.1.train.preprocessed.mat');

raw = load(dataset_file);
data = raw.data;

%% Define target sampling rates

target_fps_list = [100, 50, 20, 10];

%% Print general information

fprintf('Testing downsampling on all neurons\n');
fprintf('Dataset file: %s\n', dataset_file);
fprintf('Number of neurons/recordings: %d\n', numel(data));
fprintf('Target sampling rates: ');
fprintf('%.0f ', target_fps_list);
fprintf('Hz\n\n');

%% Prepare summary counters

n_tests = 0;
n_passed = 0;
n_failed = 0;

%% Loop over all neurons and sampling rates

fprintf('Downsampling test summary:\n');
fprintf('------------------------------------------------------------------------------------------------------------------------\n');
fprintf('%8s %10s %12s %8s %10s %12s %12s %12s %12s\n', ...
    'Neuron', 'TargetHz', 'ActualHz', 'Factor', 'Samples', ...
    'Spikes', 'ActiveBins', 'MergedBins', 'MultiBins');
fprintf('------------------------------------------------------------------------------------------------------------------------\n');
for neuron_idx = 1:numel(data)

    rec = data{neuron_idx};

    calcium = double(rec.calcium(:));
    spikes = double(rec.spikes(:));
    original_fps = double(rec.fps);

    for r = 1:length(target_fps_list)

        target_fps = target_fps_list(r);
        n_tests = n_tests + 1;

        try
            % Run the downsampling function.
            ds = downsample_calcium_and_spikes( ...
                calcium, ...
                spikes, ...
                original_fps, ...
                target_fps);

            % Print one compact row for this neuron and sampling rate.
            fprintf('%8d %10.1f %12.4f %8d %10d %12.0f %12.0f %12.0f %12.0f\n', ...
                neuron_idx, ...
                ds.target_fps, ...
                ds.actual_fps, ...
                ds.downsample_factor, ...
                ds.n_downsampled_samples, ...
                ds.downsampled_spike_count, ...
                ds.downsampled_active_spike_bins, ...
                ds.active_bin_reduction, ...
                ds.downsampled_multi_spike_bins);
            fprintf('------------------------------------------------------------------------------------------------------------------------\n');

            n_passed = n_passed + 1;

        catch ME
            % If something fails, print exactly where the problem happened.
            fprintf('\nERROR during downsampling:\n');
            fprintf('Neuron index: %d\n', neuron_idx);
            fprintf('Target fps: %.1f Hz\n', target_fps);
            fprintf('Original fps: %.4f Hz\n', original_fps);
            fprintf('Error message:\n%s\n', ME.message);

            n_failed = n_failed + 1;

            % Stop immediately, because we want to fix the issue before continuing.
            error('Downsampling test failed. See details above.');
        end
    end
end

fprintf('---------------------------------------------------------------------------------------------------\n');

%% Final test summary

fprintf('\nFinal summary:\n');
fprintf('Total tests:  %d\n', n_tests);
fprintf('Passed tests: %d\n', n_passed);
fprintf('Failed tests: %d\n', n_failed);

if n_failed == 0
    fprintf('\nAll downsampling tests passed successfully.\n');
else
    fprintf('\nSome downsampling tests failed.\n');
end